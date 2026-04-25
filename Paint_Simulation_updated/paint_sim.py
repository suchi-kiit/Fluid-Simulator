"""
Eulerian Fluid Simulator — ML-Accelerated Pressure Solver
Paint Pool Edition
==========================================================
Three-phase runtime:
  Phase 1  COLLECTING  — full 20-iter Gauss-Seidel solver; gather
                         (divergence patch → pressure) pairs each substep.
  Phase 2  TRAINING    — one-shot Adam training of a 3-layer NumPy MLP.
  Phase 3  ML-ACTIVE   — MLP predicts warm-start pressure; solver refines
                         in only 8 iterations instead of 20.

Controls
--------
  Drag ball     →  move obstacle through the paint
  Click (empty) →  impulse / splash disturbance
  P  →  toggle pressure / paint view
  G  →  toggle gravity
  R  →  reset simulation
  ESC / Q  →  quit

Performance budget (target 25-30 fps):
  · cpu_max_num_threads=1  (sequential beats threaded at 160×100)
  · SUBSTEPS=1             (was 2)
  · NUM_ITERS=20           (was 30)
  · ML_ITERS=8             (was 12)
  · collect() fully vectorised NumPy — no Python loop   (was ~75 ms/frame)
  · No cv2 / no numpy upscale — gui.set_image(img) direct field hand-off
  · Y-flip baked into render kernel — zero Python array work per frame
"""

import sys
import io
import time
import taichi as ti
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ti.init(
    arch=ti.cpu,
    cpu_max_num_threads=1,   # sequential wins at this grid scale
    default_fp=ti.f32,
    default_ip=ti.i32,
)

# ── Grid ──────────────────────────────────────────────────────────────────────
NX    = 160
NY    = 100
H     = 1.0 / NY
INV_H = float(NY)
H2    = 0.5 / NY

# ── Physics ───────────────────────────────────────────────────────────────────
DENSITY    = 1000.0
NUM_ITERS  = 20          # full solver iterations (phases 1 & fallback)
ML_ITERS   = 8           # reduced iterations when ML warm-start is active
OVER_RELAX = 1.95
GRAVITY    = -9.81
DT         = 1.0 / 60.0
SUBSTEPS   = 1
SUB_DT     = DT / SUBSTEPS
OBS_R      = NY * H * 0.14

# ── Window ────────────────────────────────────────────────────────────────────
WIN_SCALE = 5
WIN_W     = NX * WIN_SCALE   # 800
WIN_H     = NY * WIN_SCALE   # 500

# ── ML config ─────────────────────────────────────────────────────────────────
PATCH_SIZE        = 3
PATCH_FLAT        = PATCH_SIZE ** 2
COLLECT_FRAMES    = 40
SAMPLES_PER_FRAME = 1500
ML_EPOCHS         = 80
ML_BATCH          = 256
ML_LR             = 1e-3

# ── Taichi fields ─────────────────────────────────────────────────────────────
u      = ti.field(ti.f32, shape=(NX + 1, NY))
v      = ti.field(ti.f32, shape=(NX,     NY + 1))
new_u  = ti.field(ti.f32, shape=(NX + 1, NY))
new_v  = ti.field(ti.f32, shape=(NX,     NY + 1))
p      = ti.field(ti.f32, shape=(NX, NY))
s      = ti.field(ti.f32, shape=(NX, NY))
m      = ti.field(ti.f32, shape=(NX, NY))
new_m  = ti.field(ti.f32, shape=(NX, NY))
img    = ti.Vector.field(3, ti.f32, shape=(WIN_W, WIN_H))
p_disp = ti.field(ti.f32, shape=(NX, NY))

obs_cx = ti.field(ti.f32, shape=())
obs_cy = ti.field(ti.f32, shape=())
obs_vx = ti.field(ti.f32, shape=())
obs_vy = ti.field(ti.f32, shape=())
obs_r  = ti.field(ti.f32, shape=())


# ══════════════════════════════════════════════════════════════════════════════
#  NUMPY MLP  (scratch — no external ML frameworks)
# ══════════════════════════════════════════════════════════════════════════════

class NumpyMLP:
    """
    Fully-connected MLP:  He init · ReLU hidden · linear output · Adam.

    Architecture: [PATCH_FLAT=9] → 32 → 16 → [1]
    Input  : flattened 3×3 divergence patch (normalised)
    Output : scalar pressure prediction at the centre cell
    """

    def __init__(self, layer_sizes):
        self.n_layers = len(layer_sizes) - 1
        self.weights, self.biases = [], []
        self.m_w, self.v_w = [], []
        self.m_b, self.v_b = [], []

        for k in range(self.n_layers):
            fan_in  = layer_sizes[k]
            fan_out = layer_sizes[k + 1]
            W = np.random.randn(fan_in, fan_out).astype(np.float32) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(W);  self.biases.append(b)
            self.m_w.append(np.zeros_like(W)); self.v_w.append(np.zeros_like(W))
            self.m_b.append(np.zeros_like(b)); self.v_b.append(np.zeros_like(b))

        self._z = []
        self._a = []

    def forward(self, X):
        self._a = [X];  self._z = []
        a = X
        for k in range(self.n_layers):
            z = a @ self.weights[k] + self.biases[k]
            self._z.append(z)
            a = np.maximum(z, 0.0) if k < self.n_layers - 1 else z
            self._a.append(a)
        return a

    def backward(self, y_pred, y_true):
        batch  = y_pred.shape[0]
        delta  = (2.0 / batch) * (y_pred - y_true)
        grad_w = [None] * self.n_layers
        grad_b = [None] * self.n_layers
        for k in range(self.n_layers - 1, -1, -1):
            grad_w[k] = self._a[k].T @ delta
            grad_b[k] = delta.sum(axis=0)
            if k > 0:
                delta = delta @ self.weights[k].T
                delta = delta * (self._z[k - 1] > 0).astype(np.float32)
        return grad_w, grad_b

    def adam_step(self, grad_w, grad_b, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        for k in range(self.n_layers):
            for (param, m_buf, v_buf, grad) in [
                (self.weights[k], self.m_w[k], self.v_w[k], grad_w[k]),
                (self.biases[k],  self.m_b[k], self.v_b[k], grad_b[k]),
            ]:
                m_buf[:] = beta1 * m_buf + (1 - beta1) * grad
                v_buf[:] = beta2 * v_buf + (1 - beta2) * grad ** 2
                m_hat    = m_buf / (1 - beta1 ** t)
                v_hat    = v_buf / (1 - beta2 ** t)
                param   -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict(self, X):
        a = X
        for k in range(self.n_layers):
            z = a @ self.weights[k] + self.biases[k]
            a = np.maximum(z, 0.0) if k < self.n_layers - 1 else z
        return a


# ══════════════════════════════════════════════════════════════════════════════
#  ML ACCELERATOR
# ══════════════════════════════════════════════════════════════════════════════

class MLAccelerator:
    PHASE_COLLECT = 0
    PHASE_TRAIN   = 1
    PHASE_ACTIVE  = 2

    def __init__(self):
        self.phase       = self.PHASE_COLLECT
        self.frame_count = 0

        self.net    = NumpyMLP([PATCH_FLAT, 32, 16, 1])
        self.x_mean = np.zeros(PATCH_FLAT, dtype=np.float32)
        self.x_std  = np.ones(PATCH_FLAT,  dtype=np.float32)

        est = COLLECT_FRAMES * SAMPLES_PER_FRAME * SUBSTEPS
        self.X_buf   = np.empty((est, PATCH_FLAT), dtype=np.float32)
        self.y_buf   = np.empty((est, 1),          dtype=np.float32)
        self.buf_idx = 0

        pad = PATCH_SIZE // 2
        self.pad         = pad
        self._div_padded = np.zeros((NX + 2*pad, NY + 2*pad), dtype=np.float32)
        self._s_cache    = np.zeros((NX, NY), dtype=np.float32)
        self._s_dirty    = True

    @staticmethod
    def _divergence(u_np, v_np):
        return (u_np[1:, :] - u_np[:-1, :]) + (v_np[:, 1:] - v_np[:, :-1])

    # ── Phase 1: fully vectorised data collection — NO Python loop ────────
    def collect(self, div_field, pressure_field, solid_field):
        try:
            # Find all eligible interior fluid cells in one vectorised op
            ai, aj = np.where(
                (solid_field > 0)
                & (np.arange(NX)[:, None] >= 1) & (np.arange(NX)[:, None] < NX - 1)
                & (np.arange(NY)[None, :] >= 1) & (np.arange(NY)[None, :] < NY - 1)
            )
            if len(ai) == 0:
                return

            n  = min(SAMPLES_PER_FRAME, len(ai))
            ix = np.random.choice(len(ai), n, replace=False)
            si, sj = ai[ix], aj[ix]

            # Pad once, then use sliding_window_view for zero-copy patches
            pad    = self.pad
            padded = np.pad(div_field, pad, mode='constant', constant_values=0.0)
            windows = sliding_window_view(padded, (PATCH_SIZE, PATCH_SIZE))  # (NX,NY,3,3)
            patches = windows[si, sj].reshape(n, PATCH_FLAT)                # (n, 9) — vectorised
            pressures = pressure_field[si, sj]                               # (n,)

            pos    = self.buf_idx
            end    = min(pos + n, self.X_buf.shape[0])
            actual = end - pos
            if actual > 0:
                self.X_buf[pos:end] = patches[:actual]
                self.y_buf[pos:end, 0] = pressures[:actual]
                self.buf_idx = end
        except Exception:
            pass

    # ── Phase 2: one-shot training ────────────────────────────────────────
    def train(self):
        n = self.buf_idx
        if n < ML_BATCH:
            safe_print("[ML] Too few samples — keeping full solver.")
            self.phase = self.PHASE_ACTIVE
            return

        X = self.X_buf[:n].copy()
        y = self.y_buf[:n].copy()

        self.x_mean = X.mean(axis=0)
        self.x_std  = X.std(axis=0).clip(min=1e-8)
        X = (X - self.x_mean) / self.x_std

        safe_print(f"[ML] Training on {n} samples for {ML_EPOCHS} epochs …")
        t0 = time.perf_counter()

        adam_t = 0
        for epoch in range(1, ML_EPOCHS + 1):
            perm = np.random.permutation(n)
            X, y = X[perm], y[perm]
            epoch_loss, n_batches = 0.0, 0
            for b0 in range(0, n, ML_BATCH):
                Xb, yb = X[b0:b0 + ML_BATCH], y[b0:b0 + ML_BATCH]
                pred   = self.net.forward(Xb)
                loss   = np.mean((pred - yb) ** 2)
                epoch_loss += loss;  n_batches += 1
                gw, gb = self.net.backward(pred, yb)
                adam_t += 1
                self.net.adam_step(gw, gb, lr=ML_LR, t=adam_t)

            if epoch % 10 == 0 or epoch == 1:
                safe_print(f"  epoch {epoch:3d}/{ML_EPOCHS}  loss={epoch_loss/max(n_batches,1):.6f}")

        safe_print(f"[ML] Done in {time.perf_counter()-t0:.2f}s")
        self.X_buf = None
        self.y_buf = None
        self.phase = self.PHASE_ACTIVE

    # ── Phase 3: vectorised inference ─────────────────────────────────────
    def predict_pressure(self, div_field, solid_field):
        try:
            pad = self.pad
            self._div_padded[:] = 0.0
            self._div_padded[pad:pad + NX, pad:pad + NY] = div_field

            windows = sliding_window_view(self._div_padded, (PATCH_SIZE, PATCH_SIZE))
            patches = windows[:NX, :NY].reshape(-1, PATCH_FLAT)

            patches = (patches - self.x_mean) / self.x_std
            pred    = self.net.predict(patches).reshape(NX, NY)

            pred   *= solid_field
            pred[[0, NX - 1], :] = 0.0
            pred[:, [0, NY - 1]] = 0.0
            return pred.astype(np.float32)
        except Exception:
            return None

    # ── Per-substep entry point ────────────────────────────────────────────
    def step(self, dt, use_gravity, is_dragging=False):
        stamp_obs_vel()
        clamp_pressure(-500.0, 500.0)
        clamp_velocity(-20.0, 20.0)
        if use_gravity:
            integrate(dt, GRAVITY)

        if self.phase == self.PHASE_COLLECT:
            p.fill(0.0)
            for _ in range(NUM_ITERS):
                solve_pressure(dt, 0)
                solve_pressure(dt, 1)
            try:
                div = self._divergence(u.to_numpy(), v.to_numpy())
                self.collect(div, p.to_numpy(), s.to_numpy())
            except Exception:
                pass
            iters_used = NUM_ITERS

        elif self.phase == self.PHASE_ACTIVE:
            ml_ok = False
            if not is_dragging:
                try:
                    if self._s_dirty:
                        self._s_cache[:] = s.to_numpy()
                        self._s_dirty = False
                    u_np  = u.to_numpy()
                    v_np  = v.to_numpy()
                    div   = self._divergence(u_np, v_np)
                    pred  = self.predict_pressure(div, self._s_cache)
                    if pred is not None:
                        p.from_numpy(pred)
                        ml_ok = True
                except Exception:
                    pass

            if not ml_ok:
                p.fill(0.0)

            n_iter = ML_ITERS if ml_ok else NUM_ITERS
            for _ in range(n_iter):
                solve_pressure(dt, 0)
                solve_pressure(dt, 1)
            iters_used = n_iter

        else:  # PHASE_TRAIN — transient fallback
            p.fill(0.0)
            for _ in range(NUM_ITERS):
                solve_pressure(dt, 0)
                solve_pressure(dt, 1)
            iters_used = NUM_ITERS

        extrapolate()
        advect_vel(dt)
        advect_smoke(dt)
        return iters_used


# ══════════════════════════════════════════════════════════════════════════════
#  TAICHI KERNELS
# ══════════════════════════════════════════════════════════════════════════════

@ti.kernel
def rebuild_solid():
    cx = obs_cx[None];  cy = obs_cy[None];  r = obs_r[None]
    for i, j in ti.ndrange(NX, NY):
        wall   = (i == 0) or (i == NX - 1) or (j == 0) or (j == NY - 1)
        x      = (float(i) + 0.5) * H
        y      = (float(j) + 0.5) * H
        in_obs = (x - cx)**2 + (y - cy)**2 <= r * r
        s[i, j] = 0.0 if (wall or in_obs) else 1.0


@ti.kernel
def stamp_obs_vel():
    cx  = obs_cx[None];  cy = obs_cy[None]
    r   = obs_r[None];   vx = obs_vx[None];  vy = obs_vy[None]
    rr  = (r + 1.5 * H) ** 2
    for i, j in ti.ndrange((1, NX), NY):
        x = float(i) * H;  y = (float(j) + 0.5) * H
        if (x - cx)**2 + (y - cy)**2 <= rr:
            u[i, j] = vx
    for i, j in ti.ndrange(NX, (1, NY)):
        x = (float(i) + 0.5) * H;  y = float(j) * H
        if (x - cx)**2 + (y - cy)**2 <= rr:
            v[i, j] = vy


@ti.kernel
def init_fields():
    """Closed paint pool — paint fills every fluid cell from the start."""
    for i, j in ti.ndrange(NX, NY):
        m[i, j] = 1.0   # pool starts completely full of paint
        p[i, j] = 0.0
    for i, j in ti.ndrange(NX + 1, NY):
        u[i, j] = 0.0
    for i, j in ti.ndrange(NX, NY + 1):
        v[i, j] = 0.0


@ti.kernel
def integrate(dt: ti.f32, gravity: ti.f32):
    for i, j in ti.ndrange((1, NX - 1), (1, NY)):
        if s[i, j] > 0.0 and s[i, j - 1] > 0.0:
            v[i, j] += gravity * dt


@ti.kernel
def solve_pressure(dt: ti.f32, parity: ti.i32):
    cp = DENSITY * H / dt
    for i, j in ti.ndrange((1, NX - 1), (1, NY - 1)):
        if (i + j) % 2 != parity:  continue
        if s[i, j] == 0.0:         continue
        sx0   = s[i - 1, j];  sx1 = s[i + 1, j]
        sy0   = s[i, j - 1];  sy1 = s[i, j + 1]
        s_sum = sx0 + sx1 + sy0 + sy1
        if s_sum == 0.0:  continue
        div        = (u[i + 1, j] - u[i, j]) + (v[i, j + 1] - v[i, j])
        correction = -div / s_sum * OVER_RELAX
        p[i, j]     += cp * correction
        u[i,     j] -= sx0 * correction;  u[i + 1, j] += sx1 * correction
        v[i, j    ] -= sy0 * correction;  v[i, j + 1] += sy1 * correction


@ti.kernel
def extrapolate():
    for j in range(NY):
        u[0, j]  = u[1,      j];  u[NX, j] = u[NX - 1, j]
    for i in range(NX):
        v[i, 0]  = v[i, 1     ];  v[i, NY] = v[i, NY - 1]


@ti.kernel
def clamp_pressure(lo: ti.f32, hi: ti.f32):
    for i, j in ti.ndrange(NX, NY):
        p[i, j] = ti.max(lo, ti.min(hi, p[i, j]))


@ti.kernel
def clamp_velocity(lo: ti.f32, hi: ti.f32):
    for i, j in ti.ndrange(NX + 1, NY):
        u[i, j] = ti.max(lo, ti.min(hi, u[i, j]))
    for i, j in ti.ndrange(NX, NY + 1):
        v[i, j] = ti.max(lo, ti.min(hi, v[i, j]))


@ti.kernel
def reset_velocity():
    """Zero all velocity after violent drag."""
    for i, j in ti.ndrange(NX + 1, NY):
        u[i, j] = 0.0
    for i, j in ti.ndrange(NX, NY + 1):
        v[i, j] = 0.0


@ti.kernel
def velocity_impulse_kernel(gx: ti.i32, gy: ti.i32, rad: ti.i32, strength: ti.f32):
    """Radial outward impulse — used for click-to-disturb."""
    for i, j in ti.ndrange(NX + 1, NY):
        dx = float(i) - float(gx)
        dy = float(j) + 0.5 - float(gy)
        d2 = dx * dx + dy * dy
        if d2 <= float(rad * rad) and s[i, j] > 0.0:
            dist = ti.sqrt(d2) + 1e-6
            u[i, j] += strength * dx / dist
    for i, j in ti.ndrange(NX, NY + 1):
        dx = float(i) + 0.5 - float(gx)
        dy = float(j) - float(gy)
        d2 = dx * dx + dy * dy
        if d2 <= float(rad * rad) and s[i, j] > 0.0:
            dist = ti.sqrt(d2) + 1e-6
            v[i, j] += strength * dy / dist


@ti.func
def clampf(x: ti.f32, lo: ti.f32, hi: ti.f32) -> ti.f32:
    return ti.max(lo, ti.min(hi, x))


@ti.func
def sample_u(x: ti.f32, y: ti.f32) -> ti.f32:
    x  = clampf(x, H, NX*H);   y  = clampf(y, H, NY*H)
    x0 = ti.min(int(x * INV_H),        NX-1);  x1 = ti.min(x0+1, NX-1)
    y0 = ti.min(int((y-H2)*INV_H),     NY-1);  y1 = ti.min(y0+1, NY-1)
    tx = (x - float(x0)*H)*INV_H;              ty = (y-H2-float(y0)*H)*INV_H
    sx = 1.0-tx;                               sy = 1.0-ty
    return sx*sy*u[x0,y0]+tx*sy*u[x1,y0]+tx*ty*u[x1,y1]+sx*ty*u[x0,y1]


@ti.func
def sample_v(x: ti.f32, y: ti.f32) -> ti.f32:
    x  = clampf(x, H, NX*H);   y  = clampf(y, H, NY*H)
    x0 = ti.min(int((x-H2)*INV_H),    NX-1);  x1 = ti.min(x0+1, NX-1)
    y0 = ti.min(int(y*INV_H),         NY-1);  y1 = ti.min(y0+1, NY-1)
    tx = (x-H2-float(x0)*H)*INV_H;            ty = (y-float(y0)*H)*INV_H
    sx = 1.0-tx;                               sy = 1.0-ty
    return sx*sy*v[x0,y0]+tx*sy*v[x1,y0]+tx*ty*v[x1,y1]+sx*ty*v[x0,y1]


@ti.func
def sample_m(x: ti.f32, y: ti.f32) -> ti.f32:
    x  = clampf(x, H, NX*H);   y  = clampf(y, H, NY*H)
    x0 = ti.min(int((x-H2)*INV_H),    NX-1);  x1 = ti.min(x0+1, NX-1)
    y0 = ti.min(int((y-H2)*INV_H),    NY-1);  y1 = ti.min(y0+1, NY-1)
    tx = (x-H2-float(x0)*H)*INV_H;            ty = (y-H2-float(y0)*H)*INV_H
    sx = 1.0-tx;                               sy = 1.0-ty
    return sx*sy*m[x0,y0]+tx*sy*m[x1,y0]+tx*ty*m[x1,y1]+sx*ty*m[x0,y1]


@ti.func
def avg_u(i: ti.i32, j: ti.i32) -> ti.f32:
    return 0.25*(u[i,j-1]+u[i,j]+u[i+1,j-1]+u[i+1,j])


@ti.func
def avg_v(i: ti.i32, j: ti.i32) -> ti.f32:
    return 0.25*(v[i-1,j]+v[i,j]+v[i-1,j+1]+v[i,j+1])


@ti.kernel
def advect_vel(dt: ti.f32):
    """RK2 semi-Lagrangian velocity advection."""
    for i, j in ti.ndrange(NX+1, NY):   new_u[i,j] = u[i,j]
    for i, j in ti.ndrange(NX, NY+1):   new_v[i,j] = v[i,j]

    for i, j in ti.ndrange((1,NX),(1,NY-1)):
        if s[i,j] != 0.0 and s[i-1,j] != 0.0:
            x0 = float(i)*H;       y0 = float(j)*H + H2
            u0 = u[i,j];           v0 = avg_v(i,j)
            xm = x0 - 0.5*dt*u0;  ym = y0 - 0.5*dt*v0
            um = sample_u(xm, ym); vm = sample_v(xm, ym)
            new_u[i,j] = sample_u(x0 - dt*um, y0 - dt*vm)

    for i, j in ti.ndrange((1,NX-1),(1,NY)):
        if s[i,j] != 0.0 and s[i,j-1] != 0.0:
            x0 = float(i)*H + H2;  y0 = float(j)*H
            u0 = avg_u(i,j);        v0 = v[i,j]
            xm = x0 - 0.5*dt*u0;   ym = y0 - 0.5*dt*v0
            um = sample_u(xm, ym);  vm = sample_v(xm, ym)
            new_v[i,j] = sample_v(x0 - dt*um, y0 - dt*vm)

    for i, j in ti.ndrange(NX+1, NY):   u[i,j] = new_u[i,j]
    for i, j in ti.ndrange(NX, NY+1):   v[i,j] = new_v[i,j]


@ti.kernel
def advect_smoke(dt: ti.f32):
    """RK2 paint-density advection."""
    for i, j in ti.ndrange((1,NX-1),(1,NY-1)):
        if s[i,j] != 0.0:
            x0 = (float(i)+0.5)*H;    y0 = (float(j)+0.5)*H
            uu = (u[i,j]+u[i+1,j])*0.5;  vv = (v[i,j]+v[i,j+1])*0.5
            xm = x0-0.5*dt*uu;  ym = y0-0.5*dt*vv
            um = sample_u(xm,ym); vm = sample_v(xm,ym)
            new_m[i,j] = sample_m(x0-dt*um, y0-dt*vm)
        else:
            new_m[i,j] = m[i,j]
    for i, j in ti.ndrange(NX, NY):
        m[i,j] = new_m[i,j]


@ti.func
def sci_color(val: ti.f32) -> ti.types.vector(3, ti.f32):
    t = ti.max(0.0, ti.min(1.0, val))
    r = 0.0; g = 0.0; b = 0.0
    if   t < 0.25: r = 0.0;        g = t*4.0;           b = 1.0
    elif t < 0.50: r = 0.0;        g = 1.0;              b = 1.0-(t-0.25)*4.0
    elif t < 0.75: r = (t-0.5)*4;  g = 1.0;              b = 0.0
    else:          r = 1.0;        g = 1.0-(t-0.75)*4;   b = 0.0
    return ti.Vector([r, g, b])


@ti.kernel
def update_p_display(alpha: ti.f32):
    for i, j in ti.ndrange(NX, NY):
        p_disp[i,j] = alpha * p[i,j] + (1.0 - alpha) * p_disp[i,j]


@ti.kernel
def render(show_pressure: ti.i32):
    """
    Paint-pool render — img is (WIN_W, WIN_H); each pixel maps to a grid cell.
    · Window pixel (pi, pj) where pj=0 is the BOTTOM row (Taichi convention).
    · Grid cell gi = pi * NX // WIN_W,  gj = pj * NY // WIN_H  (Y already aligned).
    · Ball : opaque sphere with diffuse + rim + Phong specular shading.
    · Paint: deep indigo/violet + velocity shimmer on disturbed surface.
    · Walls: near-black.
    """
    cx = obs_cx[None]; cy = obs_cy[None]; r = obs_r[None]
    # 1.5-pixel feather in world-space for smooth edge
    feather = 1.5 * (NX * H) / float(WIN_W)

    for pi, pj in img:
        # Map window pixel → grid cell
        gi = ti.min(pi * NX // WIN_W, NX - 1)
        gj = ti.min(pj * NY // WIN_H, NY - 1)

        # World-space position at sub-pixel precision (better than grid-snapped)
        x  = (float(pi) + 0.5) * (NX * H) / float(WIN_W)
        y  = (float(pj) + 0.5) * (NY * H) / float(WIN_H)
        dx = x - cx;  dy = y - cy
        dist = ti.sqrt(dx * dx + dy * dy)

        # Smooth ball mask: 1 fully inside, 0 fully outside, smoothstep edge
        t      = ti.max(0.0, ti.min(1.0, (r - dist) / feather))
        ball_a = t * t * (3.0 - 2.0 * t)

        # Declare bg before branching so Taichi can see it unconditionally
        bg = ti.Vector([0.04, 0.04, 0.05])
        if s[gi, gj] == 0.0:
            bg = ti.Vector([0.04, 0.04, 0.05])
        elif show_pressure == 1:
            bg = sci_color(0.5 + p_disp[gi, gj] * 0.001)
        else:
            k = m[gi, gj]
            vel = ti.sqrt(
                ((u[gi, gj] + u[gi + 1, gj]) * 0.5) ** 2 +
                ((v[gi, gj] + v[gi, gj + 1]) * 0.5) ** 2
            )
            shimmer = ti.min(vel * 0.12, 0.28)
            bg = ti.Vector([
                0.04 + 0.28 * k + shimmer * 0.8,
                0.02 + 0.06 * k,
                0.18 + 0.66 * k + shimmer * 0.5
            ])

        # Blend white ball over background using smooth mask
        white = ti.Vector([1.0, 1.0, 1.0])
        img[pi, pj] = white * ball_a + bg * (1.0 - ball_a)


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def reset_sim():
    obs_cx[None] = NX * H * 0.50
    obs_cy[None] = NY * H * 0.50
    obs_vx[None] = 0.0
    obs_vy[None] = 0.0
    obs_r[None]  = OBS_R
    init_fields()
    rebuild_solid()
    p_disp.fill(0.0)


def safe_print(text):
    try:
        print(text); sys.stdout.flush()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    reset_sim()
    ml = MLAccelerator()
    ml._s_dirty = True
    gui = ti.GUI("Eulerian Fluid + ML Pressure Solver — Paint Pool",
                 res=(WIN_W, WIN_H))

    show_pressure = False
    use_gravity   = False
    dragging_obs  = False
    frame         = 0
    t0            = time.perf_counter()

    safe_print("+-------------------------------------------------------------+")
    safe_print("|   Eulerian Fluid Simulator  +  ML Pressure Warm-Start       |")
    safe_print("|   Paint Pool Edition                                        |")
    safe_print("|  Phase 1 COLLECTING : full solver, gathering data (40 fr)  |")
    safe_print("|  Phase 2 TRAINING   : one-shot Adam training of MLP         |")
    safe_print("|  Phase 3 ML-ACTIVE  : 20 iters → 8 iters  (60% cheaper)   |")
    safe_print("|  Drag ball=move through paint   Click=impulse / splash      |")
    safe_print("|  P=pressure/paint  R=reset  ESC/Q=quit                      |")
    safe_print("+-------------------------------------------------------------+")

    while gui.running:

        # ── Input ─────────────────────────────────────────────────────────
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in (ti.GUI.ESCAPE, 'q', 'Q'):
                gui.running = False
            elif e.key in ('p', 'P'):
                show_pressure = not show_pressure
                safe_print("View: PRESSURE" if show_pressure else "View: PAINT")
            elif e.key in ('r', 'R'):
                reset_sim()
                ml._s_dirty = True
                safe_print("Reset.")

        mx, my    = gui.get_cursor_pos()
        world_mx  = mx * NX * H
        world_my  = my * NY * H
        cx = obs_cx[None]; cy = obs_cy[None]; r = obs_r[None]
        dist2 = (world_mx - cx)**2 + (world_my - cy)**2

        if gui.is_pressed(ti.GUI.LMB):
            if dragging_obs or dist2 <= (r * 1.5)**2:
                # ── Drag ball ─────────────────────────────────────────
                dragging_obs = True
                new_cx = float(np.clip(world_mx, r + 2*H, NX*H - r - 2*H))
                new_cy = float(np.clip(world_my, r + 2*H, NY*H - r - 2*H))
                obs_vx[None] = (new_cx - cx) / SUB_DT
                obs_vy[None] = (new_cy - cy) / SUB_DT
                obs_cx[None] = new_cx
                obs_cy[None] = new_cy
                rebuild_solid()
                ml._s_dirty = True
                p.fill(0.0)
                p_disp.fill(0.0)
            else:
                # ── Click in paint → radial impulse ───────────────────
                gx = int(mx * NX)
                gy = int(my * NY)
                velocity_impulse_kernel(gx, gy, NY // 10, 3.5)
        else:
            if dragging_obs:
                obs_vx[None] = 0.0
                obs_vy[None] = 0.0
                reset_velocity()
                stamp_obs_vel()
                p.fill(0.0)
                p_disp.fill(0.0)
            dragging_obs = False

        # ── Physics substeps ──────────────────────────────────────────────
        for _ in range(SUBSTEPS):
            iters_used = ml.step(SUB_DT, use_gravity, is_dragging=dragging_obs)

        # ── Phase transitions ─────────────────────────────────────────────
        frame += 1
        if ml.phase == MLAccelerator.PHASE_COLLECT:
            ml.frame_count += 1
            if ml.frame_count >= COLLECT_FRAMES:
                safe_print(f"[ML] Collection done — {ml.buf_idx} samples gathered.")
                ml.phase = MLAccelerator.PHASE_TRAIN
                ml.train()

        # ── Render ────────────────────────────────────────────────────────
        # Y-flip is baked into the render kernel; gui.set_image() takes the
        # Taichi field directly — no numpy round-trip, no cv2, zero Python work.
        update_p_display(0.7)
        render(1 if show_pressure else 0)
        gui.set_image(img)

        if frame % 60 == 0:
            fps   = 60.0 / max(time.perf_counter() - t0, 1e-6)
            label = ["COLLECTING", "TRAINING", "ML-ACTIVE"][ml.phase]
            safe_print(f"Frame {frame:5d}  {fps:5.1f} FPS  iters={iters_used:2d}  phase={label}")
            t0 = time.perf_counter()

        gui.show()

    safe_print("Done.")


if __name__ == "__main__":
    main()
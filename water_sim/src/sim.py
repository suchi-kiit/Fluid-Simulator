"""
FLIP Fluid Simulation with ML-Accelerated Pressure Solver
===========================================================
3-phase pipeline:
  Phase 1 (collecting): Normal Jacobi solver + subsample training data
  Phase 2 (training):   One-time pause to train the neural network
  Phase 3 (ML-active):  NN predicts initial pressure → fewer Jacobi iters

Based on Matthias Müller's Ten Minute Physics Tutorial #18.
"""

import taichi as ti
import numpy as np
import time

from src.config import *
from src.ml_solver import PressureNet, TrainingDataCollector

# ═══════════════════════════════════════════════════════════
# DERIVED CONSTANTS
# ═══════════════════════════════════════════════════════════
h   = SIM_HEIGHT / GRID_RES
nX  = int(SIM_WIDTH / h) + 1
nY  = GRID_RES + 1
NUM_CELLS = nX * nY

p_rad = PARTICLE_RADIUS_FACTOR * h

_ppd     = 2
_spacing = h / _ppd

_fill_x0 = h + 0.01
_fill_x1 = SIM_WIDTH - h - 0.01
_fill_y0 = h + 0.01
_fill_y1 = SIM_HEIGHT * FILL_HEIGHT_FRAC

_nx_p = int((_fill_x1 - _fill_x0) / _spacing)
_ny_p = int((_fill_y1 - _fill_y0) / _spacing)
MAX_PARTICLES = _nx_p * _ny_p + 256

# Cell type enum
FLUID_CELL = 0
AIR_CELL   = 1
SOLID_CELL = 2

# ═══════════════════════════════════════════════════════════
# TAICHI FIELDS
# ═══════════════════════════════════════════════════════════
u           = ti.field(ti.f32, shape=(nX, nY))
v           = ti.field(ti.f32, shape=(nX, nY))
u_prev      = ti.field(ti.f32, shape=(nX, nY))
v_prev      = ti.field(ti.f32, shape=(nX, nY))
u_weight    = ti.field(ti.f32, shape=(nX, nY))
v_weight    = ti.field(ti.f32, shape=(nX, nY))
solid       = ti.field(ti.f32, shape=(nX, nY))
cell_type   = ti.field(ti.i32, shape=(nX, nY))
particle_density = ti.field(ti.f32, shape=(nX, nY))

# Divergence & pressure fields (extracted to NumPy for ML)
divergence_field = ti.field(ti.f32, shape=(nX, nY))

p_x    = ti.field(ti.f32, shape=MAX_PARTICLES)
p_y    = ti.field(ti.f32, shape=MAX_PARTICLES)
p_u    = ti.field(ti.f32, shape=MAX_PARTICLES)
p_v    = ti.field(ti.f32, shape=MAX_PARTICLES)
n_part = ti.field(ti.i32, shape=())

MAX_PPC = 16
cell_pcount = ti.field(ti.i32, shape=(nX, nY))
cell_plist  = ti.field(ti.i32, shape=(nX, nY, MAX_PPC))

obs_x  = ti.field(ti.f32, shape=())
obs_y  = ti.field(ti.f32, shape=())
obs_r  = ti.field(ti.f32, shape=())
obs_vx = ti.field(ti.f32, shape=())
obs_vy = ti.field(ti.f32, shape=())

rest_density_val = ti.field(ti.f32, shape=())

pixels = ti.Vector.field(3, ti.f32, shape=(WINDOW_W, WINDOW_H))


# ═══════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════

@ti.kernel
def init_solid():
    for i, j in solid:
        if i == 0 or i == nX - 1 or j == 0:
            solid[i, j] = 0.0
        else:
            solid[i, j] = 1.0


def init_particles():
    positions_x = []
    positions_y = []
    y = _fill_y0
    while y < _fill_y1:
        x = _fill_x0
        while x < _fill_x1:
            positions_x.append(x)
            positions_y.append(y)
            x += _spacing
        y += _spacing
    n = min(len(positions_x), MAX_PARTICLES)
    n_part[None] = n
    arr_x = np.zeros(MAX_PARTICLES, dtype=np.float32)
    arr_y = np.zeros(MAX_PARTICLES, dtype=np.float32)
    arr_x[:n] = np.array(positions_x[:n], dtype=np.float32)
    arr_y[:n] = np.array(positions_y[:n], dtype=np.float32)
    p_x.from_numpy(arr_x)
    p_y.from_numpy(arr_y)
    p_u.from_numpy(np.zeros(MAX_PARTICLES, dtype=np.float32))
    p_v.from_numpy(np.zeros(MAX_PARTICLES, dtype=np.float32))


def init_obstacle():
    obs_x[None] = SIM_WIDTH * OBSTACLE_X_FRAC
    obs_y[None] = SIM_HEIGHT * OBSTACLE_Y_FRAC
    obs_r[None] = SIM_HEIGHT * OBSTACLE_R_FRAC
    obs_vx[None] = 0.0
    obs_vy[None] = 0.0
    rest_density_val[None] = 0.0


# ═══════════════════════════════════════════════════════════
# SIMULATION KERNELS
# ═══════════════════════════════════════════════════════════

@ti.kernel
def integrate_particles(dt: ti.f32, grav: ti.f32):
    n = n_part[None]
    for i in range(n):
        p_v[i] += dt * grav
        p_x[i] += dt * p_u[i]
        p_y[i] += dt * p_v[i]


@ti.kernel
def clamp_particles_to_domain():
    n = n_part[None]
    min_x = h + p_rad
    max_x = (nX - 1) * h - p_rad
    min_y = h + p_rad
    max_y = (nY - 1) * h - p_rad
    ox = obs_x[None]; oy = obs_y[None]; orr = obs_r[None]
    ovx = obs_vx[None]; ovy = obs_vy[None]

    for i in range(n):
        x = p_x[i]; y = p_y[i]
        if x < min_x: x = min_x; p_u[i] = 0.0
        if x > max_x: x = max_x; p_u[i] = 0.0
        if y < min_y: y = min_y; p_v[i] = 0.0
        if y > max_y: y = max_y; p_v[i] = 0.0

        dx = x - ox; dy = y - oy
        d2 = dx * dx + dy * dy
        r_push = orr + p_rad
        if d2 < r_push * r_push:
            dist = ti.sqrt(d2)
            if dist < 1e-8: dx = 1.0; dy = 0.0; dist = 1.0
            x = ox + r_push * dx / dist
            y = oy + r_push * dy / dist
            p_u[i] = ovx; p_v[i] = ovy
        p_x[i] = x; p_y[i] = y


@ti.kernel
def build_cell_list():
    for i, j in cell_pcount:
        cell_pcount[i, j] = 0
    n = n_part[None]
    for idx in range(n):
        ci = ti.max(0, ti.min(ti.cast(p_x[idx] / h, ti.i32), nX - 1))
        cj = ti.max(0, ti.min(ti.cast(p_y[idx] / h, ti.i32), nY - 1))
        slot = ti.atomic_add(cell_pcount[ci, cj], 1)
        if slot < MAX_PPC:
            cell_plist[ci, cj, slot] = idx


@ti.kernel
def push_apart_one_pass():
    min_dist = 2.0 * p_rad; min_dist2 = min_dist * min_dist
    n = n_part[None]
    for idx in range(n):
        xi = p_x[idx]; yi = p_y[idx]
        ci = ti.cast(xi / h, ti.i32); cj = ti.cast(yi / h, ti.i32)
        for di in ti.static(range(-1, 2)):
            for dj in ti.static(range(-1, 2)):
                ni = ci + di; nj = cj + dj
                if 0 <= ni < nX and 0 <= nj < nY:
                    cnt = ti.min(cell_pcount[ni, nj], MAX_PPC)
                    for k in range(cnt):
                        jdx = cell_plist[ni, nj, k]
                        if jdx > idx:
                            dx = p_x[jdx] - xi; dy = p_y[jdx] - yi
                            d2 = dx * dx + dy * dy
                            if d2 < min_dist2 and d2 > 1e-12:
                                d = ti.sqrt(d2)
                                sf = 0.5 * (min_dist - d) / d
                                p_x[idx] -= dx * sf; p_y[idx] -= dy * sf
                                p_x[jdx] += dx * sf; p_y[jdx] += dy * sf


@ti.kernel
def mark_obstacle_cells():
    ox = obs_x[None]; oy = obs_y[None]; rr = obs_r[None]
    for i, j in solid:
        if i > 0 and i < nX - 1 and j > 0:
            cx = (i + 0.5) * h; cy = (j + 0.5) * h
            dx = cx - ox; dy = cy - oy
            if dx * dx + dy * dy < rr * rr:
                solid[i, j] = 0.0
            else:
                solid[i, j] = 1.0


@ti.kernel
def classify_cells():
    for i, j in cell_type:
        if solid[i, j] == 0.0:
            cell_type[i, j] = SOLID_CELL
        else:
            cell_type[i, j] = AIR_CELL
    n = n_part[None]
    for idx in range(n):
        ci = ti.cast(p_x[idx] / h, ti.i32)
        cj = ti.cast(p_y[idx] / h, ti.i32)
        if 0 <= ci < nX and 0 <= cj < nY:
            if cell_type[ci, cj] == AIR_CELL:
                cell_type[ci, cj] = FLUID_CELL


@ti.kernel
def clear_grid():
    for i, j in u:
        u[i, j] = 0.0; v[i, j] = 0.0
        u_weight[i, j] = 0.0; v_weight[i, j] = 0.0


@ti.kernel
def scatter_particles_to_grid():
    n = n_part[None]
    for idx in range(n):
        px_ = p_x[idx]; py_ = p_y[idx]
        pu_ = p_u[idx]; pv_ = p_v[idx]

        ux = px_ / h; uy = (py_ - 0.5 * h) / h
        i0 = ti.cast(ti.floor(ux), ti.i32); j0 = ti.cast(ti.floor(uy), ti.i32)
        fx = ux - i0; fy = uy - j0
        if 0 <= i0 < nX - 1 and 0 <= j0 < nY - 1:
            w00 = (1-fx)*(1-fy); w10 = fx*(1-fy); w01 = (1-fx)*fy; w11 = fx*fy
            ti.atomic_add(u[i0,j0], w00*pu_); ti.atomic_add(u_weight[i0,j0], w00)
            ti.atomic_add(u[i0+1,j0], w10*pu_); ti.atomic_add(u_weight[i0+1,j0], w10)
            ti.atomic_add(u[i0,j0+1], w01*pu_); ti.atomic_add(u_weight[i0,j0+1], w01)
            ti.atomic_add(u[i0+1,j0+1], w11*pu_); ti.atomic_add(u_weight[i0+1,j0+1], w11)

        vx_ = (px_ - 0.5 * h) / h; vy_ = py_ / h
        i0v = ti.cast(ti.floor(vx_), ti.i32); j0v = ti.cast(ti.floor(vy_), ti.i32)
        fx2 = vx_ - i0v; fy2 = vy_ - j0v
        if 0 <= i0v < nX - 1 and 0 <= j0v < nY - 1:
            w00 = (1-fx2)*(1-fy2); w10 = fx2*(1-fy2); w01 = (1-fx2)*fy2; w11 = fx2*fy2
            ti.atomic_add(v[i0v,j0v], w00*pv_); ti.atomic_add(v_weight[i0v,j0v], w00)
            ti.atomic_add(v[i0v+1,j0v], w10*pv_); ti.atomic_add(v_weight[i0v+1,j0v], w10)
            ti.atomic_add(v[i0v,j0v+1], w01*pv_); ti.atomic_add(v_weight[i0v,j0v+1], w01)
            ti.atomic_add(v[i0v+1,j0v+1], w11*pv_); ti.atomic_add(v_weight[i0v+1,j0v+1], w11)


@ti.kernel
def normalize_grid():
    for i, j in u:
        if u_weight[i, j] > 0.0: u[i, j] /= u_weight[i, j]
        else: u[i, j] = 0.0
    for i, j in v:
        if v_weight[i, j] > 0.0: v[i, j] /= v_weight[i, j]
        else: v[i, j] = 0.0


@ti.kernel
def save_velocities():
    for i, j in u:
        u_prev[i, j] = u[i, j]; v_prev[i, j] = v[i, j]


@ti.kernel
def enforce_boundaries():
    for i, j in u:
        if solid[i, j] == 0.0 or (i > 0 and solid[i-1, j] == 0.0):
            u[i, j] = 0.0
        if solid[i, j] == 0.0 or (j > 0 and solid[i, j-1] == 0.0):
            v[i, j] = 0.0


@ti.kernel
def compute_density():
    for i, j in particle_density:
        particle_density[i, j] = 0.0
    n = n_part[None]
    for idx in range(n):
        px_ = p_x[idx]; py_ = p_y[idx]
        cx = (px_ - 0.5 * h) / h; cy = (py_ - 0.5 * h) / h
        i0 = ti.cast(ti.floor(cx), ti.i32); j0 = ti.cast(ti.floor(cy), ti.i32)
        fx = cx - i0; fy = cy - j0
        if 0 <= i0 < nX - 1 and 0 <= j0 < nY - 1:
            ti.atomic_add(particle_density[i0,j0], (1-fx)*(1-fy))
            ti.atomic_add(particle_density[i0+1,j0], fx*(1-fy))
            ti.atomic_add(particle_density[i0,j0+1], (1-fx)*fy)
            ti.atomic_add(particle_density[i0+1,j0+1], fx*fy)


@ti.kernel
def calc_rest_density() -> ti.f32:
    total = 0.0; count = 0
    for i, j in cell_type:
        if cell_type[i, j] == FLUID_CELL:
            total += particle_density[i, j]; count += 1
    result = 0.0
    if count > 0:
        result = total / ti.cast(count, ti.f32)
    return result


@ti.kernel
def pressure_solve_iteration(omega: ti.f32, do_comp: ti.i32, rho0: ti.f32, drift_k: ti.f32):
    for i, j in cell_type:
        if cell_type[i, j] != FLUID_CELL:
            continue
        if i < 1 or i >= nX - 1 or j < 1 or j >= nY - 1:
            continue
        sl = solid[i-1,j]; sr = solid[i+1,j]; sb = solid[i,j-1]; st = solid[i,j+1]
        s_sum = sl + sr + sb + st
        if s_sum < 1e-6:
            continue
        div = u[i+1,j] - u[i,j] + v[i,j+1] - v[i,j]
        if do_comp > 0:
            comp = particle_density[i, j] - rho0
            if comp > 0.0:
                div -= drift_k * comp   # scaled by stiffness (was: div -= comp)
        p = omega * div / s_sum
        u[i,j] += sl * p; u[i+1,j] -= sr * p
        v[i,j] += sb * p; v[i,j+1] -= st * p


@ti.kernel
def grid_to_particles(flip_ratio: ti.f32):
    n = n_part[None]
    for idx in range(n):
        px_ = p_x[idx]; py_ = p_y[idx]
        ux = px_ / h; uy = (py_ - 0.5 * h) / h
        i0 = ti.cast(ti.floor(ux), ti.i32); j0 = ti.cast(ti.floor(uy), ti.i32)
        fx = ux - i0; fy = uy - j0
        if 0 <= i0 < nX - 1 and 0 <= j0 < nY - 1:
            w00=(1-fx)*(1-fy); w10=fx*(1-fy); w01=(1-fx)*fy; w11=fx*fy
            pic_u = w00*u[i0,j0]+w10*u[i0+1,j0]+w01*u[i0,j0+1]+w11*u[i0+1,j0+1]
            prv_u = w00*u_prev[i0,j0]+w10*u_prev[i0+1,j0]+w01*u_prev[i0,j0+1]+w11*u_prev[i0+1,j0+1]
            p_u[idx] = (1-flip_ratio)*pic_u + flip_ratio*(p_u[idx]+pic_u-prv_u)

        vx_ = (px_ - 0.5 * h) / h; vy_ = py_ / h
        i0v = ti.cast(ti.floor(vx_), ti.i32); j0v = ti.cast(ti.floor(vy_), ti.i32)
        fx2 = vx_ - i0v; fy2 = vy_ - j0v
        if 0 <= i0v < nX - 1 and 0 <= j0v < nY - 1:
            w00=(1-fx2)*(1-fy2); w10=fx2*(1-fy2); w01=(1-fx2)*fy2; w11=fx2*fy2
            pic_v = w00*v[i0v,j0v]+w10*v[i0v+1,j0v]+w01*v[i0v,j0v+1]+w11*v[i0v+1,j0v+1]
            prv_v = w00*v_prev[i0v,j0v]+w10*v_prev[i0v+1,j0v]+w01*v_prev[i0v,j0v+1]+w11*v_prev[i0v+1,j0v+1]
            p_v[idx] = (1-flip_ratio)*pic_v + flip_ratio*(p_v[idx]+pic_v-prv_v)


# ═══════════════════════════════════════════════════════════
# DIVERGENCE COMPUTATION (for ML data collection)
# ═══════════════════════════════════════════════════════════

@ti.kernel
def compute_divergence_field(do_comp: ti.i32, rho0: ti.f32, drift_k: ti.f32):
    """Compute divergence at every cell (used as ML input features)."""
    for i, j in divergence_field:
        divergence_field[i, j] = 0.0
        if cell_type[i, j] == FLUID_CELL:
            if 1 <= i < nX - 1 and 1 <= j < nY - 1:
                div = u[i+1,j] - u[i,j] + v[i,j+1] - v[i,j]
                if do_comp > 0:
                    comp = particle_density[i, j] - rho0
                    if comp > 0.0:
                        div -= drift_k * comp
                divergence_field[i, j] = div


@ti.kernel
def apply_pressure_field(pressure: ti.types.ndarray()):
    """Apply ML-predicted pressure corrections to the velocity field."""
    for i, j in cell_type:
        if cell_type[i, j] != FLUID_CELL:
            continue
        if i < 1 or i >= nX - 1 or j < 1 or j >= nY - 1:
            continue
        sl = solid[i-1,j]; sr = solid[i+1,j]; sb = solid[i,j-1]; st = solid[i,j+1]
        s_sum = sl + sr + sb + st
        if s_sum < 1e-6:
            continue
        p = pressure[i, j]
        u[i,j] += sl * p; u[i+1,j] -= sr * p
        v[i,j] += sb * p; v[i,j+1] -= st * p


@ti.kernel
def damp_particle_velocities(damping: ti.f32):
    """Apply gentle velocity damping to reduce jitter at rest."""
    n = n_part[None]
    for i in range(n):
        p_u[i] *= damping
        p_v[i] *= damping


@ti.kernel
def clamp_particle_velocities(max_vel: ti.f32):
    """
    CFL safety clamp: prevent any particle from exceeding max_vel.
    Without this, fast particles jump multiple grid cells per step,
    causing the explosive 'water suddenly jumping' instability.
    max_vel should be <= h / DT (the CFL limit).
    """
    n = n_part[None]
    for i in range(n):
        speed_sq = p_u[i] * p_u[i] + p_v[i] * p_v[i]
        if speed_sq > max_vel * max_vel:
            scale = max_vel / ti.sqrt(speed_sq)
            p_u[i] *= scale
            p_v[i] *= scale


# ═══════════════════════════════════════════════════════════
# 3-PHASE SIMULATION STEP
# ═══════════════════════════════════════════════════════════

class FlipSimulation:
    """
    FLIP fluid simulation with ML-accelerated pressure solving.

    Phase 1: COLLECTING  — Normal Jacobi solve, collect training data
    Phase 2: TRAINING    — One-time network training (~1 second)
    Phase 3: ML_ACTIVE   — NN predicts initial pressure, reduced Jacobi refinement
    """

    PHASE_COLLECTING = 0
    PHASE_TRAINING   = 1
    PHASE_ACTIVE     = 2

    def __init__(self):
        self.frame = 0
        self.phase = self.PHASE_COLLECTING if ML_ENABLED else self.PHASE_ACTIVE
        self.ml_enabled = ML_ENABLED

        # ML components
        if ML_ENABLED:
            self.net = PressureNet(
                input_size=ML_PATCH_SIZE ** 2,
                hidden1=ML_HIDDEN_1,
                hidden2=ML_HIDDEN_2,
                lr=ML_LEARNING_RATE,
                beta1=ML_ADAM_BETA1,
                beta2=ML_ADAM_BETA2,
                eps=ML_ADAM_EPS,
            )
            self.collector = TrainingDataCollector(
                max_samples=ML_COLLECT_FRAMES * ML_SAMPLES_PER_FRAME,
                patch_size=ML_PATCH_SIZE,
            )
        else:
            self.net = None
            self.collector = None

        # Pre-allocated numpy buffers for ML data transfer
        self._div_np = np.zeros((nX, nY), dtype=np.float32)
        self._pressure_np = np.zeros((nX, nY), dtype=np.float32)
        self._solid_np = np.zeros((nX, nY), dtype=np.float32)
        self._ct_np = np.zeros((nX, nY), dtype=np.int32)

    def _extract_grid_state(self):
        """Copy Taichi fields to NumPy for ML processing."""
        self._solid_np[:] = solid.to_numpy()
        self._ct_np[:] = cell_type.to_numpy()
        self._div_np[:] = divergence_field.to_numpy()

    def _compute_pressure_from_divergence(self):
        """
        After Jacobi solve, estimate per-cell pressure correction
        from the divergence reduction. This is our ML target.
        """
        # The "pressure" at each cell is approximated as:
        # p[i,j] ≈ overrelax * divergence / s_sum
        # We compute this directly from the divergence field
        s = self._solid_np
        d = self._div_np
        pressure = np.zeros_like(d)

        for i in range(1, nX - 1):
            for j in range(1, nY - 1):
                if self._ct_np[i, j] == FLUID_CELL:
                    s_sum = s[i-1,j] + s[i+1,j] + s[i,j-1] + s[i,j+1]
                    if s_sum > 1e-6:
                        pressure[i, j] = OVER_RELAX * d[i, j] / s_sum
        return pressure

    def step(self):
        """One full simulation step with ML integration."""

        # ── 1. Integrate particles ──
        integrate_particles(DT, GRAVITY)

        # ── 2. Push particles apart ──
        if SEPARATE_PARTICLES:
            for _ in range(NUM_PARTICLE_ITERS):
                build_cell_list()
                push_apart_one_pass()

        # ── 3. Boundary collisions ──
        clamp_particles_to_domain()

        # ── 4. Cell classification ──
        mark_obstacle_cells()
        classify_cells()

        # ── 5. Particle → Grid ──
        clear_grid()
        scatter_particles_to_grid()
        normalize_grid()

        # ── 6. Save for FLIP ──
        save_velocities()

        # ── 7. Enforce boundaries ──
        enforce_boundaries()

        # ── 8. Density ──
        compute_density()
        rho0 = rest_density_val[None]
        if rho0 == 0.0:
            rho0 = calc_rest_density()
            rest_density_val[None] = rho0

        comp = 1 if COMPENSATE_DRIFT else 0

        # ══════════════════════════════════════════════════
        # 9. PRESSURE SOLVE — 3-phase ML pipeline
        # ══════════════════════════════════════════════════

        if self.phase == self.PHASE_COLLECTING:
            # Phase 1: Normal Jacobi solve + collect data
            # Compute divergence BEFORE solving (ML input)
            compute_divergence_field(comp, rho0, DRIFT_STIFFNESS)

            # Full Jacobi solve
            for _ in range(NUM_PRESSURE_ITERS):
                pressure_solve_iteration(OVER_RELAX, comp, rho0, DRIFT_STIFFNESS)

            # Collect training data from this frame
            try:
                self._extract_grid_state()
                pressure_target = self._compute_pressure_from_divergence()
                self.collector.collect_from_frame(
                    self._div_np, pressure_target,
                    self._solid_np, self._ct_np,
                    n_samples=ML_SAMPLES_PER_FRAME
                )
            except Exception as e:
                print(f"  [ML] Collection error (frame {self.frame}): {e}")

            # Check if enough data collected
            if self.frame >= ML_COLLECT_FRAMES:
                self.phase = self.PHASE_TRAINING

        elif self.phase == self.PHASE_TRAINING:
            # Phase 2: Train the neural network (one-time)
            # Still do a normal solve this frame
            for _ in range(NUM_PRESSURE_ITERS):
                pressure_solve_iteration(OVER_RELAX, comp, rho0, DRIFT_STIFFNESS)

            try:
                print(f"\n{'='*50}")
                print(f"  [ML] Phase 2: Training neural network...")
                print(f"{'='*50}")

                X_train, y_train = self.collector.get_training_data()
                self.net.train(
                    X_train, y_train,
                    epochs=ML_EPOCHS,
                    batch_size=ML_BATCH_SIZE,
                    print_every=ML_PRINT_LOSS_EVERY,
                )
                self.collector.clear()  # free memory

                print(f"  [ML] Switching to Phase 3: ML-active")
                print(f"  [ML] Jacobi iters: {NUM_PRESSURE_ITERS} → {ML_JACOBI_ITERS_AFTER}")
                print(f"{'='*50}\n")

                self.phase = self.PHASE_ACTIVE

            except Exception as e:
                print(f"  [ML] Training failed: {e}")
                print(f"  [ML] Falling back to pure Jacobi solver")
                self.ml_enabled = False
                self.phase = self.PHASE_ACTIVE

        elif self.phase == self.PHASE_ACTIVE:
            # Phase 3: ML prediction + reduced Jacobi refinement
            ml_applied = False

            if self.ml_enabled and self.net and self.net.trained:
                try:
                    # Compute divergence for ML input
                    compute_divergence_field(comp, rho0, DRIFT_STIFFNESS)
                    self._extract_grid_state()

                    # NN predicts initial pressure field (vectorized)
                    pressure_pred = self.net.predict_pressure_field(
                        self._div_np, self._solid_np, self._ct_np,
                        patch_size=ML_PATCH_SIZE
                    )

                    # Clamp ML predictions: an overconfident network can inject
                    # too much momentum in one step, causing sudden velocity spikes.
                    if ML_PRESSURE_CLIP > 0.0:
                        np.clip(pressure_pred, -ML_PRESSURE_CLIP, ML_PRESSURE_CLIP,
                                out=pressure_pred)

                    # Apply ML-predicted pressure to velocity field
                    apply_pressure_field(pressure_pred)
                    ml_applied = True

                except Exception as e:
                    if self.frame % 300 == 0:
                        print(f"  [ML] Inference error: {e}")

            # Jacobi refinement (fewer iters if ML was applied)
            n_iters = ML_JACOBI_ITERS_AFTER if ml_applied else NUM_PRESSURE_ITERS
            for _ in range(n_iters):
                pressure_solve_iteration(OVER_RELAX, comp, rho0, DRIFT_STIFFNESS)

        # ── 10. Grid → Particle ──
        grid_to_particles(FLIP_RATIO)

        # ── 11. Velocity damping (reduces jitter at rest) ──
        damp_particle_velocities(VELOCITY_DAMPING)

        # ── 12. CFL clamp (prevents particles from jumping multiple cells) ──
        if MAX_PARTICLE_VELOCITY > 0.0:
            clamp_particle_velocities(MAX_PARTICLE_VELOCITY)

        self.frame += 1

    def get_phase_name(self):
        """Return human-readable phase name."""
        names = {
            self.PHASE_COLLECTING: f"Collecting ({self.frame}/{ML_COLLECT_FRAMES})",
            self.PHASE_TRAINING:   "Training...",
            self.PHASE_ACTIVE:     "ML-Active" if (self.ml_enabled and self.net and self.net.trained) else "Jacobi-Only",
        }
        return names.get(self.phase, "Unknown")

    def reset(self):
        """Reset simulation and ML state."""
        init_solid()
        init_particles()
        init_obstacle()
        rest_density_val[None] = 0.0
        self.frame = 0
        if ML_ENABLED:
            self.phase = self.PHASE_COLLECTING
            self.ml_enabled = True
            self.net = PressureNet(
                input_size=ML_PATCH_SIZE ** 2,
                hidden1=ML_HIDDEN_1,
                hidden2=ML_HIDDEN_2,
                lr=ML_LEARNING_RATE,
            )
            self.collector = TrainingDataCollector(
                max_samples=ML_COLLECT_FRAMES * ML_SAMPLES_PER_FRAME,
                patch_size=ML_PATCH_SIZE,
            )


# ═══════════════════════════════════════════════════════════
# RENDERING
# ═══════════════════════════════════════════════════════════

@ti.kernel
def render_frame(show_grid: ti.i32):
    for i, j in pixels:
        t_sky = ti.cast(j, ti.f32) / WINDOW_H
        pixels[i, j] = ti.Vector([
            0.01 + 0.04 * t_sky,
            0.01 + 0.06 * t_sky,
            0.05 + 0.12 * t_sky
        ])

    if show_grid:
        for ci in range(nX):
            for cj in range(nY):
                px0 = int(ci * h / SIM_WIDTH * WINDOW_W)
                py0 = int(cj * h / SIM_HEIGHT * WINDOW_H)
                px1 = int((ci + 1) * h / SIM_WIDTH * WINDOW_W)
                py1 = int((cj + 1) * h / SIM_HEIGHT * WINDOW_H)
                cr = 0.02; cg = 0.02; cb = 0.08
                if cell_type[ci, cj] == SOLID_CELL:
                    cr = 0.25; cg = 0.22; cb = 0.20
                elif cell_type[ci, cj] == FLUID_CELL:
                    cr = 0.0; cg = 0.04; cb = 0.12
                for pi in range(ti.max(0, px0), ti.min(WINDOW_W, px1)):
                    for pj in range(ti.max(0, py0), ti.min(WINDOW_H, py1)):
                        pixels[pi, pj] = ti.Vector([cr, cg, cb])

    ox_px = obs_x[None] / SIM_WIDTH * WINDOW_W
    oy_px = obs_y[None] / SIM_HEIGHT * WINDOW_H
    r_px  = obs_r[None] / SIM_WIDTH * WINDOW_W
    for i, j in pixels:
        dx = ti.cast(i, ti.f32) - ox_px
        dy = ti.cast(j, ti.f32) - oy_px
        d2 = dx * dx + dy * dy
        if d2 < (r_px + 2.0) * (r_px + 2.0):
            if d2 < r_px * r_px:
                dist_norm = ti.sqrt(d2) / r_px
                highlight = ti.max(0.0, 1.0 - ti.sqrt((dx + r_px*0.3)**2 + (dy - r_px*0.3)**2) / (r_px*0.8))
                br = 0.55 - 0.2*dist_norm; bg = 0.12 - 0.05*dist_norm; bb = 0.08 - 0.03*dist_norm
                pixels[i, j] = ti.Vector([
                    ti.min(1.0, br + 0.5*highlight*highlight),
                    ti.min(1.0, bg + 0.35*highlight*highlight),
                    ti.min(1.0, bb + 0.3*highlight*highlight)
                ])
            else:
                pixels[i, j] = ti.Vector([0.35, 0.08, 0.05])

    n = n_part[None]
    rad_px = ti.max(1.0, p_rad / SIM_WIDTH * WINDOW_W)
    for idx in range(n):
        cx = p_x[idx] / SIM_WIDTH * WINDOW_W
        cy = p_y[idx] / SIM_HEIGHT * WINDOW_H
        depth = ti.cast(cy, ti.f32) / WINDOW_H
        speed = ti.sqrt(p_u[idx]**2 + p_v[idx]**2)
        spd_t = ti.min(speed / 5.0, 1.0)

        base_r = 0.02 + 0.14 * depth
        base_g = 0.06 + 0.40 * depth
        base_b = 0.30 + 0.40 * depth
        cr = base_r + (0.90 - base_r) * spd_t * spd_t
        cg = base_g + (0.97 - base_g) * spd_t * spd_t
        cb = base_b + (1.00 - base_b) * spd_t * 0.6
        noise = ti.sin(ti.cast(idx, ti.f32) * 0.1) * 0.02
        cr = ti.max(0.0, ti.min(1.0, cr + noise))
        cg = ti.max(0.0, ti.min(1.0, cg + noise))
        cb = ti.max(0.0, ti.min(1.0, cb + noise * 0.5))

        r_int = ti.cast(ti.ceil(rad_px), ti.i32)
        ci_int = ti.cast(cx, ti.i32); cj_int = ti.cast(cy, ti.i32)
        for di in range(-r_int, r_int + 1):
            for dj in range(-r_int, r_int + 1):
                pi_ = ci_int + di; pj_ = cj_int + dj
                if 0 <= pi_ < WINDOW_W and 0 <= pj_ < WINDOW_H:
                    if di*di + dj*dj <= r_int*r_int:
                        pixels[pi_, pj_] = ti.Vector([cr, cg, cb])

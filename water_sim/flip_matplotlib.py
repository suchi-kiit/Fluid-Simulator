"""
FLIP Fluid Simulation — Matplotlib Renderer (Fallback)
========================================================
Same simulation core as flip_simulation.py, but uses matplotlib
for rendering. Use this if Taichi's GGUI window doesn't work
on your system (e.g., no Vulkan support, WSL, remote desktop).

Usage:  python flip_matplotlib.py
"""

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import time

ti.init(arch=ti.cpu)

# ─── Constants ───
FLUID_CELL = 0
AIR_CELL   = 1
SOLID_CELL = 2

# ─── Parameters (reduced for matplotlib overhead) ───
SIM_W = 2.0
SIM_H = 2.0
GRID_RES = 60
h = SIM_H / GRID_RES
nX = int(SIM_W / h) + 1
nY = GRID_RES + 1

DT = 1.0 / 30.0
GRAVITY = -9.81
FLIP_RATIO = 0.9
OVER_RELAX = 1.9
N_PRESS_ITERS = 30
N_PART_ITERS = 2
p_rad = 0.3 * h

# Particles
_ppd = 2
_sp = h / _ppd
_fx0 = h + 0.01; _fx1 = SIM_W - h - 0.01
_fy0 = h + 0.01; _fy1 = SIM_H * 0.55
MAX_P = int((_fx1 - _fx0) / _sp) * int((_fy1 - _fy0) / _sp) + 256

# Fields
u = ti.field(ti.f32, (nX, nY)); v = ti.field(ti.f32, (nX, nY))
u_prev = ti.field(ti.f32, (nX, nY)); v_prev = ti.field(ti.f32, (nX, nY))
uw = ti.field(ti.f32, (nX, nY)); vw = ti.field(ti.f32, (nX, nY))
solid = ti.field(ti.f32, (nX, nY))
ct = ti.field(ti.i32, (nX, nY))
pd = ti.field(ti.f32, (nX, nY))

px = ti.field(ti.f32, MAX_P); py = ti.field(ti.f32, MAX_P)
pu = ti.field(ti.f32, MAX_P); pv = ti.field(ti.f32, MAX_P)
np_ = ti.field(ti.i32, ())

MAX_PPC = 12
cpc = ti.field(ti.i32, (nX, nY))
cpl = ti.field(ti.i32, (nX, nY, MAX_PPC))

obs_x = ti.field(ti.f32, ()); obs_y = ti.field(ti.f32, ())
obs_r = ti.field(ti.f32, ()); rho0 = ti.field(ti.f32, ())


@ti.kernel
def init_solid():
    for i, j in solid:
        solid[i, j] = 1.0 if (i > 0 and i < nX-1 and j > 0) else 0.0

def init_particles():
    xs, ys = [], []
    y = _fy0
    while y < _fy1:
        x = _fx0
        while x < _fx1:
            xs.append(x); ys.append(y); x += _sp
        y += _sp
    n = min(len(xs), MAX_P)
    np_[None] = n
    arr_x = np.zeros(MAX_P, dtype=np.float32)
    arr_y = np.zeros(MAX_P, dtype=np.float32)
    arr_x[:n] = np.array(xs[:n], dtype=np.float32)
    arr_y[:n] = np.array(ys[:n], dtype=np.float32)
    px.from_numpy(arr_x)
    py.from_numpy(arr_y)
    pu.from_numpy(np.zeros(MAX_P, dtype=np.float32))
    pv.from_numpy(np.zeros(MAX_P, dtype=np.float32))

def init_obs():
    obs_x[None] = SIM_W * 0.5; obs_y[None] = SIM_H * 0.65
    obs_r[None] = SIM_H * 0.08; rho0[None] = 0.0

@ti.kernel
def integrate(dt: ti.f32):
    n = np_[None]
    for i in range(n):
        pv[i] += dt * GRAVITY; px[i] += dt * pu[i]; py[i] += dt * pv[i]

@ti.kernel
def clamp():
    n = np_[None]
    mn_x = h + p_rad; mx_x = (nX-1)*h - p_rad
    mn_y = h + p_rad; mx_y = (nY-1)*h - p_rad
    ox = obs_x[None]; oy = obs_y[None]; orr = obs_r[None]
    for i in range(n):
        x = px[i]; y = py[i]
        if x < mn_x: x = mn_x; pu[i] = 0.0
        if x > mx_x: x = mx_x; pu[i] = 0.0
        if y < mn_y: y = mn_y; pv[i] = 0.0
        if y > mx_y: y = mx_y; pv[i] = 0.0
        dx = x - ox; dy = y - oy; d2 = dx*dx + dy*dy; rp = orr + p_rad
        if d2 < rp*rp:
            d = ti.sqrt(d2)
            if d < 1e-8: dx = 1.0; dy = 0.0; d = 1.0
            x = ox + rp * dx/d; y = oy + rp * dy/d; pu[i] = 0.0; pv[i] = 0.0
        px[i] = x; py[i] = y

@ti.kernel
def build_cl():
    for i, j in cpc: cpc[i, j] = 0
    n = np_[None]
    for idx in range(n):
        ci = ti.cast(px[idx]/h, ti.i32); cj = ti.cast(py[idx]/h, ti.i32)
        ci = ti.max(0, ti.min(ci, nX-1)); cj = ti.max(0, ti.min(cj, nY-1))
        s = ti.atomic_add(cpc[ci, cj], 1)
        if s < MAX_PPC: cpl[ci, cj, s] = idx

@ti.kernel
def push_apart():
    md = 2.0*p_rad; md2 = md*md; n = np_[None]
    for idx in range(n):
        xi = px[idx]; yi = py[idx]
        ci = ti.cast(xi/h, ti.i32); cj = ti.cast(yi/h, ti.i32)
        for di in ti.static(range(-1,2)):
            for dj in ti.static(range(-1,2)):
                ni = ci+di; nj = cj+dj
                if 0 <= ni < nX and 0 <= nj < nY:
                    cnt = ti.min(cpc[ni,nj], MAX_PPC)
                    for k in range(cnt):
                        jdx = cpl[ni,nj,k]
                        if jdx > idx:
                            dx = px[jdx]-xi; dy = py[jdx]-yi; d2 = dx*dx+dy*dy
                            if d2 < md2 and d2 > 1e-12:
                                d = ti.sqrt(d2); sf = 0.5*(md-d)/d
                                px[idx] -= dx*sf; py[idx] -= dy*sf
                                px[jdx] += dx*sf; py[jdx] += dy*sf

@ti.kernel
def mark_obs():
    ox = obs_x[None]; oy = obs_y[None]; rr = obs_r[None]
    for i, j in solid:
        if i > 0 and i < nX-1 and j > 0:
            cx = (i+0.5)*h; cy = (j+0.5)*h
            dx = cx-ox; dy = cy-oy
            solid[i,j] = 0.0 if dx*dx+dy*dy < rr*rr else 1.0

@ti.kernel
def classify():
    for i, j in ct:
        ct[i,j] = SOLID_CELL if solid[i,j] == 0.0 else AIR_CELL
    n = np_[None]
    for idx in range(n):
        ci = ti.cast(px[idx]/h, ti.i32); cj = ti.cast(py[idx]/h, ti.i32)
        if 0 <= ci < nX and 0 <= cj < nY:
            if ct[ci,cj] == AIR_CELL: ct[ci,cj] = FLUID_CELL

@ti.kernel
def p2g():
    for i, j in u: u[i,j]=0;v[i,j]=0;uw[i,j]=0;vw[i,j]=0
    n = np_[None]
    for idx in range(n):
        ppx=px[idx];ppy=py[idx];ppu=pu[idx];ppv=pv[idx]
        ux_=ppx/h;uy_=(ppy-0.5*h)/h;i0=ti.cast(ti.floor(ux_),ti.i32);j0=ti.cast(ti.floor(uy_),ti.i32)
        fx=ux_-i0;fy=uy_-j0
        if 0<=i0<nX-1 and 0<=j0<nY-1:
            w00=(1-fx)*(1-fy);w10=fx*(1-fy);w01=(1-fx)*fy;w11=fx*fy
            ti.atomic_add(u[i0,j0],w00*ppu);ti.atomic_add(uw[i0,j0],w00)
            ti.atomic_add(u[i0+1,j0],w10*ppu);ti.atomic_add(uw[i0+1,j0],w10)
            ti.atomic_add(u[i0,j0+1],w01*ppu);ti.atomic_add(uw[i0,j0+1],w01)
            ti.atomic_add(u[i0+1,j0+1],w11*ppu);ti.atomic_add(uw[i0+1,j0+1],w11)
        vx_=(ppx-0.5*h)/h;vy_=ppy/h;i0v=ti.cast(ti.floor(vx_),ti.i32);j0v=ti.cast(ti.floor(vy_),ti.i32)
        fx2=vx_-i0v;fy2=vy_-j0v
        if 0<=i0v<nX-1 and 0<=j0v<nY-1:
            w00=(1-fx2)*(1-fy2);w10=fx2*(1-fy2);w01=(1-fx2)*fy2;w11=fx2*fy2
            ti.atomic_add(v[i0v,j0v],w00*ppv);ti.atomic_add(vw[i0v,j0v],w00)
            ti.atomic_add(v[i0v+1,j0v],w10*ppv);ti.atomic_add(vw[i0v+1,j0v],w10)
            ti.atomic_add(v[i0v,j0v+1],w01*ppv);ti.atomic_add(vw[i0v,j0v+1],w01)
            ti.atomic_add(v[i0v+1,j0v+1],w11*ppv);ti.atomic_add(vw[i0v+1,j0v+1],w11)

@ti.kernel
def norm_g():
    for i,j in u:
        u[i,j] = u[i,j]/uw[i,j] if uw[i,j]>0 else 0.0
    for i,j in v:
        v[i,j] = v[i,j]/vw[i,j] if vw[i,j]>0 else 0.0

@ti.kernel
def save_vel():
    for i,j in u: u_prev[i,j]=u[i,j]; v_prev[i,j]=v[i,j]

@ti.kernel
def enforce_bc():
    for i,j in u:
        if solid[i,j]==0 or (i>0 and solid[i-1,j]==0): u[i,j]=0
        if solid[i,j]==0 or (j>0 and solid[i,j-1]==0): v[i,j]=0

@ti.kernel
def comp_dens():
    for i,j in pd: pd[i,j]=0
    n = np_[None]
    for idx in range(n):
        cx_=(px[idx]-0.5*h)/h;cy_=(py[idx]-0.5*h)/h
        i0=ti.cast(ti.floor(cx_),ti.i32);j0=ti.cast(ti.floor(cy_),ti.i32)
        fx=cx_-i0;fy=cy_-j0
        if 0<=i0<nX-1 and 0<=j0<nY-1:
            ti.atomic_add(pd[i0,j0],(1-fx)*(1-fy))
            ti.atomic_add(pd[i0+1,j0],fx*(1-fy))
            ti.atomic_add(pd[i0,j0+1],(1-fx)*fy)
            ti.atomic_add(pd[i0+1,j0+1],fx*fy)

@ti.kernel
def avg_dens() -> ti.f32:
    t=0.0;c=0
    for i,j in ct:
        if ct[i,j]==FLUID_CELL: t+=pd[i,j]; c+=1
    r = 0.0
    if c > 0:
        r = t/ti.cast(c,ti.f32)
    return r

@ti.kernel
def pressure_iter(omega:ti.f32, dc:ti.i32, r0:ti.f32):
    for i,j in ct:
        if ct[i,j]!=FLUID_CELL: continue
        if i<1 or i>=nX-1 or j<1 or j>=nY-1: continue
        sl=solid[i-1,j];sr=solid[i+1,j];sb=solid[i,j-1];st=solid[i,j+1]
        ss=sl+sr+sb+st
        if ss<1e-6: continue
        d=u[i+1,j]-u[i,j]+v[i,j+1]-v[i,j]
        if dc>0:
            comp=pd[i,j]-r0
            if comp>0: d-=comp
        p=omega*d/ss
        u[i,j]+=sl*p;u[i+1,j]-=sr*p;v[i,j]+=sb*p;v[i,j+1]-=st*p

@ti.kernel
def g2p(fr:ti.f32):
    n = np_[None]
    for idx in range(n):
        ppx=px[idx];ppy=py[idx]
        ux_=ppx/h;uy_=(ppy-0.5*h)/h;i0=ti.cast(ti.floor(ux_),ti.i32);j0=ti.cast(ti.floor(uy_),ti.i32)
        fx=ux_-i0;fy=uy_-j0
        if 0<=i0<nX-1 and 0<=j0<nY-1:
            w00=(1-fx)*(1-fy);w10=fx*(1-fy);w01=(1-fx)*fy;w11=fx*fy
            pic=w00*u[i0,j0]+w10*u[i0+1,j0]+w01*u[i0,j0+1]+w11*u[i0+1,j0+1]
            prv=w00*u_prev[i0,j0]+w10*u_prev[i0+1,j0]+w01*u_prev[i0,j0+1]+w11*u_prev[i0+1,j0+1]
            pu[idx]=(1-fr)*pic+fr*(pu[idx]+pic-prv)
        vx_=(ppx-0.5*h)/h;vy_=ppy/h;i0v=ti.cast(ti.floor(vx_),ti.i32);j0v=ti.cast(ti.floor(vy_),ti.i32)
        fx2=vx_-i0v;fy2=vy_-j0v
        if 0<=i0v<nX-1 and 0<=j0v<nY-1:
            w00=(1-fx2)*(1-fy2);w10=fx2*(1-fy2);w01=(1-fx2)*fy2;w11=fx2*fy2
            pic=w00*v[i0v,j0v]+w10*v[i0v+1,j0v]+w01*v[i0v,j0v+1]+w11*v[i0v+1,j0v+1]
            prv=w00*v_prev[i0v,j0v]+w10*v_prev[i0v+1,j0v]+w01*v_prev[i0v,j0v+1]+w11*v_prev[i0v+1,j0v+1]
            pv[idx]=(1-fr)*pic+fr*(pv[idx]+pic-prv)


def step():
    integrate(DT)
    for _ in range(N_PART_ITERS): build_cl(); push_apart()
    clamp()
    mark_obs(); classify()
    p2g(); norm_g(); save_vel(); enforce_bc()
    comp_dens()
    r = rho0[None]
    if r == 0: r = avg_dens(); rho0[None] = r
    for _ in range(N_PRESS_ITERS): pressure_iter(OVER_RELAX, 1, r)
    g2p(FLIP_RATIO)


# ═══════════════════════════════════════════════════════════
# MATPLOTLIB ANIMATION
# ═══════════════════════════════════════════════════════════

def main():
    print(f"FLIP Fluid (matplotlib) — Grid: {nX}×{nY}, Particles: ~{MAX_P}")
    init_solid(); init_particles(); init_obs()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, SIM_W); ax.set_ylim(0, SIM_H)
    ax.set_aspect('equal')
    ax.set_facecolor('#0a0a1e')
    fig.patch.set_facecolor('#0a0a1e')
    ax.tick_params(colors='gray')

    scatter = ax.scatter([], [], s=1.2, c=[], cmap='cool', vmin=0, vmax=3)
    circle = plt.Circle((obs_x[None], obs_y[None]), obs_r[None],
                         color='#cc4422', fill=True, zorder=5)
    ax.add_patch(circle)
    title = ax.set_title("FLIP Fluid — Frame 0", color='white', fontsize=12)

    def update(frame):
        t0 = time.perf_counter()
        step()
        dt_ms = (time.perf_counter() - t0) * 1000

        n = np_[None]
        x = px.to_numpy()[:n]
        y = py.to_numpy()[:n]
        u_arr = pu.to_numpy()[:n]
        v_arr = pv.to_numpy()[:n]
        speed = np.sqrt(u_arr**2 + v_arr**2)

        scatter.set_offsets(np.column_stack([x, y]))
        scatter.set_array(speed)

        circle.center = (obs_x[None], obs_y[None])
        title.set_text(f"FLIP Fluid — Frame {frame}  ({dt_ms:.0f} ms/step, {n} particles)")
        return scatter, circle, title

    ani = animation.FuncAnimation(fig, update, frames=None, interval=33, blit=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

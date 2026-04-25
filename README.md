<div align="center">

<!-- Animated header banner using SVG badge trick -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0D1B2A,1B4965,2196F3&height=200&section=header&text=FluidSimulator&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Real-time%20fluid%20dynamics%20%E2%80%94%20CPU-native%2C%20ML-accelerated&descSize=16&descAlignY=60&animation=fadeIn" alt="FluidSimulator" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Taichi](https://img.shields.io/badge/Taichi-1.6%2B-FF6B35?style=flat-square)](https://taichi-lang.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey?style=flat-square)](https://github.com/FluidSimulator)
[![CPU Only](https://img.shields.io/badge/GPU-Not%20Required-orange?style=flat-square)]()

</div>

---

## 〰 What We Build

**FluidSimulator** is an open-source collection of real-time, physically-based fluid simulations — all running smoothly on CPU, no GPU required. Each project implements a different class of fluid phenomenon, combining classical numerical methods (Eulerian grids, MAC staggering, pressure projection) with modern machine learning pipelines to push the boundaries of what's achievable on commodity hardware.

We believe fluid simulation shouldn't require a workstation. Our sims run on laptops.

---

## 🧪 Projects

### 🔥 [Fire Sim](https://github.com/FluidSimulator/Fire_sim) — 2D Eulerian Fire & Smoke
> Real-time fire simulation using the Stable Fluids method. Semi-Lagrangian advection, vorticity confinement, temperature-driven buoyancy, and a cinematic multi-stop color ramp with soft bloom.

- **Method:** Staggered MAC grid · SOR-Jacobi pressure solver · vorticity confinement
- **Performance:** ~65 FPS on Intel i7 (CPU-only)
- **Interactive:** Click to ignite, Space to reset
- **Presets:** Campfire · Inferno · Smoke Column · Vortex Storm

```
pip install -r requirements.txt && python main.py
```

---

### 💧 [Water Sim](https://github.com/FluidSimulator/water_sim) — FLIP + ML Pressure Prediction
> PIC/FLIP water simulator with a from-scratch neural network (NumPy MLP) that learns to predict pressure corrections, cutting Jacobi iterations from 50 to 25.

- **Method:** PIC/FLIP particle-in-cell · Red-Black Gauss-Seidel
- **ML:** 3-layer MLP (~900 params) · Adam optimizer · pure NumPy
- **Pipeline:** Collect → Train → ML-Active (3-phase, fully automatic)
- **No frameworks** — neural net built from scratch with matrix ops only

```
pip install -r requirements.txt && python flip_matplotlib.py
```

---

### 🌬️ [Wind Tunnel](https://github.com/FluidSimulator/Wind_Tunnel) — Eulerian CFD + CNN Super-Resolution
> A 2D Navier-Stokes wind tunnel with a CNN that learns to 2× super-resolve smoke density from simulation data, recovering sharp vortex filaments lost to coarse discretization.

- **Method:** MAC grid · Red-Black Gauss-Seidel (warm-started) · semi-Lagrangian advection
- **ML:** `SmokeUpsampleNet` (~6K params) · PyTorch · async pipelined inference
- **Loss:** MSE + gradient loss (penalises blurry edges)
- **Optimised:** 9 specific tuning passes bringing 15 FPS → 34 FPS

```
pip install taichi torch numpy && python Wind_Tunnel.py
```

---

### 🎨 [Eulerian Paint Sim](https://github.com/FluidSimulator/Eulerian_paint_sim) — Interactive Paint Fluid
> Fluid simulation with interactive paint mixing and advection on an Eulerian grid.

```
# See repo for install instructions
```

---

## 🏗 Shared Architecture

All simulators share a common physics core:

```
MAC Grid (staggered velocity) → Semi-Lagrangian Advection
         ↓
Pressure Projection (Jacobi / Gauss-Seidel)
         ↓
Boundary Conditions + External Forces
         ↓
ML Enhancement (project-specific)
         ↓
Real-Time Render
```

| Concept | Used In |
|---|---|
| Staggered MAC grid | All projects |
| Semi-Lagrangian advection | Fire Sim, Wind Tunnel |
| PIC/FLIP particles | Water Sim |
| SOR / Gauss-Seidel pressure solve | All projects |
| Vorticity confinement | Fire Sim |
| MLP pressure prediction (NumPy) | Water Sim |
| CNN super-resolution (PyTorch) | Wind Tunnel |

---

## ⚡ Quick Start (Any Project)

```bash
# 1. Clone any simulator
git clone https://github.com/FluidSimulator/<repo-name>
cd <repo-name>

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate.bat     # Windows

# 3. Install + run
pip install -r requirements.txt
python main.py
```

> All projects target **Python 3.9+** and run on Linux, Windows, and macOS.

---

## 📚 References

Our work builds on foundational research and tutorials:

| Resource | Relevance |
|---|---|
| [Jos Stam — Stable Fluids (SIGGRAPH 1999)](https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/ns.pdf) | Core advection-projection method |
| [Matthias Müller — Ten Minute Physics](https://matthias-research.github.io/pages/tenMinutePhysics/) | #17 (wind tunnel), #18 (FLIP), #21 (fire) |
| [Robert Bridson — Fluid Simulation for Computer Graphics](https://www.cs.ubc.ca/~rbridson/fluidsimulation/) | MAC grids, pressure solvers |
| [Kingma & Ba — Adam Optimizer (2015)](https://arxiv.org/abs/1412.6980) | ML training |
| [Taichi Programming Language](https://taichi-lang.org) | Parallel CPU kernels |

---

## 🤝 Contributing

We welcome contributions across all repositories. Ideas we're actively interested in:

- **GPU backend** — switching Taichi to `ti.gpu` / `ti.cuda`
- **3D extension** — lifting any solver to 3D on a voxel grid
- **Higher-order advection** — MacCormack / BFECC schemes
- **Multigrid pressure solver** — faster convergence at higher resolutions
- **Screenshot / GIF export** — frame capture pipelines
- **More ML experiments** — reinforcement learning obstacle control, physics-informed networks

Please open an **Issue** first to discuss major changes before submitting a PR.

---

## 📄 License

All repositories are released under the [MIT License](https://opensource.org/licenses/MIT) — free to use, modify, and redistribute with attribution.

---

<div align="center">

*Built with physics, Python, and a healthy obsession with incompressible flow.*

</div>

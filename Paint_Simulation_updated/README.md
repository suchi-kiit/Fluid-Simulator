# Eulerian Fluid Simulator — Paint Pool Edition

An interactive 2D fluid simulation with an ML-accelerated pressure solver. Drag a solid white ball through a pool of deep indigo paint and watch it displace the fluid in real time. Click anywhere in the paint to fire a radial splash impulse. The simulation starts by training a small neural network on its own solver data, then uses it to cut pressure-solve iterations nearly in half.

![Simulation overview: white ball in indigo paint pool](screenshot.png)

---

## Features

- **Eulerian staggered-grid fluid solver** — standard MAC grid with Red-Black Gauss-Seidel pressure projection and RK2 semi-Lagrangian advection
- **Three-phase ML warm-start** — the sim collects its own training data, trains a NumPy MLP on the fly, then uses it to predict a good pressure initialisation each frame, reducing solver iterations from 20 to 8
- **Interactive obstacle** — drag the ball through the paint; it pushes fluid correctly and stamps its velocity into surrounding cells
- **Radial splash impulse** — click anywhere in the paint to create an outward velocity burst
- **Velocity shimmer** — disturbed paint regions brighten to show flow patterns
- **Smooth anti-aliased ball** — sub-pixel distance + smoothstep feathering, no gradient or shading

---

## Requirements

```
Python 3.9+
taichi
numpy
```

Install dependencies:

```bash
pip install taichi numpy
```

---

## Running

```bash
python paint_sim.py
```

The window opens at **800 × 500** pixels. On first launch you will see the three startup phases printed to the console:

```
Phase 1  COLLECTING  ~40 frames   collecting divergence→pressure pairs
Phase 2  TRAINING    ~1-2 seconds one-shot Adam training of the MLP
Phase 3  ML-ACTIVE   ongoing      warm-start pressure prediction active
```

After Phase 2 completes, the solver switches to the faster ML-assisted path automatically.

---

## Controls

| Input | Action |
|---|---|
| **Left-click + drag** on ball | Move the ball through the paint |
| **Left-click** in empty paint | Fire a radial splash impulse |
| **P** | Toggle between paint view and pressure field view |
| **R** | Reset the simulation to its initial state |
| **ESC** or **Q** | Quit |

---

## How It Works

### Fluid Solver

The simulation uses a **staggered MAC grid** (160 × 100 cells). Each frame:

1. **Integrate** — apply body forces to velocity components on cell faces
2. **Pressure projection** — Red-Black Gauss-Seidel iteration enforces incompressibility (∇·u = 0) by computing and applying a pressure correction
3. **Extrapolate** — copy boundary velocities outward so advection doesn't sample from invalid cells
4. **Advect velocity** — RK2 semi-Lagrangian back-trace updates face velocities
5. **Advect paint density** — same RK2 scheme advects the scalar paint field

The obstacle is represented as solid cells in the `s` (solid) field. When the ball moves, `rebuild_solid()` recomputes which cells are inside the circle, and `stamp_obs_vel()` imprints the ball's velocity onto surrounding face samples.

### ML Pressure Warm-Start

The pressure solver converges faster if it starts from a good initial guess. The MLP learns to predict this guess from local divergence patterns.

**Phase 1 — COLLECTING (40 frames)**

Each substep, `collect()` samples up to 1,500 random fluid cells. For each cell it extracts a flattened 3×3 patch of divergence values around it (the input feature) and records the converged pressure at that cell (the label). The entire operation is vectorised with NumPy `sliding_window_view` — no Python loop.

**Phase 2 — TRAINING (one-shot)**

A 3-layer MLP (`9 → 32 → 16 → 1`, ReLU hidden, linear output) is trained for 80 epochs with Adam on the collected pairs. This takes 1-2 seconds on CPU and happens once per session.

**Phase 3 — ML-ACTIVE**

Every substep, the trained network runs inference over all grid cells in one vectorised batch, producing a full pressure field prediction. This is written into `p` as the warm-start, and then only 8 Gauss-Seidel iterations are needed to converge instead of 20 — a ~60% reduction in solver work.

### Render Pipeline

The `img` field is sized to the full window resolution (800 × 500). Each pixel maps back to the nearest grid cell via integer scaling. The ball edge is anti-aliased using a smoothstep blend over a 1.5-pixel feather zone computed in world-space, giving a smooth circle without any upscaling or post-processing. `gui.set_image(img)` hands the Taichi field directly to the GUI — no NumPy round-trip.

---

## Performance Notes

Target frame rate is **25-30 FPS** on a modern CPU. Key decisions that achieve this:

| Setting | Value | Reason |
|---|---|---|
| `cpu_max_num_threads` | 1 | Sequential is faster than parallel at this grid size |
| `SUBSTEPS` | 1 | Single substep per frame is sufficient |
| `NUM_ITERS` | 20 | Full solver iteration count (phases 1 & fallback) |
| `ML_ITERS` | 8 | Warm-start needs fewer correction steps |
| `collect()` | Vectorised NumPy | Eliminates a 1,500-iteration Python loop (~75 ms → ~1 ms) |
| Rendering | Direct field hand-off | No `cv2`, no `np.flip`, no Python array work per frame |

---

## Configuration

All tunable constants are at the top of the file:

```python
NX, NY            = 160, 100     # grid resolution
WIN_SCALE         = 5            # window = grid × scale  (800×500)
NUM_ITERS         = 20           # full pressure solver iterations
ML_ITERS          = 8            # iterations after ML warm-start
COLLECT_FRAMES    = 40           # frames to spend in data collection
SAMPLES_PER_FRAME = 1500         # training samples gathered per frame
ML_EPOCHS         = 80           # training epochs
OBS_R             = NY * H * 0.14  # ball radius (fraction of domain height)
```

Increasing `NX`/`NY` gives a more detailed simulation but reduces FPS. Increasing `COLLECT_FRAMES` or `ML_EPOCHS` may improve ML accuracy at the cost of a longer startup phase.

---

## Project Structure

```
paint_sim.py   — single-file simulation (all physics, ML, rendering)
README.md      — this file
```

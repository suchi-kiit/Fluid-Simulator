"""
FLIP Fluid Simulation — Main Entry Point
==========================================
Run with:  python main.py

Controls:
  LMB drag  — move obstacle
  R         — reset simulation + ML
  SPACE     — pause / resume
  G         — toggle grid overlay
  M         — toggle ML on/off
  ESC / Q   — quit
"""

import taichi as ti
import time

# Taichi MUST be initialized before importing sim (which creates fields)
ti.init(arch=ti.cpu)

from src.config import *
from src.sim import (
    FlipSimulation, init_solid, init_particles, init_obstacle,
    render_frame, pixels, obs_x, obs_y, obs_vx, obs_vy,
    nX, nY, n_part, MAX_PARTICLES
)


def main():
    print("=" * 60)
    print("  FLIP Fluid Simulation + ML Pressure Solver")
    print(f"  Grid: {nX}×{nY}  |  Max Particles: ~{MAX_PARTICLES}")
    print(f"  ML: {'Enabled' if ML_ENABLED else 'Disabled'}")
    if ML_ENABLED:
        print(f"  ML phases: Collect({ML_COLLECT_FRAMES} frames) → Train → Active")
        print(f"  Network: Input(9) → Dense({ML_HIDDEN_1},ReLU) → Dense({ML_HIDDEN_2},ReLU) → Dense(1)")
    print("  Controls: LMB=drag  R=reset  SPACE=pause  G=grid  M=ML  ESC=quit")
    print("=" * 60)

    # Initialize
    init_solid()
    init_particles()
    init_obstacle()

    sim = FlipSimulation()

    window = ti.ui.Window("FLIP Fluid + ML (CPU)", (WINDOW_W, WINDOW_H), vsync=True)
    canvas = window.get_canvas()

    paused = False
    show_grid = False
    frame = 0

    while window.running:
        # ── Events ──
        for e in window.get_events(ti.ui.PRESS):
            if e.key in (ti.ui.ESCAPE, 'q'):
                window.running = False
            elif e.key == 'r':
                sim.reset()
                frame = 0
            elif e.key == ti.ui.SPACE:
                paused = not paused
            elif e.key == 'g':
                show_grid = not show_grid
            elif e.key == 'm':
                if sim.ml_enabled:
                    sim.ml_enabled = False
                    print("  [ML] Disabled — using pure Jacobi")
                else:
                    sim.ml_enabled = True
                    print("  [ML] Re-enabled")

        # ── Mouse interaction ──
        mx, my = window.get_cursor_pos()
        if window.is_pressed(ti.ui.LMB):
            new_x = mx * SIM_WIDTH
            new_y = my * SIM_HEIGHT
            obs_vx[None] = (new_x - obs_x[None]) / max(DT, 1e-6)
            obs_vy[None] = (new_y - obs_y[None]) / max(DT, 1e-6)
            obs_x[None] = new_x
            obs_y[None] = new_y
        else:
            # Smooth decay instead of instant snap to zero
            # Prevents shockwave when user releases mouse after fast drag
            obs_vx[None] *= OBSTACLE_VELOCITY_DECAY
            obs_vy[None] *= OBSTACLE_VELOCITY_DECAY
            # Kill very small residual velocities
            if abs(obs_vx[None]) < 0.01:
                obs_vx[None] = 0.0
            if abs(obs_vy[None]) < 0.01:
                obs_vy[None] = 0.0

        # ── Simulate ──
        if not paused:
            t0 = time.perf_counter()
            sim.step()
            dt_ms = (time.perf_counter() - t0) * 1000

            if frame % 60 == 0:
                phase = sim.get_phase_name()
                print(f"  Frame {frame:4d}  sim: {dt_ms:6.1f} ms  "
                      f"particles: {n_part[None]}  phase: {phase}")

        # ── Render ──
        render_frame(1 if show_grid else 0)
        canvas.set_image(pixels)
        window.show()
        frame += 1


if __name__ == "__main__":
    main()

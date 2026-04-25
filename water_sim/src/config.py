"""
Configuration for FLIP Fluid Simulation + ML Pressure Solver
=============================================================
All tunable parameters in one place.
"""

# ═══════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════
WINDOW_W = 800
WINDOW_H = 800

# ═══════════════════════════════════════════════════════════
# SIMULATION DOMAIN
# ═══════════════════════════════════════════════════════════
SIM_HEIGHT = 3.0
SIM_WIDTH  = SIM_HEIGHT * (WINDOW_W / WINDOW_H)

# ═══════════════════════════════════════════════════════════
# GRID
# ═══════════════════════════════════════════════════════════
GRID_RES = 80          # cells along height (lower = faster)

# ═══════════════════════════════════════════════════════════
# TIME
# ═══════════════════════════════════════════════════════════
DT = 1.0 / 40.0

# ═══════════════════════════════════════════════════════════
# PHYSICS
# ═══════════════════════════════════════════════════════════
GRAVITY            = -9.81
FLIP_RATIO         = 0.6     # 0 = full PIC (smooth), 1 = full FLIP (noisy). 0.6 is more stable than 0.7
OVER_RELAX         = 1.7     # Gauss-Seidel overrelaxation (lowered from 1.8 → less overshoot)
NUM_PRESSURE_ITERS = 50      # More iterations → better convergence → less jitter
NUM_PARTICLE_ITERS = 2       # push-apart iterations
COMPENSATE_DRIFT   = True
DRIFT_STIFFNESS    = 0.05    # How strongly drift correction pushes (0.05 = gentle, 1.0 = original aggressive)
SEPARATE_PARTICLES = True
VELOCITY_DAMPING   = 0.995   # Per-frame velocity decay (increased damping from 0.998 → reduces jitter)

# CFL velocity clamp — set high enough to not trap energy during settling
MAX_PARTICLE_VELOCITY = 3.0  # m/s  (raised from 1.2 — too tight was trapping energy)

# Obstacle velocity smoothing — when user releases mouse, obstacle velocity
# decays gradually instead of snapping to zero (prevents shockwave)
OBSTACLE_VELOCITY_DECAY = 0.85  # per-frame decay (0.85 = smooth stop over ~10 frames)

# Particle sizing
PARTICLE_RADIUS_FACTOR = 0.3  # fraction of cell size h

# Particle fill region (fraction of domain)
FILL_WIDTH_FRAC  = 1.0       # fill full width
FILL_HEIGHT_FRAC = 0.55      # fill lower 55%

# Obstacle defaults
OBSTACLE_X_FRAC = 0.5
OBSTACLE_Y_FRAC = 0.65
OBSTACLE_R_FRAC = 0.08       # fraction of SIM_HEIGHT

# ═══════════════════════════════════════════════════════════
# ML PRESSURE SOLVER
# ═══════════════════════════════════════════════════════════
ML_ENABLED = True

# --- Data collection (Phase 1) ---
ML_COLLECT_FRAMES    = 40       # how many frames to collect training data
ML_SAMPLES_PER_FRAME = 1500    # randomly subsampled cells per frame

# --- Network architecture ---
ML_PATCH_SIZE     = 3          # 3×3 input patch of divergence values
ML_HIDDEN_1       = 32         # first hidden layer neurons
ML_HIDDEN_2       = 16         # second hidden layer neurons
# Input:  PATCH_SIZE² = 9 features
# Output: 1 (predicted pressure correction for center cell)
# Total params: (9×32+32) + (32×16+16) + (16×1+1) = 320+32+512+16+16+1 = 897

# --- Training (Phase 2) ---
ML_EPOCHS         = 80         # training epochs
ML_LEARNING_RATE  = 0.001      # Adam initial learning rate
ML_BATCH_SIZE     = 256        # mini-batch size for SGD
ML_ADAM_BETA1     = 0.9        # Adam momentum
ML_ADAM_BETA2     = 0.999      # Adam RMSprop
ML_ADAM_EPS       = 1e-8       # Adam epsilon

# --- Inference (Phase 3) ---
ML_JACOBI_ITERS_AFTER = 25    # reduced Jacobi refinement after ML prediction
ML_PRESSURE_CLIP      = 0.4   # clamp ML pressure predictions to ±this value (prevents energy injection spikes)

# --- Logging ---
ML_PRINT_LOSS_EVERY = 10       # print loss every N epochs during training

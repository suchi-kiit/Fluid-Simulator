# FLIP Fluid Simulation + ML-Accelerated Pressure Solver

A CPU-based PIC/FLIP water simulator with a **from-scratch neural network**
that learns to predict pressure corrections, reducing the number of expensive
Jacobi solver iterations needed each frame.

## Quick Start

```bash
pip install -r requirements.txt
python flip_ml_single.py
```

---

## Controls

| Key / Input     | Action                           |
|-----------------|----------------------------------|
| **LMB drag**    | Move obstacle through water      |
| **R**           | Reset simulation + retrain ML    |
| **Space**       | Pause / resume                   |
| **G**           | Toggle grid overlay              |
| **M**           | Toggle ML on/off                 |
| **Q / ESC**     | Quit                             |

---

## Files

| File                     | Description                                    |
|--------------------------|------------------------------------------------|
| `flip_ml_single.py`     | Complete simulation — single runnable file     |
| `requirements.txt`      | Python dependencies                            |
| `README.md`             | This file                                      |

---

## ML Integration Details

### Architecture: 3-Layer MLP (NumPy from Scratch)

```
Input(9) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear)
```

- **Input**: 3×3 patch of divergence values around each cell (flattened to 9 features)
- **Output**: predicted pressure correction for the center cell
- **Parameters**: ~897 trainable weights and biases
- **No ML frameworks** — pure NumPy matrix operations

### ML Concepts Demonstrated

| Concept                    | Implementation                                              |
|----------------------------|-------------------------------------------------------------|
| Feedforward Neural Network | 3-layer MLP with configurable hidden sizes                  |
| He/Kaiming Initialization  | `W * sqrt(2/fan_in)` — prevents vanishing gradients         |
| ReLU Activation            | `max(0, x)` between hidden layers                           |
| Forward Propagation        | Matrix multiplications: `X @ W + b` through 3 layers       |
| MSE Loss Function          | `mean((y_pred - y_true)²)`                                  |
| Backpropagation            | Chain rule gradients through all 3 layers                   |
| Adam Optimizer             | Adaptive learning rates with momentum + bias correction     |
| Mini-batch SGD             | Shuffled random subsets each epoch                          |
| Feature Normalization      | Zero-mean, unit-variance input scaling                      |
| Batched Inference          | `sliding_window_view` for vectorized patch extraction       |

### 3-Phase Simulation Pipeline

```
Phase 1: COLLECTING (frames 0–39)
├── Normal Jacobi solver runs (50 iterations)
├── Randomly subsample 1,500 fluid cells per frame
└── Store (divergence_patch, pressure_correction) pairs

Phase 2: TRAINING (one-time, ~1 second)
├── Concatenate all collected data (~60,000 samples)
├── Train neural network (80 epochs, Adam optimizer)
└── Print training loss every 10 epochs

Phase 3: ML-ACTIVE (all subsequent frames)
├── Neural network predicts initial pressure field
├── Apply predictions to velocity field
├── Only 25 Jacobi iterations refine (down from 50)
└── try/except — falls back silently on any error
```

### Performance Optimizations

| Optimization                          | Savings                             |
|---------------------------------------|-------------------------------------|
| Vectorized ML inference               | Replaces nX×nY Python loop with 1 matmul |
| `sliding_window_view` patch extract   | Zero-copy strided view, no allocation |
| Pre-allocated NumPy buffers           | No allocation per frame for ML I/O  |
| Reduced Jacobi iterations (50 → 25)  | 50% fewer pressure solve iterations |
| Velocity damping (0.998)              | Reduces jitter at rest              |
| Error protection (try/except)         | Never crashes, falls back to Jacobi |

---

## Configuration

All parameters are at the top of `flip_ml_single.py`. Key settings:

| Parameter                | Default | Description                          |
|--------------------------|---------|--------------------------------------|
| `ML_ENABLED`             | True    | Enable/disable ML pipeline           |
| `ML_COLLECT_FRAMES`      | 40      | Frames to collect training data      |
| `ML_SAMPLES_PER_FRAME`   | 1500    | Cells subsampled per frame           |
| `ML_HIDDEN_1`            | 32      | First hidden layer neurons           |
| `ML_HIDDEN_2`            | 16      | Second hidden layer neurons          |
| `ML_EPOCHS`              | 80      | Training epochs                      |
| `ML_LEARNING_RATE`       | 0.001   | Adam learning rate                   |
| `ML_BATCH_SIZE`          | 256     | Mini-batch size                      |
| `ML_JACOBI_ITERS_AFTER`  | 25      | Jacobi iterations after ML predict   |
| `NUM_PRESSURE_ITERS`     | 50      | Jacobi iterations (Phase 1/fallback) |
| `FLIP_RATIO`             | 0.7     | PIC/FLIP blend (0=smooth, 1=noisy)   |
| `VELOCITY_DAMPING`       | 0.998   | Per-frame velocity decay             |

---

## System Requirements

- **Python**: 3.8 – 3.13
- **OS**: Windows 10/11, Linux, macOS (64-bit)
- **RAM**: 4 GB minimum
- **CPU**: Any modern x86-64 (optimized for Intel 12th Gen)
- **GPU**: Not required

---

## References

- Müller, M. — [Ten Minute Physics #18: FLIP Water Simulator]
- Bridson, R. — *Fluid Simulation for Computer Graphics*
- Kingma & Ba — "Adam: A Method for Stochastic Optimization" (2015)
- He et al. — "Delving Deep into Rectifiers" (Kaiming Init, 2015)

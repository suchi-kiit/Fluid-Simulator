"""
ML Pressure Solver — Neural Network from Scratch (NumPy only)
==============================================================
A 3-layer MLP that learns to predict pressure corrections from
local divergence patches, replacing part of the iterative Jacobi
solver with a learned initial guess.

Architecture:
  Input(9) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear)
  Total trainable parameters: 897

Concepts demonstrated:
  - Feedforward neural network (MLP)
  - He/Kaiming weight initialization
  - ReLU activation
  - Forward propagation
  - MSE loss
  - Backpropagation (chain rule through 3 layers)
  - Adam optimizer (adaptive learning rates + momentum)
  - Mini-batch SGD
  - Feature normalization (zero-mean, unit-variance)
  - Batched inference via sliding_window_view
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import time


class PressureNet:
    """
    3-layer MLP for pressure prediction.

    Input:  3×3 patch of divergence values (flattened to 9)
    Output: scalar pressure correction for center cell
    """

    def __init__(self, input_size=9, hidden1=32, hidden2=16, output_size=1,
                 lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # ── He/Kaiming initialization ──
        # Prevents vanishing gradients with ReLU activations
        self.W1 = np.random.randn(input_size, hidden1).astype(np.float32) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden1, dtype=np.float32)

        self.W2 = np.random.randn(hidden1, hidden2).astype(np.float32) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2, dtype=np.float32)

        self.W3 = np.random.randn(hidden2, output_size).astype(np.float32) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(output_size, dtype=np.float32)

        # ── Adam optimizer state (momentum + RMSprop for each parameter) ──
        self._init_adam()

        # ── Feature normalization stats ──
        self.feat_mean = np.zeros(input_size, dtype=np.float32)
        self.feat_std  = np.ones(input_size, dtype=np.float32)

        # ── Training state ──
        self.trained = False
        self.adam_t = 0  # timestep counter for bias correction

        self._count_params()

    def _init_adam(self):
        """Initialize Adam optimizer first/second moment estimates."""
        self.mW1 = np.zeros_like(self.W1)
        self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.vb1 = np.zeros_like(self.b1)

        self.mW2 = np.zeros_like(self.W2)
        self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)
        self.vb2 = np.zeros_like(self.b2)

        self.mW3 = np.zeros_like(self.W3)
        self.vW3 = np.zeros_like(self.W3)
        self.mb3 = np.zeros_like(self.b3)
        self.vb3 = np.zeros_like(self.b3)

    def _count_params(self):
        """Count total trainable parameters."""
        self.n_params = (
            self.W1.size + self.b1.size +
            self.W2.size + self.b2.size +
            self.W3.size + self.b3.size
        )

    # ═══════════════════════════════════════════════════════
    # FORWARD PROPAGATION
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def _relu(x):
        """ReLU activation: max(0, x)"""
        return np.maximum(0, x)

    @staticmethod
    def _relu_deriv(x):
        """Derivative of ReLU: 1 if x > 0, else 0"""
        return (x > 0).astype(np.float32)

    def _normalize_input(self, X):
        """Apply zero-mean, unit-variance normalization."""
        return (X - self.feat_mean) / (self.feat_std + 1e-8)

    def forward(self, X):
        """
        Forward pass through all 3 layers.

        Args:
            X: (batch_size, 9) input patches
        Returns:
            output: (batch_size, 1) predicted pressure corrections
        """
        X_norm = self._normalize_input(X)

        # Layer 1: Input → Hidden1
        self._z1 = X_norm @ self.W1 + self.b1     # (batch, 32)
        self._a1 = self._relu(self._z1)            # (batch, 32)

        # Layer 2: Hidden1 → Hidden2
        self._z2 = self._a1 @ self.W2 + self.b2   # (batch, 16)
        self._a2 = self._relu(self._z2)            # (batch, 16)

        # Layer 3: Hidden2 → Output (linear, no activation)
        self._z3 = self._a2 @ self.W3 + self.b3   # (batch, 1)

        return self._z3

    def predict(self, X):
        """Forward pass without storing intermediates (inference only)."""
        X_norm = self._normalize_input(X)
        a1 = self._relu(X_norm @ self.W1 + self.b1)
        a2 = self._relu(a1 @ self.W2 + self.b2)
        return a2 @ self.W3 + self.b3

    # ═══════════════════════════════════════════════════════
    # LOSS FUNCTION
    # ═══════════════════════════════════════════════════════

    @staticmethod
    def mse_loss(y_pred, y_true):
        """Mean Squared Error loss."""
        diff = y_pred - y_true
        return np.mean(diff * diff)

    # ═══════════════════════════════════════════════════════
    # BACKPROPAGATION
    # ═══════════════════════════════════════════════════════

    def backward(self, X, y_true):
        """
        Backpropagation: compute gradients for all parameters
        using the chain rule through 3 layers.

        Args:
            X:      (batch, 9)  — input (raw, will be normalized)
            y_true: (batch, 1)  — target pressure corrections
        Returns:
            loss: scalar MSE loss
        """
        batch_size = X.shape[0]
        X_norm = self._normalize_input(X)

        # Forward (stores intermediates in self._z1, _a1, etc.)
        y_pred = self.forward(X)
        loss = self.mse_loss(y_pred, y_true)

        # ── Output layer gradient ──
        # d_loss/d_z3 = 2/N * (y_pred - y_true) for MSE
        dz3 = (2.0 / batch_size) * (y_pred - y_true)   # (batch, 1)
        dW3 = self._a2.T @ dz3                           # (16, 1)
        db3 = np.sum(dz3, axis=0)                         # (1,)

        # ── Hidden layer 2 gradient ──
        da2 = dz3 @ self.W3.T                             # (batch, 16)
        dz2 = da2 * self._relu_deriv(self._z2)           # (batch, 16)
        dW2 = self._a1.T @ dz2                            # (32, 16)
        db2 = np.sum(dz2, axis=0)                          # (16,)

        # ── Hidden layer 1 gradient ──
        da1 = dz2 @ self.W2.T                             # (batch, 32)
        dz1 = da1 * self._relu_deriv(self._z1)           # (batch, 32)
        dW1 = X_norm.T @ dz1                              # (9, 32)
        db1 = np.sum(dz1, axis=0)                          # (32,)

        # Store gradients for optimizer
        self._grads = {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2,
            'W3': dW3, 'b3': db3,
        }

        return loss

    # ═══════════════════════════════════════════════════════
    # ADAM OPTIMIZER STEP
    # ═══════════════════════════════════════════════════════

    def _adam_update(self, param, grad, m, v_):
        """
        One Adam optimizer step for a single parameter.

        Adam combines momentum (exponential moving average of gradients)
        with RMSprop (exponential moving average of squared gradients),
        plus bias correction for the initial steps.
        """
        m[:] = self.beta1 * m + (1 - self.beta1) * grad
        v_[:] = self.beta2 * v_ + (1 - self.beta2) * (grad * grad)

        # Bias correction (important for early steps)
        m_hat = m / (1 - self.beta1 ** self.adam_t)
        v_hat = v_ / (1 - self.beta2 ** self.adam_t)

        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def optimizer_step(self):
        """Apply Adam update to all parameters."""
        self.adam_t += 1
        g = self._grads

        self._adam_update(self.W1, g['W1'], self.mW1, self.vW1)
        self._adam_update(self.b1, g['b1'], self.mb1, self.vb1)
        self._adam_update(self.W2, g['W2'], self.mW2, self.vW2)
        self._adam_update(self.b2, g['b2'], self.mb2, self.vb2)
        self._adam_update(self.W3, g['W3'], self.mW3, self.vW3)
        self._adam_update(self.b3, g['b3'], self.mb3, self.vb3)

    # ═══════════════════════════════════════════════════════
    # TRAINING
    # ═══════════════════════════════════════════════════════

    def fit_normalization(self, X_all):
        """Compute feature normalization stats from training data."""
        self.feat_mean = np.mean(X_all, axis=0).astype(np.float32)
        self.feat_std  = np.std(X_all, axis=0).astype(np.float32)
        # Prevent division by zero for constant features
        self.feat_std[self.feat_std < 1e-8] = 1.0

    def train(self, X_train, y_train, epochs=80, batch_size=256,
              print_every=10):
        """
        Train the network using mini-batch SGD with Adam.

        Args:
            X_train:    (N, 9)  input patches
            y_train:    (N, 1)  target pressure corrections
            epochs:     number of full passes through the data
            batch_size: mini-batch size
            print_every: log loss every N epochs
        """
        N = X_train.shape[0]
        if N == 0:
            print("  [ML] No training data, skipping.")
            return

        # Fit normalization on training data
        self.fit_normalization(X_train)

        print(f"  [ML] Training: {N} samples, {self.n_params} params, "
              f"{epochs} epochs, batch={batch_size}")
        t0 = time.perf_counter()

        for epoch in range(epochs):
            # Shuffle data each epoch
            perm = np.random.permutation(N)
            X_shuf = X_train[perm]
            y_shuf = y_train[perm]

            epoch_loss = 0.0
            n_batches = 0

            # Mini-batch loop
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]

                loss = self.backward(X_batch, y_batch)
                self.optimizer_step()

                epoch_loss += loss
                n_batches += 1

            if (epoch + 1) % print_every == 0 or epoch == 0:
                avg = epoch_loss / max(n_batches, 1)
                print(f"    Epoch {epoch+1:3d}/{epochs}  loss: {avg:.6f}")

        dt = time.perf_counter() - t0
        self.trained = True
        print(f"  [ML] Training complete in {dt:.2f}s")

    # ═══════════════════════════════════════════════════════
    # BATCHED INFERENCE (for full pressure field)
    # ═══════════════════════════════════════════════════════

    def predict_pressure_field(self, divergence_2d, solid_2d, cell_type_2d,
                               patch_size=3):
        """
        Predict initial pressure field from divergence using the trained
        network. Uses sliding_window_view for vectorized extraction of
        all 3×3 patches in a single operation.

        Args:
            divergence_2d: (nX, nY) divergence field
            solid_2d:      (nX, nY) solid mask (0=solid, 1=open)
            cell_type_2d:  (nX, nY) cell types (0=fluid)
            patch_size:    patch side length (3)

        Returns:
            pressure_2d:   (nX, nY) predicted pressure corrections
        """
        if not self.trained:
            return np.zeros_like(divergence_2d)

        nX, nY = divergence_2d.shape
        pad = patch_size // 2  # 1 for 3×3

        # Pad divergence field with zeros at boundaries
        div_padded = np.pad(divergence_2d, pad, mode='constant',
                            constant_values=0.0)

        # Extract ALL 3×3 patches at once using stride tricks
        # Shape: (nX, nY, 3, 3)
        windows = sliding_window_view(div_padded, (patch_size, patch_size))

        # Flatten patches to (nX*nY, 9)
        patches_flat = windows.reshape(-1, patch_size * patch_size).astype(np.float32)

        # Single vectorized forward pass for ALL cells
        predictions = self.predict(patches_flat)  # (nX*nY, 1)

        # Reshape back to grid
        pressure = predictions.reshape(nX, nY)

        # Zero out non-fluid cells
        pressure[cell_type_2d != 0] = 0.0   # 0 = FLUID_CELL
        pressure[solid_2d == 0.0] = 0.0

        return pressure.astype(np.float32)


# ═══════════════════════════════════════════════════════════
# DATA COLLECTION UTILITIES
# ═══════════════════════════════════════════════════════════

class TrainingDataCollector:
    """
    Collects (divergence_patch, pressure_correction) pairs during
    Phase 1 of simulation by randomly subsampling fluid cells.
    """

    def __init__(self, max_samples=60000, patch_size=3):
        self.patch_size = patch_size
        self.max_samples = max_samples
        self.X_list = []   # list of (N, 9) arrays
        self.y_list = []   # list of (N, 1) arrays
        self.total = 0

    def collect_from_frame(self, divergence_2d, pressure_2d, solid_2d,
                           cell_type_2d, n_samples=1500):
        """
        Subsample n_samples fluid cells from the current frame and
        extract their 3×3 divergence patches + pressure corrections.

        Args:
            divergence_2d: (nX, nY) divergence after P→G transfer
            pressure_2d:   (nX, nY) pressure after Jacobi solve
            solid_2d:      (nX, nY) solid mask
            cell_type_2d:  (nX, nY) cell types
            n_samples:     how many cells to sample this frame
        """
        nX, nY = divergence_2d.shape
        pad = self.patch_size // 2

        # Find fluid cells (type 0) not on boundary
        fluid_mask = (cell_type_2d == 0) & (solid_2d > 0)
        fluid_mask[:pad, :] = False
        fluid_mask[-pad:, :] = False
        fluid_mask[:, :pad] = False
        fluid_mask[:, -pad:] = False

        fluid_ij = np.argwhere(fluid_mask)
        if len(fluid_ij) == 0:
            return

        # Random subsample
        n = min(n_samples, len(fluid_ij))
        if self.total >= self.max_samples:
            return
        n = min(n, self.max_samples - self.total)

        indices = np.random.choice(len(fluid_ij), size=n, replace=False)
        sampled = fluid_ij[indices]

        # Pad divergence for patch extraction
        div_padded = np.pad(divergence_2d, pad, mode='constant',
                            constant_values=0.0)

        # Extract patches and targets
        patches = np.zeros((n, self.patch_size * self.patch_size),
                           dtype=np.float32)
        targets = np.zeros((n, 1), dtype=np.float32)

        for k in range(n):
            i, j = sampled[k]
            # Patch from padded array (offset by pad)
            patch = div_padded[i:i + self.patch_size,
                               j:j + self.patch_size]
            patches[k] = patch.flatten()
            targets[k, 0] = pressure_2d[i, j]

        self.X_list.append(patches)
        self.y_list.append(targets)
        self.total += n

    def get_training_data(self):
        """Concatenate all collected data into single arrays."""
        if not self.X_list:
            return np.zeros((0, self.patch_size**2), dtype=np.float32), \
                   np.zeros((0, 1), dtype=np.float32)
        X = np.concatenate(self.X_list, axis=0)
        y = np.concatenate(self.y_list, axis=0)
        return X, y

    def clear(self):
        """Free collected data after training."""
        self.X_list.clear()
        self.y_list.clear()
        self.total = 0

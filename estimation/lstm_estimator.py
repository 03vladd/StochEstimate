"""
LSTM-based Ornstein-Uhlenbeck Parameter Estimator

Treats OU parameter estimation as supervised regression:
- Trained on synthetic OU paths with known parameters
- At inference time, takes a fixed-length spread window and outputs (θ, μ, σ)
  with uncertainty estimates via Monte Carlo Dropout.

Model architecture:
  LSTM(input=1, hidden=64, layers=2, dropout=0.2)
  Linear(64 → 32) + ReLU
  Linear(32 → 3) → (log_θ, μ_norm, log_σ_norm)

Normalization strategy:
  Input X_norm = (X - mean(X)) / std(X) before feeding to LSTM
  De-normalization at output:
    θ_real = θ          (scale-invariant)
    μ_real = mean + μ_norm × std
    σ_real = σ_norm × std
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic_data.ou_generator import generate_ou_process


@dataclass
class OULSTMResult:
    """
    Result of LSTM OU parameter estimation.
    Mirrors OUEstimationResult from mle.py for drop-in comparison.
    """
    theta: float        # Speed of mean reversion (per day)
    mu: float           # Long-term mean level
    sigma: float        # Volatility (diffusion coefficient)

    # 95% CI from MC Dropout samples (2.5th / 97.5th percentiles)
    theta_ci: tuple
    mu_ci: tuple
    sigma_ci: tuple

    # MC standard deviation (uncertainty magnitude)
    theta_std: float
    mu_std: float
    sigma_std: float

    # Raw MC draws for thesis plots
    mc_draws: dict = field(default_factory=dict)
    n_mc_samples: int = 200

    @property
    def detailed_result(self) -> str:
        """Human-readable result summary matching MLE style"""
        half_life = f"{0.693147 / self.theta:.2f}" if self.theta > 0 else "N/A"
        lines = [
            f"OU PARAMETER ESTIMATION (LSTM with MC Dropout)",
            f"",
            f"Parameter Estimates:",
            f"  θ (mean reversion speed): {self.theta:.6f} per day",
            f"    95% CI: [{self.theta_ci[0]:.6f}, {self.theta_ci[1]:.6f}]",
            f"    Std Dev (MC): {self.theta_std:.6f}",
            f"    Interpretation: {half_life} days for 63% reversion",
            f"",
            f"  μ (long-term mean): {self.mu:.6f}",
            f"    95% CI: [{self.mu_ci[0]:.6f}, {self.mu_ci[1]:.6f}]",
            f"    Std Dev (MC): {self.mu_std:.6f}",
            f"",
            f"  σ (volatility): {self.sigma:.6f}",
            f"    95% CI: [{self.sigma_ci[0]:.6f}, {self.sigma_ci[1]:.6f}]",
            f"    Std Dev (MC): {self.sigma_std:.6f}",
            f"",
            f"MC Dropout: {self.n_mc_samples} forward passes",
        ]
        return "\n".join(lines)


@dataclass
class TrainingHistory:
    """Records per-epoch training progress"""
    train_losses: list = field(default_factory=list)
    val_losses: list = field(default_factory=list)
    epochs: int = 0
    final_train_loss: float = float('inf')
    final_val_loss: float = float('inf')


class OULSTMNet(nn.Module):
    """
    LSTM network for OU parameter estimation.

    Input:  (batch, window_size, 1) — normalized spread values
    Output: (batch, 3) — (log_θ, μ_norm, log_σ_norm)
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]   # (batch, hidden_size)
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out  # (batch, 3): (log_θ, μ_norm, log_σ_norm)


class OULSTMEstimator:
    """
    LSTM-based OU parameter estimator.

    Trained on synthetic OU paths, estimates (θ, μ, σ) from a fixed-length
    spread window with uncertainty quantification via Monte Carlo Dropout.
    """

    def __init__(self, window_size: int = 126, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.device = torch.device('cpu')
        self.model = OULSTMNet(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.model.to(self.device)
        self.is_trained = False

    def _generate_training_data(self, n_samples: int, theta_range: tuple,
                                sigma_range: tuple, seed: int):
        """
        Generate synthetic OU paths for training.

        Returns:
            X: (n_samples, window_size, 1) normalized inputs
            y: (n_samples, 3) targets in log-space: (log_θ, μ_norm, log_σ_norm)
        """
        rng = np.random.default_rng(seed)
        path_length = 300  # Generate longer paths, extract window_size windows

        X_list = []
        y_list = []

        for _ in range(n_samples):
            # Sample θ ~ LogUniform, μ = 0 (normalized space), σ ~ Uniform
            log_theta_min = np.log(theta_range[0])
            log_theta_max = np.log(theta_range[1])
            theta = np.exp(rng.uniform(log_theta_min, log_theta_max))
            mu = 0.0        # In normalized space the mean is always 0
            sigma = rng.uniform(sigma_range[0], sigma_range[1])

            # Generate OU path
            path = generate_ou_process(
                mu=mu, theta=theta, sigma=sigma,
                n_steps=path_length, dt=1.0, initial_value=0.0,
                seed=None
            ).values

            # Random window extraction
            start = rng.integers(0, path_length - self.window_size)
            window = path[start:start + self.window_size]

            # Normalize input
            w_mean = np.mean(window)
            w_std = np.std(window)
            if w_std < 1e-8:
                w_std = 1e-8

            window_norm = (window - w_mean) / w_std

            # Targets (in normalized space)
            mu_norm = (mu - w_mean) / w_std
            sigma_norm = sigma / w_std

            log_theta = np.log(theta)
            log_sigma_norm = np.log(max(sigma_norm, 1e-6))

            X_list.append(window_norm.reshape(-1, 1).astype(np.float32))
            y_list.append([log_theta, mu_norm, log_sigma_norm])

        X = np.array(X_list, dtype=np.float32)   # (n_samples, window_size, 1)
        y = np.array(y_list, dtype=np.float32)   # (n_samples, 3)
        return X, y

    def train(self, n_samples: int = 50000, epochs: int = 100, batch_size: int = 256,
              theta_range: tuple = (0.005, 0.5), sigma_range: tuple = (0.1, 2.0),
              lr: float = 1e-3, val_split: float = 0.1, seed: int = 42) -> TrainingHistory:
        """
        Train the LSTM on synthetic OU paths.

        Args:
            n_samples:    Number of synthetic paths to generate
            epochs:       Training epochs
            batch_size:   Mini-batch size
            theta_range:  (min, max) for θ ~ LogUniform
            sigma_range:  (min, max) for σ ~ Uniform
            lr:           Learning rate
            val_split:    Fraction held out for validation
            seed:         Random seed

        Returns:
            TrainingHistory with per-epoch train/val losses
        """
        print(f"Generating {n_samples} synthetic OU paths...", flush=True)
        X, y = self._generate_training_data(n_samples, theta_range, sigma_range, seed)

        # Train / val split
        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val

        X_train = torch.tensor(X[:n_train])
        y_train = torch.tensor(y[:n_train])
        X_val = torch.tensor(X[n_train:])
        y_val = torch.tensor(y[n_train:])

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )
        criterion = nn.MSELoss()

        history = TrainingHistory()

        print(f"Training LSTM: {n_train} train, {n_val} val, {epochs} epochs", flush=True)
        print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}", flush=True)
        print("-" * 36, flush=True)

        for epoch in range(epochs):
            # Training pass
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * len(X_batch)

            train_loss /= n_train

            # Validation pass
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val.to(self.device))
                val_loss = criterion(val_pred, y_val.to(self.device)).item()

            scheduler.step(val_loss)
            history.train_losses.append(train_loss)
            history.val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"{epoch + 1:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}", flush=True)

        history.epochs = epochs
        history.final_train_loss = history.train_losses[-1]
        history.final_val_loss = history.val_losses[-1]
        self.is_trained = True

        print(f"\nTraining complete. Final val loss: {history.final_val_loss:.6f}", flush=True)
        return history

    def estimate(self, spread: pd.Series, n_mc_samples: int = 200) -> OULSTMResult:
        """
        Estimate OU parameters from a spread series using MC Dropout.

        Args:
            spread:        Time series of spread values
            n_mc_samples:  Number of MC Dropout forward passes

        Returns:
            OULSTMResult with point estimates (mean of draws) and 95% CI
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model must be trained before calling estimate(). "
                "Call train() or load() first."
            )

        values = np.array(spread.values, dtype=np.float32)
        n = len(values)

        # Extract last window_size observations (pad with first value if shorter)
        if n >= self.window_size:
            window = values[-self.window_size:]
        else:
            pad_len = self.window_size - n
            window = np.concatenate([np.full(pad_len, values[0]), values])

        # Normalize input
        sample_mean = float(np.mean(window))
        sample_std = float(np.std(window))
        if sample_std < 1e-8:
            sample_std = 1e-8

        window_norm = (window - sample_mean) / sample_std
        X = torch.tensor(
            window_norm.reshape(1, self.window_size, 1), dtype=torch.float32
        ).to(self.device)

        # MC Dropout: keep model in train() to activate dropout
        self.model.train()

        theta_draws, mu_draws, sigma_draws = [], [], []

        with torch.no_grad():
            for _ in range(n_mc_samples):
                pred = self.model(X)           # (1, 3)
                log_theta, mu_norm, log_sigma_norm = pred.cpu().numpy()[0]

                theta_real = float(np.exp(log_theta))
                sigma_norm = float(np.exp(log_sigma_norm))
                mu_real = sample_mean + float(mu_norm) * sample_std
                sigma_real = sigma_norm * sample_std

                theta_draws.append(theta_real)
                mu_draws.append(mu_real)
                sigma_draws.append(sigma_real)

        theta_arr = np.array(theta_draws)
        mu_arr = np.array(mu_draws)
        sigma_arr = np.array(sigma_draws)

        return OULSTMResult(
            theta=float(np.mean(theta_arr)),
            mu=float(np.mean(mu_arr)),
            sigma=float(np.mean(sigma_arr)),
            theta_ci=(float(np.percentile(theta_arr, 2.5)), float(np.percentile(theta_arr, 97.5))),
            mu_ci=(float(np.percentile(mu_arr, 2.5)), float(np.percentile(mu_arr, 97.5))),
            sigma_ci=(float(np.percentile(sigma_arr, 2.5)), float(np.percentile(sigma_arr, 97.5))),
            theta_std=float(np.std(theta_arr)),
            mu_std=float(np.std(mu_arr)),
            sigma_std=float(np.std(sigma_arr)),
            mc_draws={'theta': theta_draws, 'mu': mu_draws, 'sigma': sigma_draws},
            n_mc_samples=n_mc_samples,
        )

    def save(self, path: str):
        """Save model weights and hyperparameters to file"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'window_size': self.window_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'is_trained': self.is_trained,
        }, path)
        print(f"✓ Model saved to {path}")

    def load(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)

        self.window_size = checkpoint['window_size']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.dropout = checkpoint['dropout']
        self.is_trained = checkpoint.get('is_trained', True)

        self.model = OULSTMNet(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"✓ Model loaded from {path}")

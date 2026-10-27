"""
lstm_model.py

LSTM regressor for h-day log returns, with the same fit/predict interface as the
scikit-learn random forest:

    model = LSTMRegressor()
    model.fit(X_train, y_train)          # X: (samples, lookback, features)
    y_hat = model.predict(X_val)         # predicted log return over h days

Design choices (each one is something to be able to defend):

1. Small network. About 1 670 training windows and a noisy target. One LSTM
   layer with 32 hidden units and 16 inputs has 6 433 weights. Bigger nets only fit noise.
2. Many-to-one. The LSTM reads the 20-day window step by step and only the last
   hidden state h_T goes to a linear output layer. The forecast is a single number.
3. Target standardised with the training mean and std. Daily-scale returns are
   around 0.01 to 0.05, which gives tiny gradients for MSE. The scaling is undone
   in predict(), so outputs are log returns again.
4. MSE loss, because the shared evaluation metric is RMSE.
5. Early stopping on the LAST part of the training period (chronological, with a
   10-day embargo), not on the validation set. Validation is kept for model
   choice and for the comparison with RF, so it is not used to fit the weights or
   to pick the stopping epoch. After early stopping the net is retrained on all
   training windows for the number of epochs that was found.
6. Seed ensemble. A small LSTM on noisy data gives clearly different results for
   different random initialisations. The prediction is the mean over n_seeds nets,
   which reduces this variance. predict_members() returns each net separately.

The windows are independent samples (the LSTM state is reset for every window,
"stateless"), so shuffling the training windows between batches is allowed.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


class _Net(nn.Module):
    """x (batch, lookback, features) -> LSTM -> last hidden state -> dropout -> linear -> y_hat."""

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)          # h_n: (num_layers, batch, hidden)
        return self.head(self.drop(h_n[-1])).squeeze(-1)


class LSTMRegressor:
    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 20,
        es_fraction: float = 0.15,
        embargo: int = 10,
        refit_full: bool = True,
        n_seeds: int = 5,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.es_fraction = es_fraction
        self.embargo = embargo
        self.refit_full = refit_full
        self.n_seeds = n_seeds
        self.seed = seed
        self.verbose = verbose

    # ------------------------------------------------------------------ helpers
    def _new_net(self, n_features: int, seed: int) -> _Net:
        torch.manual_seed(seed)
        return _Net(n_features, self.hidden_size, self.num_layers, self.dropout)

    def _train(self, net, X, y, epochs, X_es=None, y_es=None, seed=0):
        """Mini-batch Adam on MSE. With X_es given: stop when the early-stopping
        loss has not improved for `patience` epochs and keep the best weights.
        Returns (net, best_epoch)."""
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        gen = torch.Generator().manual_seed(seed)

        best_loss, best_epoch, best_state, waited = np.inf, epochs, None, 0
        for epoch in range(1, epochs + 1):
            net.train()
            order = torch.randperm(len(Xt), generator=gen)
            for i in range(0, len(Xt), self.batch_size):
                b = order[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(net(Xt[b]), yt[b])
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)   # guards against exploding gradients
                opt.step()

            if X_es is not None:
                es_loss = self._mse(net, X_es, y_es)
                if es_loss < best_loss - 1e-6:
                    best_loss, best_epoch, waited = es_loss, epoch, 0
                    best_state = copy.deepcopy(net.state_dict())
                else:
                    waited += 1
                    if waited >= self.patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best_epoch

    @staticmethod
    def _mse(net, X, y) -> float:
        net.eval()
        with torch.no_grad():
            return float(torch.mean((net(torch.from_numpy(X)) - torch.from_numpy(y)) ** 2))

    # --------------------------------------------------------------------- API
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMRegressor":
        """X: (samples, lookback, features), already scaled by lstm_data().
        y: h-day log returns. Rows must be in date order."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)

        if self.n_seeds < 1:
            raise ValueError("n_seeds must be at least 1")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        if len(y) == 0 or float(y.std()) == 0.0:
            raise ValueError("y must be non-empty and not constant")

        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std())
        ys = ((y - self.y_mean_) / self.y_std_).astype(np.float32)

        # Chronological early-stopping split inside the training period.
        n_es = int(round(self.es_fraction * len(X)))
        cut = len(X) - n_es
        if n_es < 1 or cut - self.embargo < 1:
            raise ValueError("too few samples: early-stopping part and fit part (after embargo) must both be non-empty")
        X_fit, y_fit = X[:cut - self.embargo], ys[:cut - self.embargo]
        X_es, y_es = X[cut:], ys[cut:]

        self.nets_, self.best_epochs_ = [], []
        for k in range(self.n_seeds):
            s = self.seed + k
            net, best_epoch = self._train(self._new_net(X.shape[2], s), X_fit, y_fit,
                                          self.max_epochs, X_es, y_es, seed=s)
            if self.refit_full:
                net, _ = self._train(self._new_net(X.shape[2], s), X, ys, best_epoch, seed=s)
            net.eval()
            self.nets_.append(net)
            self.best_epochs_.append(best_epoch)
            if self.verbose:
                print(f"  seed {s}: stopped at epoch {best_epoch}")
        return self

    def predict_members(self, X: np.ndarray) -> np.ndarray:
        """Predictions from each seed, shape (n_seeds, samples), in log-return units."""
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        with torch.no_grad():
            P = np.stack([net(Xt).numpy() for net in self.nets_])
        return P * self.y_std_ + self.y_mean_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_members(X).mean(axis=0)

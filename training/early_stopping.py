"""Batch-wise early stopping for PhysLLaVA training stages.

Uses an exponential moving average (EMA) of the per-step loss to smooth
out batch-level noise, then checks every ``check_every_n_steps`` steps
whether the EMA has improved by at least ``min_delta``.  If it has not
improved for ``patience`` consecutive checks, :meth:`EarlyStopper.update`
returns ``True`` to signal the training loop to stop.

Typical usage::

    stopper = EarlyStopper.from_config(stage_cfg)
    stopped = False
    for epoch in range(num_epochs):
        for batch in dataloader:
            loss = ...
            if stopper.update(loss, global_step):
                stopped = True
                break
        if stopped:
            break
"""

from __future__ import annotations


class EarlyStopper:
    """Batch-wise early stopper based on EMA loss.

    Args:
        patience: Number of consecutive checks without improvement
            before stopping.
        min_delta: Minimum absolute improvement in EMA loss to count
            as progress.
        check_every_n_steps: How often (in optimizer steps) to evaluate
            whether training has plateaued.
        ema_alpha: Smoothing factor for the exponential moving average.
            Smaller values give a slower-responding, smoother average
            (e.g. 0.05 means each new loss contributes 5% to the EMA).
        min_steps: Do not trigger stopping before this many steps have
            elapsed (lets the model get past noisy early warmup).
        enabled: If ``False``, :meth:`update` always returns ``False``.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-3,
        check_every_n_steps: int = 200,
        ema_alpha: float = 0.05,
        min_steps: int = 500,
        enabled: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.check_every_n_steps = check_every_n_steps
        self.ema_alpha = ema_alpha
        self.min_steps = min_steps
        self.enabled = enabled

        self._ema: float | None = None
        self._best_ema: float = float("inf")
        self._checks_without_improvement: int = 0

    @classmethod
    def from_config(cls, stage_cfg: dict) -> "EarlyStopper":
        """Build an :class:`EarlyStopper` from a stage config dict.

        Reads the ``early_stopping`` sub-dict if present.  All keys are
        optional and fall back to the class defaults.

        Args:
            stage_cfg: A ``stage1`` or ``stage2`` config dict.

        Returns:
            Configured :class:`EarlyStopper` instance.
        """
        es_cfg = stage_cfg.get("early_stopping", {})
        return cls(
            patience=es_cfg.get("patience", 5),
            min_delta=es_cfg.get("min_delta", 1e-3),
            check_every_n_steps=es_cfg.get("check_every_n_steps", 200),
            ema_alpha=es_cfg.get("ema_alpha", 0.05),
            min_steps=es_cfg.get("min_steps", 500),
            enabled=es_cfg.get("enabled", True),
        )

    @property
    def ema(self) -> float | None:
        """Current EMA of the loss, or ``None`` before any updates."""
        return self._ema

    def update(self, loss: float, step: int) -> bool:
        """Update the EMA and check the stopping criterion.

        Args:
            loss: Raw loss value for the current step.
            step: Current global optimizer step (0-indexed).

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        if not self.enabled:
            return False

        # Update EMA
        if self._ema is None:
            self._ema = loss
        else:
            self._ema = self.ema_alpha * loss + (1.0 - self.ema_alpha) * self._ema

        # Only check after min_steps and on the scheduled interval
        if step < self.min_steps:
            return False
        if (step + 1) % self.check_every_n_steps != 0:
            return False

        # Check for improvement
        if self._ema < self._best_ema - self.min_delta:
            self._best_ema = self._ema
            self._checks_without_improvement = 0
        else:
            self._checks_without_improvement += 1

        if self._checks_without_improvement >= self.patience:
            return True

        return False

    def status(self) -> str:
        """Return a short human-readable status string."""
        if self._ema is None:
            return "EarlyStopper: no data yet"
        return (
            f"EarlyStopper: ema={self._ema:.4f}  best={self._best_ema:.4f}  "
            f"checks_flat={self._checks_without_improvement}/{self.patience}"
        )

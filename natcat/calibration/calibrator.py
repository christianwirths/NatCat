"""Fit vulnerability parameters to observed storm losses.

The objective is the weighted mean squared decadic log ratio of modelled to
observed loss,

.. math::

    J(\\theta) = \\frac{\\sum_s w_s \\,[\\log_{10} L_s(\\theta) - \\log_{10} O_s]^2}{\\sum_s w_s},

which treats a factor-of-two overestimate and underestimate symmetrically and
keeps a $25 B and a $1 B storm on equal footing. It is minimised over the
model's :attr:`~natcat.vulnerability.VulnerabilityModel.params` with SciPy,
either locally (Nelder-Mead from the current parameters) or globally
(differential evolution within bounds, then a local polish).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..vulnerability.base import VulnerabilityModel
from .cases import StormCase

__all__ = ["CalibrationResult", "Calibrator"]

logger = logging.getLogger(__name__)

_LOSS_FLOOR = 1.0  # USD; keeps log10 finite when a parameter set produces no loss


@dataclass
class CalibrationResult:
    """Outcome of :meth:`Calibrator.fit`.

    Attributes
    ----------
    model : VulnerabilityModel
        Calibrated model.
    initial_model : VulnerabilityModel
        Model the calibration started from.
    params, initial_params : dict
        Parameter values after and before calibration.
    objective, initial_objective : float
        Objective values after and before.
    table : pandas.DataFrame
        Per-storm ``observed``, ``modelled_initial``, ``modelled`` (USD) and
        the corresponding ``log_ratio_initial`` / ``log_ratio``.
    n_evaluations : int
        Objective evaluations used by the optimiser.
    method : str
        Optimiser used.
    message : str
        Optimiser status message.
    """

    model: VulnerabilityModel
    initial_model: VulnerabilityModel
    params: dict[str, float]
    initial_params: dict[str, float]
    objective: float
    initial_objective: float
    table: pd.DataFrame
    n_evaluations: int
    method: str
    message: str

    @property
    def rmse_log10(self) -> float:
        """Root of the calibrated objective: typical factor error is ``10**rmse``."""
        return float(np.sqrt(self.objective))

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines = [f"Calibration ({self.method}, {self.n_evaluations} evaluations): {self.message}"]
        for key in self.params:
            lines.append(f"  {key:>14}: {self.initial_params[key]:9.4g} -> {self.params[key]:9.4g}")
        lines.append(
            f"  objective     : {self.initial_objective:.4f} -> {self.objective:.4f} "
            f"(typical factor error {10 ** np.sqrt(self.initial_objective):.2f}x -> "
            f"{10**self.rmse_log10:.2f}x)"
        )
        return "\n".join(lines)


class Calibrator:
    """Minimise the log-ratio objective over a vulnerability model's parameters.

    Parameters
    ----------
    cases : sequence of StormCase
        Precomputed storm footprints with observed losses (see
        :func:`natcat.calibration.build_cases`).
    model : VulnerabilityModel
        Starting model; must implement ``params`` and ``with_params``.
    param_names : sequence of str, optional
        Subset of ``model.params`` to optimise. Defaults to all.
    bounds : dict, optional
        ``{name: (lo, hi)}``. Defaults to the model's ``DEFAULT_BOUNDS`` where
        available, otherwise ``(0.1x, 10x)`` of the starting value.

    Examples
    --------
    >>> calibrator = Calibrator(cases, ValueDependentVulnerability())  # doctest: +SKIP
    >>> result = calibrator.fit(method="global", seed=0)  # doctest: +SKIP
    >>> print(result.summary())  # doctest: +SKIP
    """

    def __init__(
        self,
        cases: Sequence[StormCase],
        model: VulnerabilityModel,
        *,
        param_names: Sequence[str] | None = None,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if len(cases) == 0:
            raise ValueError("Calibrator needs at least one case")
        if not model.params:
            raise ValueError(f"{type(model).__name__} exposes no calibratable parameters")
        self.cases = list(cases)
        self.model = model
        self.param_names = tuple(param_names or model.params)
        unknown = set(self.param_names) - set(model.params)
        if unknown:
            raise ValueError(
                f"Unknown parameter(s) {sorted(unknown)}; model has {list(model.params)}"
            )

        defaults = getattr(model, "DEFAULT_BOUNDS", {})
        self.bounds: dict[str, tuple[float, float]] = {}
        for name in self.param_names:
            if bounds and name in bounds:
                lo, hi = bounds[name]
            elif name in defaults:
                lo, hi = defaults[name]
            else:
                start = model.params[name]
                lo, hi = sorted((0.1 * start, 10.0 * start)) if start else (-1.0, 1.0)
            self.bounds[name] = (float(lo), float(hi))

        self.observed = np.array([c.observed_loss for c in self.cases], dtype=np.float64)
        self.weights = np.array([c.weight for c in self.cases], dtype=np.float64)
        if (self.observed <= 0).any():
            raise ValueError("Every case needs a positive observed loss")
        self._n_evaluations = 0

    # -- evaluation ---------------------------------------------------------------
    def _model_for(self, x: np.ndarray) -> VulnerabilityModel:
        return self.model.with_params(**dict(zip(self.param_names, map(float, x), strict=True)))

    def modelled_losses(self, model: VulnerabilityModel | None = None) -> np.ndarray:
        """Modelled loss per case (USD) for ``model`` (default: the starting model)."""
        model = self.model if model is None else model
        return np.array([case.loss(model) for case in self.cases], dtype=np.float64)

    def objective(self, x: np.ndarray) -> float:
        """Weighted mean squared log10 ratio for a parameter vector."""
        self._n_evaluations += 1
        modelled = np.maximum(self.modelled_losses(self._model_for(np.asarray(x))), _LOSS_FLOOR)
        log_ratio = np.log10(modelled / self.observed)
        return float(np.sum(self.weights * log_ratio**2) / self.weights.sum())

    def table(self, model: VulnerabilityModel) -> pd.DataFrame:
        """Per-storm comparison of the starting model and ``model``."""
        initial = self.modelled_losses()
        current = self.modelled_losses(model)
        return pd.DataFrame(
            {
                "storm_id": [c.storm_id for c in self.cases],
                "name": [c.name for c in self.cases],
                "observed": self.observed,
                "modelled_initial": initial,
                "modelled": current,
                "log_ratio_initial": np.log10(np.maximum(initial, _LOSS_FLOOR) / self.observed),
                "log_ratio": np.log10(np.maximum(current, _LOSS_FLOOR) / self.observed),
                "weight": self.weights,
            }
        )

    # -- fitting -------------------------------------------------------------------
    def fit(
        self,
        *,
        method: str = "global",
        maxiter: int = 200,
        seed: int | None = None,
        polish: bool = True,
    ) -> CalibrationResult:
        """Run the optimiser.

        Parameters
        ----------
        method : {"global", "local"}, default "global"
            ``"global"`` runs :func:`scipy.optimize.differential_evolution`
            within the bounds (then a Nelder-Mead polish if ``polish``);
            ``"local"`` runs bounded Nelder-Mead from the starting model.
        maxiter : int, default 200
            Iteration budget of the main optimiser.
        seed : int, optional
            Seed for differential evolution.
        polish : bool, default True
            Refine the global optimum locally.

        Returns
        -------
        CalibrationResult
        """
        from scipy.optimize import differential_evolution, minimize

        self._n_evaluations = 0
        x0 = np.array([self.model.params[name] for name in self.param_names], dtype=np.float64)
        bounds = [self.bounds[name] for name in self.param_names]
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        initial_objective = self.objective(x0)

        if method == "global":
            outcome = differential_evolution(
                self.objective, bounds, maxiter=maxiter, seed=seed, tol=1e-6, polish=False
            )
            x_best, message = outcome.x, str(outcome.message)
            if polish:
                local = minimize(
                    self.objective, x_best, method="Nelder-Mead", bounds=bounds,
                    options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-8},
                )  # fmt: skip
                if local.fun <= outcome.fun:
                    x_best, message = local.x, f"{message}; polished"
        elif method == "local":
            outcome = minimize(
                self.objective, x0, method="Nelder-Mead", bounds=bounds,
                options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-8},
            )  # fmt: skip
            x_best, message = outcome.x, str(outcome.message)
        else:
            raise ValueError(f"Unknown method {method!r}; use 'global' or 'local'")

        final_objective = self.objective(x_best)
        if final_objective > initial_objective:  # never return something worse than the start
            logger.warning("Optimiser did not improve on the starting model; keeping it")
            x_best, final_objective, message = x0, initial_objective, message + "; no improvement"

        model = self._model_for(x_best)
        return CalibrationResult(
            model=model,
            initial_model=self.model,
            params=model.params,
            initial_params=self.model.params,
            objective=final_objective,
            initial_objective=initial_objective,
            table=self.table(model),
            n_evaluations=self._n_evaluations,
            method=method,
            message=message,
        )

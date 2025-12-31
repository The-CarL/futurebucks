"""Sensitivity analysis module for financial simulations.

Provides tools to analyze which input variables most impact final outcomes:
- Tornado analysis: Shows impact of varying each input variable
- What-if matrices: 2D heatmaps showing impact of varying two variables
"""

from copy import deepcopy
from typing import Literal, Callable
import logging

import numpy as np
from pydantic import BaseModel, Field

from futurebucks.models import Scenario, Distribution
from futurebucks.simulation import run_simulation

logger = logging.getLogger(__name__)


class SensitivityResult(BaseModel):
    """Result of varying a single parameter."""

    variable_path: str
    variable_label: str
    base_value: float
    low_value: float
    high_value: float
    low_result: float  # Metric value at low setting
    high_result: float  # Metric value at high setting
    base_result: float  # Metric value at base setting
    impact: float  # Absolute difference (high - low)


class TornadoResult(BaseModel):
    """Results from tornado analysis."""

    metric_name: str
    base_value: float
    sensitivities: list[SensitivityResult]


class WhatIfCell(BaseModel):
    """A single cell in the what-if matrix."""

    var1_value: float
    var2_value: float
    result: float


class WhatIfMatrix(BaseModel):
    """Results from what-if matrix analysis."""

    variable1_path: str
    variable1_label: str
    variable1_values: list[float]
    variable2_path: str
    variable2_label: str
    variable2_values: list[float]
    metric_name: str
    cells: list[WhatIfCell]
    base_result: float


# Type for target metrics
TargetMetric = Literal["final_net_worth", "fire_year", "final_savings_rate"]


class SensitivityAnalyzer:
    """Analyze which input variables most impact final outcomes."""

    # Variables available for tornado analysis
    # Format: (path, label, is_distribution)
    TORNADO_VARIABLES: list[tuple[str, str, bool]] = [
        ("income_sources.*.amount", "Income Amount", False),
        ("income_sources.*.growth_rate", "Income Growth Rate", True),
        ("assets.*.balance", "Starting Balance", False),
        ("assets.*.expected_return", "Investment Return", True),
        ("expenses.*.amount", "Expense Amount", False),
        ("assumptions.inflation_rates.general", "General Inflation", True),
        ("assumptions.inflation_rates.healthcare", "Healthcare Inflation", True),
        ("assumptions.state_tax_rate", "State Tax Rate", False),
        ("simulation_years", "Simulation Years", False),
    ]

    def __init__(self, scenario: Scenario):
        """Initialize analyzer with a base scenario.

        Args:
            scenario: The base scenario to analyze
        """
        self.scenario = scenario
        self._base_result = None

    def _get_base_result(self):
        """Get or compute the base simulation result."""
        if self._base_result is None:
            self._base_result = run_simulation(self.scenario)
        return self._base_result

    def _get_metric_value(self, result, metric: TargetMetric) -> float:
        """Extract the target metric from a simulation result."""
        if metric == "final_net_worth":
            return result.snapshots[-1].net_worth
        elif metric == "fire_year":
            fire_year = result.milestones.get("fire_number")
            if fire_year is None:
                # Return a large number to indicate never achieved
                return result.snapshots[-1].year + 100
            return fire_year
        elif metric == "final_savings_rate":
            final_snapshot = result.snapshots[-1]
            if final_snapshot.gross_income > 0:
                return final_snapshot.savings / final_snapshot.gross_income
            return 0.0
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _resolve_path(self, path: str) -> list[tuple[str, float, bool, str | None]]:
        """Resolve a variable path to actual values in the scenario.

        Returns list of (resolved_path, current_value, is_distribution, item_name) tuples.
        Handles:
        - Simple paths: 'simulation_years'
        - Nested paths: 'assumptions.state_tax_rate'
        - Wildcard paths: 'income_sources.*.amount'
        - Indexed paths: 'income_sources[0].amount'
        """
        results = []

        # Parse path into parts, handling both dot notation and array indices
        # e.g., "income_sources[0].amount" -> ["income_sources", "[0]", "amount"]
        import re
        parts = re.split(r'\.(?![^\[]*\])', path)  # Split on dots not inside brackets

        # Further split array indices
        parsed_parts = []
        for part in parts:
            if '[' in part:
                # Split "income_sources[0]" into "income_sources" and "[0]"
                match = re.match(r'([^\[]+)(\[\d+\])', part)
                if match:
                    parsed_parts.append(match.group(1))
                    parsed_parts.append(match.group(2))
                else:
                    parsed_parts.append(part)
            else:
                parsed_parts.append(part)

        def _resolve_recursive(obj, parts, current_path, item_name=None):
            if not parts:
                return

            part = parts[0]
            remaining = parts[1:]

            # Handle array index like "[0]"
            if part.startswith('[') and part.endswith(']'):
                try:
                    idx = int(part[1:-1])
                    if isinstance(obj, list) and idx < len(obj):
                        item = obj[idx]
                        item_path = f"{current_path}{part}"
                        # Capture the item's name if it has one
                        name = getattr(item, "name", None)
                        if remaining:
                            _resolve_recursive(item, remaining, item_path, name)
                        else:
                            results.append((item_path, self._get_value(item), False, name))
                except (ValueError, IndexError):
                    pass
            elif part == "*":
                # Wildcard - iterate over list
                if isinstance(obj, list):
                    for i, item in enumerate(obj):
                        item_path = f"{current_path}[{i}]"
                        # Capture the item's name if it has one
                        name = getattr(item, "name", None)
                        if remaining:
                            _resolve_recursive(item, remaining, item_path, name)
                        else:
                            # Get value based on type
                            results.append((item_path, self._get_value(item), False, name))
            elif hasattr(obj, part):
                attr = getattr(obj, part)
                new_path = f"{current_path}.{part}" if current_path else part

                if not remaining:
                    # End of path - get value
                    is_dist = isinstance(attr, Distribution)
                    value = attr.mean if is_dist else attr
                    results.append((new_path, value, is_dist, item_name))
                else:
                    _resolve_recursive(attr, remaining, new_path, item_name)
            elif isinstance(obj, dict) and part in obj:
                attr = obj[part]
                new_path = f"{current_path}.{part}" if current_path else part

                if not remaining:
                    is_dist = isinstance(attr, Distribution)
                    value = attr.mean if is_dist else attr
                    results.append((new_path, value, is_dist, item_name))
                else:
                    _resolve_recursive(attr, remaining, new_path, item_name)

        _resolve_recursive(self.scenario, parsed_parts, "")
        return results

    def _get_value(self, obj) -> float:
        """Get numeric value from an object (handles Distribution)."""
        if isinstance(obj, Distribution):
            return obj.mean
        elif isinstance(obj, (int, float)):
            return float(obj)
        elif hasattr(obj, "amount"):
            return obj.amount
        elif hasattr(obj, "balance"):
            return obj.balance
        elif hasattr(obj, "mean"):
            return obj.mean
        return 0.0

    def _set_value(self, scenario: Scenario, path: str, value: float, is_distribution: bool):
        """Set a value in a scenario at the given path.

        Handles paths like:
        - 'simulation_years'
        - 'assumptions.state_tax_rate'
        - 'income_sources[0].amount'
        """
        import re

        # Parse path into parts, handling both dot notation and array indices
        parts = re.split(r'\.(?![^\[]*\])', path)

        # Further split array indices
        parsed_parts = []
        for part in parts:
            if '[' in part:
                match = re.match(r'([^\[]+)(\[\d+\])', part)
                if match:
                    parsed_parts.append(match.group(1))
                    parsed_parts.append(match.group(2))
                else:
                    parsed_parts.append(part)
            else:
                parsed_parts.append(part)

        obj = scenario

        # Navigate to parent (all parts except the last)
        for part in parsed_parts[:-1]:
            if part.startswith('[') and part.endswith(']'):
                # Array index
                idx = int(part[1:-1])
                obj = obj[idx]
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict):
                obj = obj[part]
            elif isinstance(obj, list):
                # This shouldn't happen if path is well-formed
                pass

        # Set final value
        final_part = parsed_parts[-1]
        if final_part.startswith('[') and final_part.endswith(']'):
            # Setting a list item directly (rare)
            idx = int(final_part[1:-1])
            obj[idx] = value
        else:
            current = getattr(obj, final_part, None)
            if is_distribution or isinstance(current, Distribution):
                # Set as Distribution
                setattr(obj, final_part, Distribution(mean=value, stddev=0.0))
            else:
                setattr(obj, final_part, value)

    def _create_modified_scenario(
        self, path: str, value: float, is_distribution: bool
    ) -> Scenario:
        """Create a copy of the scenario with one value modified."""
        modified = deepcopy(self.scenario)
        self._set_value(modified, path, value, is_distribution)
        return modified

    def run_tornado_analysis(
        self,
        target_metric: TargetMetric = "final_net_worth",
        variation_pct: float = 0.20,
        top_n: int = 10,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> TornadoResult:
        """Run tornado analysis to identify most impactful variables.

        Args:
            target_metric: The metric to analyze ("final_net_worth", "fire_year", "final_savings_rate")
            variation_pct: Percentage to vary each variable (e.g., 0.20 = ±20%)
            top_n: Number of top variables to include in results
            progress_callback: Optional callback(completed, total) for progress updates

        Returns:
            TornadoResult with sensitivity analysis for each variable
        """
        base_result = self._get_base_result()
        base_metric = self._get_metric_value(base_result, target_metric)

        sensitivities = []
        total_vars = 0

        # Count total variables for progress
        for path, label, is_dist in self.TORNADO_VARIABLES:
            resolved = self._resolve_path(path)
            total_vars += len(resolved)

        completed = 0

        for path, label, is_dist in self.TORNADO_VARIABLES:
            resolved = self._resolve_path(path)

            for resolved_path, base_value, is_distribution, item_name in resolved:
                if base_value == 0:
                    # Skip zero values to avoid division issues
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total_vars)
                    continue

                # Calculate low and high values
                low_value = base_value * (1 - variation_pct)
                high_value = base_value * (1 + variation_pct)

                # Run simulations with modified values
                try:
                    low_scenario = self._create_modified_scenario(
                        resolved_path, low_value, is_distribution or is_dist
                    )
                    low_result = run_simulation(low_scenario)
                    low_metric = self._get_metric_value(low_result, target_metric)

                    high_scenario = self._create_modified_scenario(
                        resolved_path, high_value, is_distribution or is_dist
                    )
                    high_result = run_simulation(high_scenario)
                    high_metric = self._get_metric_value(high_result, target_metric)

                    # Calculate impact
                    impact = abs(high_metric - low_metric)

                    # Create descriptive label using actual item name
                    var_label = label
                    if item_name:
                        var_label = f"{label} ({item_name})"

                    sensitivities.append(
                        SensitivityResult(
                            variable_path=resolved_path,
                            variable_label=var_label,
                            base_value=base_value,
                            low_value=low_value,
                            high_value=high_value,
                            low_result=low_metric,
                            high_result=high_metric,
                            base_result=base_metric,
                            impact=impact,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error analyzing {resolved_path}: {e}")

                completed += 1
                if progress_callback:
                    progress_callback(completed, total_vars)

        # Sort by impact and take top N
        sensitivities.sort(key=lambda x: x.impact, reverse=True)
        sensitivities = sensitivities[:top_n]

        return TornadoResult(
            metric_name=target_metric,
            base_value=base_metric,
            sensitivities=sensitivities,
        )

    def run_what_if_matrix(
        self,
        variable1_path: str,
        variable1_range: tuple[float, float, int],
        variable2_path: str,
        variable2_range: tuple[float, float, int],
        target_metric: TargetMetric = "final_net_worth",
        progress_callback: Callable[[int, int], None] | None = None,
        variable1_label: str | None = None,
        variable2_label: str | None = None,
    ) -> WhatIfMatrix:
        """Run what-if analysis varying two variables.

        Args:
            variable1_path: Path to first variable (e.g., "income_sources[0].amount")
            variable1_range: (min, max, num_steps) for first variable
            variable2_path: Path to second variable
            variable2_range: (min, max, num_steps) for second variable
            target_metric: The metric to analyze
            progress_callback: Optional callback(completed, total) for progress updates

        Returns:
            WhatIfMatrix with results for all combinations
        """
        base_result = self._get_base_result()
        base_metric = self._get_metric_value(base_result, target_metric)

        # Generate value ranges
        var1_values = list(
            np.linspace(variable1_range[0], variable1_range[1], variable1_range[2])
        )
        var2_values = list(
            np.linspace(variable2_range[0], variable2_range[1], variable2_range[2])
        )

        # Resolve paths to get is_distribution flag
        resolved1 = self._resolve_path(variable1_path)
        resolved2 = self._resolve_path(variable2_path)

        # Tuple format: (resolved_path, current_value, is_distribution, item_name)
        is_dist1 = resolved1[0][2] if resolved1 else False
        is_dist2 = resolved2[0][2] if resolved2 else False

        cells = []
        total = len(var1_values) * len(var2_values)
        completed = 0

        for v1 in var1_values:
            for v2 in var2_values:
                try:
                    # Create scenario with both modifications
                    modified = deepcopy(self.scenario)
                    self._set_value(modified, variable1_path, v1, is_dist1)
                    self._set_value(modified, variable2_path, v2, is_dist2)

                    result = run_simulation(modified)
                    metric = self._get_metric_value(result, target_metric)

                    cells.append(
                        WhatIfCell(
                            var1_value=v1,
                            var2_value=v2,
                            result=metric,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error in what-if analysis ({v1}, {v2}): {e}")
                    cells.append(
                        WhatIfCell(
                            var1_value=v1,
                            var2_value=v2,
                            result=base_metric,  # Fallback to base
                        )
                    )

                completed += 1
                if progress_callback:
                    progress_callback(completed, total)

        # Use provided labels or generate from paths
        label1 = variable1_label if variable1_label else self._path_to_label(variable1_path)
        label2 = variable2_label if variable2_label else self._path_to_label(variable2_path)

        return WhatIfMatrix(
            variable1_path=variable1_path,
            variable1_label=label1,
            variable1_values=var1_values,
            variable2_path=variable2_path,
            variable2_label=label2,
            variable2_values=var2_values,
            metric_name=target_metric,
            cells=cells,
            base_result=base_metric,
        )

    def _path_to_label(self, path: str) -> str:
        """Convert a path to a human-readable label."""
        # Map known paths to labels
        path_labels = {
            "income_sources": "Income",
            "assets": "Assets",
            "expenses": "Expenses",
            "assumptions": "Assumptions",
            "inflation_rates": "Inflation",
            "expected_return": "Return Rate",
            "growth_rate": "Growth Rate",
            "amount": "Amount",
            "balance": "Balance",
            "state_tax_rate": "State Tax Rate",
            "simulation_years": "Years",
            "general": "General",
            "healthcare": "Healthcare",
        }

        parts = path.replace("[", ".").replace("]", "").split(".")
        labels = []

        for part in parts:
            if part.isdigit():
                continue  # Skip numeric indices
            if part in path_labels:
                labels.append(path_labels[part])
            else:
                # Convert snake_case to Title Case
                labels.append(part.replace("_", " ").title())

        return " ".join(labels[-2:]) if len(labels) >= 2 else labels[-1] if labels else path


def run_sensitivity_analysis(
    scenario: Scenario,
    target_metric: TargetMetric = "final_net_worth",
    variation_pct: float = 0.20,
    top_n: int = 10,
    progress_callback: Callable[[int, int], None] | None = None,
) -> TornadoResult:
    """Convenience function to run tornado analysis on a scenario.

    Args:
        scenario: The scenario to analyze
        target_metric: The metric to optimize
        variation_pct: Percentage to vary each variable
        top_n: Number of top variables to return
        progress_callback: Optional progress callback

    Returns:
        TornadoResult with top sensitivity factors
    """
    analyzer = SensitivityAnalyzer(scenario)
    return analyzer.run_tornado_analysis(
        target_metric=target_metric,
        variation_pct=variation_pct,
        top_n=top_n,
        progress_callback=progress_callback,
    )

"""Monte Carlo simulation engine for financial projections."""

from datetime import datetime
from copy import deepcopy
from typing import Callable

import numpy as np

from futurebucks.models import (
    Scenario,
    MonteCarloConfig,
    MonteCarloResult,
    MonteCarloSnapshot,
    MilestoneProbability,
    SimulationResult,
    YearlySnapshot,
    IncomeSource,
    Expense,
    Distribution,
    DEFAULT_VOLATILITIES,
)
from futurebucks.simulation import (
    _is_active,
    _calculate_gross_income,
    _calculate_expenses,
    _calculate_healthcare_expenses,
    _apply_life_events,
    _allocate_savings,
    _apply_returns,
    _calculate_pre_tax_deductions,
    _calculate_roth_ira_limit,
)
from futurebucks.tax_engine import (
    calculate_taxes,
    inflate_value,
    inflate_brackets,
    FEDERAL_BRACKETS_2024_MFJ,
    BASE_YEAR,
    BASE_STANDARD_DEDUCTION_MFJ,
    BASE_SS_WAGE_CAP,
    BASE_401K_LIMIT,
    BASE_401K_CATCHUP,
    BASE_IRA_LIMIT,
    BASE_IRA_CATCHUP,
    BASE_HSA_LIMIT_FAMILY,
    BASE_ROTH_IRA_INCOME_LIMIT_START,
    BASE_ROTH_IRA_INCOME_LIMIT_END,
)


class SampledScenario:
    """A scenario with pre-sampled stochastic values for Monte Carlo simulation."""

    def __init__(self, scenario: Scenario, rng: np.random.Generator):
        self.scenario = scenario
        self.rng = rng
        self.num_years = scenario.simulation_years + 1

        # Check for conservative defaults mode
        self.use_conservative = scenario.assumptions.use_conservative_defaults

        # Pre-sample all stochastic values
        self._sample_all_values()

    def _apply_conservative_adjustment(self, dist: Distribution, is_return: bool = False) -> Distribution:
        """Apply conservative adjustments to a distribution.

        Conservative mode:
        - Reduces return means by 1% (e.g., 7% -> 6%)
        - Increases stddevs by 20% (e.g., 0.15 -> 0.18)
        """
        if not self.use_conservative:
            return dist

        new_mean = dist.mean - 0.01 if is_return else dist.mean
        new_stddev = dist.stddev * 1.20 if dist.stddev > 0 else dist.stddev

        return Distribution(
            type=dist.type,
            mean=new_mean,
            stddev=new_stddev,
            min=dist.min,
            max=dist.max,
        )

    def _sample_all_values(self):
        """Pre-sample all stochastic values for each year."""
        assumptions = self.scenario.assumptions

        # Sample category-specific inflation rates
        self.inflation_samples = {}
        for category in ["general", "healthcare", "education", "housing"]:
            inflation_dist = assumptions.get_inflation_rate(category)

            # Apply default stddev from DEFAULT_VOLATILITIES if not specified
            if inflation_dist.stddev == 0:
                default_key = f"inflation_{category}"
                default_stddev = DEFAULT_VOLATILITIES.get(default_key, 0.01)
                inflation_dist = Distribution(
                    type=inflation_dist.type,
                    mean=inflation_dist.mean,
                    stddev=default_stddev,
                )

            # Apply conservative adjustments
            inflation_dist = self._apply_conservative_adjustment(inflation_dist)
            self.inflation_samples[category] = self._sample_distribution(
                inflation_dist, self.num_years
            )

        # Sample asset returns per year per asset
        self.asset_return_samples = {}
        for asset in self.scenario.assets:
            asset_dist = asset.expected_return

            # Apply default stddev from DEFAULT_VOLATILITIES based on asset type
            if asset_dist.stddev == 0:
                default_stddev = DEFAULT_VOLATILITIES.get(asset.type, 0.10)
                asset_dist = Distribution(
                    type=asset_dist.type,
                    mean=asset_dist.mean,
                    stddev=default_stddev,
                )

            # Apply conservative adjustments (returns get mean reduction)
            asset_dist = self._apply_conservative_adjustment(asset_dist, is_return=True)
            self.asset_return_samples[asset.name] = self._sample_distribution(
                asset_dist, self.num_years
            )

        # Sample income growth per year per source
        self.income_growth_samples = {}
        for source in self.scenario.income_sources:
            growth_dist = source.growth_rate

            # Apply default stddev from DEFAULT_VOLATILITIES
            if growth_dist.stddev == 0:
                default_stddev = DEFAULT_VOLATILITIES.get("income_growth", 0.03)
                growth_dist = Distribution(
                    type=growth_dist.type,
                    mean=growth_dist.mean,
                    stddev=default_stddev,
                )

            # Apply conservative adjustments
            growth_dist = self._apply_conservative_adjustment(growth_dist)
            self.income_growth_samples[source.name] = self._sample_distribution(
                growth_dist, self.num_years
            )

    def _sample_distribution(self, dist: Distribution, n: int) -> np.ndarray:
        """Sample n values from a distribution."""
        # Use the Distribution's sample method to respect distribution type
        return np.array([dist.sample(self.rng) for _ in range(n)])

    def get_inflation_rate(self, year_index: int, category: str = "general") -> float:
        """Get sampled inflation rate for a year and category."""
        if category in self.inflation_samples:
            return float(self.inflation_samples[category][year_index])
        # Fallback to general inflation
        return float(self.inflation_samples["general"][year_index])

    def get_asset_return(self, asset_name: str, year_index: int) -> float:
        """Get sampled return for an asset in a year."""
        if asset_name in self.asset_return_samples:
            return float(self.asset_return_samples[asset_name][year_index])
        return self.scenario.assumptions.market_return_mean

    def get_income_growth(self, source_name: str, year_index: int) -> float:
        """Get sampled growth rate for an income source in a year."""
        if source_name in self.income_growth_samples:
            return float(self.income_growth_samples[source_name][year_index])
        return 0.03  # Default growth rate


def run_simulation_with_samples(
    scenario: Scenario,
    sampled: SampledScenario,
) -> SimulationResult:
    """Run a single simulation using pre-sampled stochastic values.

    This is a modified version of run_simulation that uses sampled values
    instead of fixed means.
    """
    current_year = datetime.now().year
    end_year = current_year + scenario.simulation_years

    # Initialize mutable state
    income_sources = deepcopy(scenario.income_sources)
    expenses = deepcopy(scenario.expenses)
    assets = {a.name: a.balance for a in scenario.assets}

    snapshots: list[YearlySnapshot] = []
    milestones: dict[str, int | None] = {
        "net_worth_100k": None,
        "net_worth_500k": None,
        "net_worth_1m": None,
        "net_worth_2m": None,
        "net_worth_3m": None,
        "net_worth_5m": None,
        "net_worth_10m": None,
        "net_worth_20m": None,
        "net_worth_50m": None,
        "net_worth_100m": None,
        "retirement_ready": None,
        "fire_number": None,
    }

    cumulative_taxes = 0.0

    for year_idx, year in enumerate(range(current_year, end_year + 1)):
        age = year - scenario.person.birth_year
        years_from_base = max(0, year - BASE_YEAR)

        # Get sampled inflation rates for this year (category-specific)
        general_inflation = sampled.get_inflation_rate(year_idx, "general")
        healthcare_inflation = sampled.get_inflation_rate(year_idx, "healthcare")

        # Build inflation overrides for expense calculation
        inflation_rate_overrides = {
            "general": general_inflation,
            "healthcare": healthcare_inflation,
            "education": sampled.get_inflation_rate(year_idx, "education"),
            "housing": sampled.get_inflation_rate(year_idx, "housing"),
        }

        # Calculate inflation-adjusted values for this year (using general inflation)
        adjusted_standard_deduction = inflate_value(
            BASE_STANDARD_DEDUCTION_MFJ, general_inflation, years_from_base
        )
        adjusted_ss_wage_cap = inflate_value(
            BASE_SS_WAGE_CAP, general_inflation, years_from_base
        )
        adjusted_401k_limit = inflate_value(
            BASE_401K_LIMIT, general_inflation, years_from_base
        )
        adjusted_401k_catchup = inflate_value(
            BASE_401K_CATCHUP, general_inflation, years_from_base
        )
        adjusted_ira_limit = inflate_value(
            BASE_IRA_LIMIT, general_inflation, years_from_base
        )
        adjusted_ira_catchup = inflate_value(
            BASE_IRA_CATCHUP, general_inflation, years_from_base
        )
        adjusted_hsa_limit = inflate_value(
            BASE_HSA_LIMIT_FAMILY, general_inflation, years_from_base
        )
        adjusted_roth_phase_out_start = inflate_value(
            BASE_ROTH_IRA_INCOME_LIMIT_START, general_inflation, years_from_base
        )
        adjusted_roth_phase_out_end = inflate_value(
            BASE_ROTH_IRA_INCOME_LIMIT_END, general_inflation, years_from_base
        )
        adjusted_brackets = inflate_brackets(
            FEDERAL_BRACKETS_2024_MFJ, general_inflation, years_from_base
        )

        # Apply life events for this year
        income_sources, expenses, assets, windfall = _apply_life_events(
            year=year,
            income_sources=income_sources,
            expenses=expenses,
            assets=assets,
            events=scenario.life_events,
            inflation_rate=general_inflation,
            start_year=current_year,
        )

        # Build growth rate overrides from sampled values
        growth_rate_overrides = {
            source.name: sampled.get_income_growth(source.name, year_idx)
            for source in income_sources
        }

        # Calculate income with sampled growth rates
        gross_income = _calculate_gross_income(
            income_sources, year, current_year, growth_rate_overrides
        )
        gross_income += windfall

        # Calculate regular expenses with category-specific inflation
        total_expenses = _calculate_expenses(
            expenses, year, current_year, general_inflation, inflation_rate_overrides
        )

        # Add healthcare expenses with healthcare-specific inflation
        healthcare_expenses = _calculate_healthcare_expenses(
            scenario, year, current_year, healthcare_inflation
        )
        total_expenses += healthcare_expenses

        # Estimate savings for pre-tax deduction calculation
        estimated_savings = gross_income * 0.7 - total_expenses
        pre_tax_deductions = _calculate_pre_tax_deductions(
            max(0, estimated_savings),
            scenario.assets,
            age,
            adjusted_401k_limit,
            adjusted_401k_catchup,
            adjusted_hsa_limit,
        )

        # Calculate taxes
        tax_result = calculate_taxes(
            gross_income=gross_income,
            pre_tax_deductions=pre_tax_deductions,
            standard_deduction=adjusted_standard_deduction,
            state_rate=scenario.assumptions.state_tax_rate,
            federal_brackets=adjusted_brackets,
            ss_wage_cap=adjusted_ss_wage_cap,
        )

        # Calculate actual savings
        net_income = gross_income - tax_result.total_tax
        savings = net_income - total_expenses
        cumulative_taxes += tax_result.total_tax

        # Allocate savings to assets
        if savings > 0:
            assets = _allocate_savings(
                savings=savings,
                assets=assets,
                asset_configs=scenario.assets,
                age=age,
                gross_income=gross_income,
                limit_401k=adjusted_401k_limit,
                limit_401k_catchup=adjusted_401k_catchup,
                limit_hsa=adjusted_hsa_limit,
                limit_ira=adjusted_ira_limit,
                limit_ira_catchup=adjusted_ira_catchup,
                roth_phase_out_start=adjusted_roth_phase_out_start,
                roth_phase_out_end=adjusted_roth_phase_out_end,
            )
        elif savings < 0:
            # Draw from taxable account if spending exceeds income
            for name, config in [(a.name, a) for a in scenario.assets]:
                if config.type == "taxable" and name in assets:
                    draw = min(-savings, assets[name])
                    assets[name] -= draw
                    savings += draw
                    if savings >= 0:
                        break

        # Build return overrides from sampled values
        return_overrides = {
            asset.name: sampled.get_asset_return(asset.name, year_idx)
            for asset in scenario.assets
        }

        # Apply investment returns with sampled values
        assets = _apply_returns(
            assets=assets,
            asset_configs=scenario.assets,
            default_return=scenario.assumptions.market_return_mean,
            return_overrides=return_overrides,
        )

        # Calculate net worth
        net_worth = sum(assets.values())

        # Create snapshot
        snapshot = YearlySnapshot(
            year=year,
            age=age,
            gross_income=gross_income,
            federal_tax=tax_result.federal_tax,
            state_tax=tax_result.state_tax,
            fica_tax=tax_result.fica_tax,
            total_tax=tax_result.total_tax,
            net_income=net_income,
            total_expenses=total_expenses,
            savings=savings,
            assets=dict(assets),
            net_worth=net_worth,
            cumulative_taxes_paid=cumulative_taxes,
        )
        snapshots.append(snapshot)

        # Check net worth milestones
        milestone_thresholds = [
            ("net_worth_100k", 100_000),
            ("net_worth_500k", 500_000),
            ("net_worth_1m", 1_000_000),
            ("net_worth_2m", 2_000_000),
            ("net_worth_3m", 3_000_000),
            ("net_worth_5m", 5_000_000),
            ("net_worth_10m", 10_000_000),
            ("net_worth_20m", 20_000_000),
            ("net_worth_50m", 50_000_000),
            ("net_worth_100m", 100_000_000),
        ]
        for key, threshold in milestone_thresholds:
            if milestones[key] is None and net_worth >= threshold:
                milestones[key] = year

    # Calculate FIRE milestone
    fire_basis = scenario.assumptions.fire_expense_basis
    if fire_basis == "max":
        fire_target = max(s.total_expenses for s in snapshots)
    elif fire_basis == "first_year":
        fire_target = snapshots[0].total_expenses
    elif fire_basis == "target":
        fire_target = scenario.assumptions.fire_target_expenses
        if fire_target is None:
            fire_target = max(s.total_expenses for s in snapshots)
    else:
        fire_target = None

    # Check FIRE and retirement milestones
    for snapshot in snapshots:
        year = snapshot.year
        net_worth = snapshot.net_worth
        year_fire_target = snapshot.total_expenses if fire_basis == "current" else fire_target

        if milestones["retirement_ready"] is None and net_worth >= year_fire_target * 25:
            milestones["retirement_ready"] = year
        if milestones["fire_number"] is None and net_worth * 0.04 >= year_fire_target:
            milestones["fire_number"] = year

    return SimulationResult(
        scenario_name=scenario.name,
        snapshots=snapshots,
        milestones=milestones,
        fire_target_expenses=fire_target if fire_target else max(s.total_expenses for s in snapshots),
    )


def run_monte_carlo(
    scenario: Scenario,
    config: MonteCarloConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation.

    Args:
        scenario: The scenario to simulate
        config: Monte Carlo configuration (uses scenario.monte_carlo if None)
        progress_callback: Optional callback(completed, total) for progress updates

    Returns:
        MonteCarloResult with aggregated statistics
    """
    if config is None:
        config = scenario.monte_carlo

    # Use effective_num_simulations which handles detailed_analysis_mode
    num_sims = config.effective_num_simulations

    # Generate seeds for reproducibility
    master_rng = np.random.default_rng(config.seed)
    seeds = master_rng.integers(0, 2**31, size=num_sims)

    results: list[SimulationResult] = []

    # Run simulations
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        sampled = SampledScenario(scenario, rng)
        result = run_simulation_with_samples(scenario, sampled)
        results.append(result)

        if progress_callback:
            progress_callback(i + 1, num_sims)

    return _aggregate_results(scenario, results, config.percentiles)


def _aggregate_results(
    scenario: Scenario,
    results: list[SimulationResult],
    percentiles: list[int],
) -> MonteCarloResult:
    """Aggregate individual simulation results into Monte Carlo statistics."""
    num_years = len(results[0].snapshots)
    num_sims = len(results)

    snapshots = []
    for year_idx in range(num_years):
        year_data = [r.snapshots[year_idx] for r in results]

        net_worths = np.array([s.net_worth for s in year_data])
        incomes = np.array([s.gross_income for s in year_data])
        expenses = np.array([s.total_expenses for s in year_data])
        savings = np.array([s.savings for s in year_data])

        snapshot = MonteCarloSnapshot(
            year=year_data[0].year,
            age=year_data[0].age,
            net_worth_p10=float(np.percentile(net_worths, 10)),
            net_worth_p25=float(np.percentile(net_worths, 25)),
            net_worth_p50=float(np.percentile(net_worths, 50)),
            net_worth_p75=float(np.percentile(net_worths, 75)),
            net_worth_p90=float(np.percentile(net_worths, 90)),
            net_worth_mean=float(np.mean(net_worths)),
            net_worth_stddev=float(np.std(net_worths)),
            gross_income_median=float(np.median(incomes)),
            total_expenses_median=float(np.median(expenses)),
            savings_median=float(np.median(savings)),
        )
        snapshots.append(snapshot)

    # Calculate milestone probabilities with retirement context
    retirement_year = scenario.person.birth_year + scenario.person.retirement_age
    milestone_probs = _calculate_milestone_probabilities(
        results,
        target_year=retirement_year,
        target_age=scenario.person.retirement_age,
    )

    # Final net worth distribution
    final_net_worths = [r.snapshots[-1].net_worth for r in results]

    return MonteCarloResult(
        scenario_name=scenario.name,
        num_simulations=num_sims,
        snapshots=snapshots,
        milestone_probabilities=milestone_probs,
        final_net_worth_distribution=final_net_worths,
    )


def _calculate_milestone_probabilities(
    results: list[SimulationResult],
    target_year: int | None = None,
    target_age: int | None = None,
) -> list[MilestoneProbability]:
    """Calculate probability of reaching various milestones.

    Args:
        results: List of simulation results
        target_year: Target year for "probability by" calculations (e.g., retirement year)
        target_age: Target age for display purposes
    """
    # Determine which milestones to show based on final net worth range
    final_net_worths = [r.snapshots[-1].net_worth for r in results]
    median_final = float(np.median(final_net_worths))

    # Always include these core milestones
    milestones = [
        ("net_worth_1m", 1_000_000),
        ("net_worth_2m", 2_000_000),
        ("net_worth_5m", 5_000_000),
        ("fire_number", None),
    ]

    # Add higher milestones if relevant
    if median_final > 5_000_000:
        milestones.insert(3, ("net_worth_10m", 10_000_000))
    if median_final > 10_000_000:
        milestones.insert(4, ("net_worth_20m", 20_000_000))
    if median_final > 20_000_000:
        milestones.insert(5, ("net_worth_50m", 50_000_000))

    probs = []
    for name, target in milestones:
        years_achieved = []

        for result in results:
            year = result.milestones.get(name)
            if year is not None:
                years_achieved.append(year)

        # Overall probability of ever reaching
        probability = len(years_achieved) / len(results)

        # Probability of reaching BY the target year (e.g., retirement age)
        probability_by_target = None
        if target_year is not None:
            achieved_by_target = sum(1 for y in years_achieved if y <= target_year)
            probability_by_target = achieved_by_target / len(results)

        if years_achieved:
            years_achieved_sorted = sorted(years_achieved)
            median_year = int(np.median(years_achieved_sorted))
            p10_year = int(np.percentile(years_achieved_sorted, 10))
            p90_year = int(np.percentile(years_achieved_sorted, 90))
        else:
            median_year = p10_year = p90_year = None

        probs.append(MilestoneProbability(
            milestone=name,
            target_value=target,
            probability=probability,
            probability_by_target=probability_by_target,
            target_year=target_year,
            target_age=target_age,
            median_year=median_year,
            p10_year=p10_year,
            p90_year=p90_year,
        ))

    return probs

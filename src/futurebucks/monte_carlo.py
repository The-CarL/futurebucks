"""Monte Carlo simulation engine for financial projections."""

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
    _calculate_healthcare_expenses,
    _apply_life_events,
    _allocate_savings,
    _apply_returns,
    _rebalance_cash_to_investments,
    _calculate_pre_tax_deductions,
    _calculate_accessible_net_worth,
    _calculate_bridge_needed,
)
from futurebucks.tax_engine import (
    calculate_taxes,
    inflate_brackets,
    FEDERAL_BRACKETS_2024_MFJ,
    BASE_YEAR,
    BASE_STANDARD_DEDUCTION_MFJ,
    BASE_STATE_STANDARD_DEDUCTION_NC,
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

    Fix #1: Uses cumulative compounding for inflation/growth instead of
    applying a single year's rate across all years.

    Also applies Fix #3 (FICA), Fix #5 (MAGI), Fix #6 (net_income),
    Fix #11 (bridge inflation), Fix #12 (state deduction).
    """
    start_year = scenario.start_year
    end_year = start_year + scenario.simulation_years
    retirement_year = scenario.person.birth_year + scenario.person.retirement_age

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

    # Fix #1: Track cumulative inflation factors for proper year-over-year compounding.
    # From BASE_YEAR to start_year, use deterministic mean inflation.
    base_to_start_years = max(0, start_year - BASE_YEAR)
    base_inflation_mean = float(scenario.assumptions.get_inflation_rate("general").mean)
    cumulative_general_factor = (1 + base_inflation_mean) ** base_to_start_years

    # Track running income amounts per source (apply growth incrementally)
    income_amounts: dict[str, float] = {
        source.name: source.amount for source in income_sources
    }

    # Track running expense amounts per expense (apply inflation incrementally)
    expense_amounts: dict[str, float] = {
        expense.name: expense.amount for expense in expenses
    }

    # Track cumulative healthcare inflation factor from simulation start
    cumulative_healthcare_factor = 1.0

    for year_idx, year in enumerate(range(start_year, end_year + 1)):
        age = year - scenario.person.birth_year
        yearly_cap_gains_tax = 0.0
        yearly_assets_sold = 0.0

        # Get sampled inflation rates for this year
        general_inflation_this_year = sampled.get_inflation_rate(year_idx, "general")
        healthcare_inflation_this_year = sampled.get_inflation_rate(year_idx, "healthcare")

        # Fix #1: Update cumulative factors year-over-year
        if year_idx > 0:
            cumulative_general_factor *= (1 + general_inflation_this_year)
            cumulative_healthcare_factor *= (1 + healthcare_inflation_this_year)

            # Apply this year's inflation to running expense amounts
            for expense in expenses:
                if expense.inflation_adjusted and expense.name in expense_amounts:
                    category = getattr(expense, "inflation_category", "general")
                    category_rate = sampled.get_inflation_rate(year_idx, category)
                    expense_amounts[expense.name] *= (1 + category_rate)

            # Apply this year's growth to running income amounts
            for source in income_sources:
                if source.name in income_amounts:
                    growth_rate = sampled.get_income_growth(source.name, year_idx)
                    income_amounts[source.name] *= (1 + growth_rate)

        # Use cumulative factor for tax bracket/limit inflation from BASE_YEAR
        adjusted_standard_deduction = BASE_STANDARD_DEDUCTION_MFJ * cumulative_general_factor
        adjusted_state_standard_deduction = BASE_STATE_STANDARD_DEDUCTION_NC * cumulative_general_factor
        adjusted_ss_wage_cap = BASE_SS_WAGE_CAP * cumulative_general_factor
        adjusted_401k_limit = BASE_401K_LIMIT * cumulative_general_factor
        adjusted_401k_catchup = BASE_401K_CATCHUP * cumulative_general_factor
        adjusted_ira_limit = BASE_IRA_LIMIT * cumulative_general_factor
        adjusted_ira_catchup = BASE_IRA_CATCHUP * cumulative_general_factor
        adjusted_hsa_limit = BASE_HSA_LIMIT_FAMILY * cumulative_general_factor
        adjusted_roth_phase_out_start = BASE_ROTH_IRA_INCOME_LIMIT_START * cumulative_general_factor
        adjusted_roth_phase_out_end = BASE_ROTH_IRA_INCOME_LIMIT_END * cumulative_general_factor
        adjusted_brackets = [
            (limit * cumulative_general_factor if limit != float("inf") else float("inf"), rate)
            for limit, rate in FEDERAL_BRACKETS_2024_MFJ
        ]

        # Apply life events for this year
        income_sources, expenses, assets, windfall = _apply_life_events(
            year=year,
            income_sources=income_sources,
            expenses=expenses,
            assets=assets,
            events=scenario.life_events,
            inflation_rate=general_inflation_this_year,
            start_year=start_year,
        )

        # Update tracking dicts for any new sources/expenses from life events
        for source in income_sources:
            if source.name not in income_amounts:
                income_amounts[source.name] = source.amount
        for expense in expenses:
            if expense.name not in expense_amounts:
                expense_amounts[expense.name] = expense.amount

        # Fix #3: Calculate employment income separately from windfall
        employment_income = 0.0
        for source in income_sources:
            if not _is_active(source, year):
                continue
            if (
                source.end_year is None
                and retirement_year is not None
                and year > retirement_year
            ):
                continue
            employment_income += income_amounts.get(source.name, source.amount)

        gross_income = employment_income + windfall

        # Calculate expenses using running amounts (Fix #1: proper cumulative inflation)
        total_expenses = 0.0
        for expense in expenses:
            if not _is_active(expense, year):
                continue
            total_expenses += expense_amounts.get(expense.name, expense.amount)

        # Add healthcare expenses with cumulative healthcare inflation
        if scenario.healthcare is not None:
            base_cost = scenario.healthcare.calculate_annual_cost(
                year=year,
                person_birth_year=scenario.person.birth_year,
            )
            if base_cost > 0:
                total_expenses += base_cost * cumulative_healthcare_factor

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

        # Fix #3: pass employment_income for FICA; Fix #12: state deduction
        tax_result = calculate_taxes(
            gross_income=gross_income,
            pre_tax_deductions=pre_tax_deductions,
            standard_deduction=adjusted_standard_deduction,
            state_rate=scenario.assumptions.state_tax_rate,
            federal_brackets=adjusted_brackets,
            ss_wage_cap=adjusted_ss_wage_cap,
            employment_income=employment_income,
            state_standard_deduction=adjusted_state_standard_deduction,
        )

        # Calculate actual savings
        net_income = gross_income - tax_result.total_tax
        savings = net_income - total_expenses
        cumulative_taxes += tax_result.total_tax

        # Fix #5: Use MAGI for Roth IRA phase-out
        magi = gross_income - pre_tax_deductions

        # Allocate savings to assets
        if savings > 0:
            assets = _allocate_savings(
                savings=savings,
                assets=assets,
                asset_configs=scenario.assets,
                age=age,
                gross_income=magi,
                limit_401k=adjusted_401k_limit,
                limit_401k_catchup=adjusted_401k_catchup,
                limit_hsa=adjusted_hsa_limit,
                limit_ira=adjusted_ira_limit,
                limit_ira_catchup=adjusted_ira_catchup,
                roth_phase_out_start=adjusted_roth_phase_out_start,
                roth_phase_out_end=adjusted_roth_phase_out_end,
            )
        elif savings < 0:
            gain_fraction = 0.50
            ltcg_rate = 0.15

            needed = -savings
            for name, config in [(a.name, a) for a in scenario.assets]:
                if config.type == "taxable" and name in assets and needed > 0:
                    tax_drag = ltcg_rate * gain_fraction
                    gross_needed = needed / (1 - tax_drag)

                    gross_draw = min(gross_needed, assets[name])
                    assets[name] -= gross_draw
                    yearly_assets_sold += gross_draw

                    cap_gains_tax = gross_draw * gain_fraction * ltcg_rate
                    yearly_cap_gains_tax += cap_gains_tax

                    net_proceeds = gross_draw - cap_gains_tax
                    needed -= net_proceeds
                    savings += net_proceeds

        # Add capital gains tax to cumulative taxes
        cumulative_taxes += yearly_cap_gains_tax

        # Fix #6: net_income includes all taxes
        total_tax_with_cap_gains = tax_result.total_tax + yearly_cap_gains_tax
        net_income = gross_income - total_tax_with_cap_gains

        # Rebalance: move excess cash above buffer target to investments
        assets = _rebalance_cash_to_investments(
            assets=assets,
            asset_configs=scenario.assets,
            annual_expenses=total_expenses,
            cash_buffer_months=scenario.assumptions.cash_buffer_months,
        )

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

        # Calculate net worth (total and accessible)
        net_worth = sum(assets.values())
        accessible_net_worth = _calculate_accessible_net_worth(assets, scenario.assets)

        # Create snapshot
        snapshot = YearlySnapshot(
            year=year,
            age=age,
            gross_income=gross_income,
            federal_tax=tax_result.federal_tax,
            state_tax=tax_result.state_tax,
            fica_tax=tax_result.fica_tax,
            capital_gains_tax=yearly_cap_gains_tax,
            total_tax=total_tax_with_cap_gains,
            net_income=net_income,
            total_expenses=total_expenses,
            savings=savings,
            assets_sold=yearly_assets_sold,
            assets=dict(assets),
            net_worth=net_worth,
            accessible_net_worth=accessible_net_worth,
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
    # Use mean general inflation for bridge calculation
    general_inflation_mean = float(scenario.assumptions.get_inflation_rate("general").mean)
    for snapshot in snapshots:
        year = snapshot.year
        age = snapshot.age

        year_fire_target = snapshot.total_expenses if fire_basis == "current" else fire_target

        if milestones["retirement_ready"] is None and snapshot.net_worth >= year_fire_target * 25:
            milestones["retirement_ready"] = year

        if milestones["fire_number"] is None:
            years_until_irs = scenario.assumptions.get_years_until_irs_access(age)

            if years_until_irs <= 0:
                is_fire_ready = snapshot.net_worth * 0.04 >= year_fire_target
            else:
                # Fix #11: bridge accounts for inflation
                bridge_needed = _calculate_bridge_needed(
                    year_fire_target, years_until_irs, general_inflation_mean
                )

                retirement_accounts = snapshot.net_worth - snapshot.accessible_net_worth

                expected_return = scenario.assumptions.market_return_mean
                projected_retirement = retirement_accounts * ((1 + expected_return) ** years_until_irs)

                has_bridge = snapshot.accessible_net_worth >= bridge_needed
                has_post_irs = projected_retirement * 0.04 >= year_fire_target

                is_fire_ready = has_bridge and has_post_irs

            if is_fire_ready:
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
        expenses_arr = np.array([s.total_expenses for s in year_data])
        savings_arr = np.array([s.savings for s in year_data])

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
            total_expenses_median=float(np.median(expenses_arr)),
            savings_median=float(np.median(savings_arr)),
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

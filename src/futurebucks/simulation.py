"""Year-by-year financial simulation engine."""

from datetime import datetime
from copy import deepcopy

from futurebucks.models import (
    Scenario,
    YearlySnapshot,
    SimulationResult,
    IncomeSource,
    Asset,
    Expense,
    LifeEvent,
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


def _calculate_roth_ira_limit(
    income: float,
    base_limit: float,
    phase_out_start: float,
    phase_out_end: float,
) -> float:
    """Calculate Roth IRA contribution limit based on income phase-out.

    Args:
        income: Modified Adjusted Gross Income (MAGI)
        base_limit: Base IRA contribution limit for the year
        phase_out_start: Income where phase-out begins
        phase_out_end: Income where contributions are fully phased out

    Returns:
        Maximum Roth IRA contribution allowed
    """
    if income <= phase_out_start:
        return base_limit
    elif income >= phase_out_end:
        return 0.0
    else:
        # Linear phase-out
        phase_out_range = phase_out_end - phase_out_start
        income_over = income - phase_out_start
        reduction_ratio = income_over / phase_out_range
        return base_limit * (1 - reduction_ratio)


def _is_active(item: IncomeSource | Expense, year: int) -> bool:
    """Check if an income source or expense is active in a given year."""
    if item.start_year is not None and year < item.start_year:
        return False
    if item.end_year is not None and year > item.end_year:
        return False
    return True


def _calculate_gross_income(
    income_sources: list[IncomeSource],
    year: int,
    start_year: int,
) -> float:
    """Calculate total gross income for a year."""
    total = 0.0
    for source in income_sources:
        if not _is_active(source, year):
            continue

        # Calculate years of growth since income started
        source_start = source.start_year or start_year
        years_active = max(0, year - source_start)
        amount = source.amount * ((1 + source.growth_rate) ** years_active)
        total += amount

    return total


def _calculate_expenses(
    expenses: list[Expense],
    year: int,
    start_year: int,
    inflation_rate: float,
) -> float:
    """Calculate total expenses for a year."""
    total = 0.0
    for expense in expenses:
        if not _is_active(expense, year):
            continue

        expense_start = expense.start_year or start_year
        years_active = max(0, year - expense_start)

        if expense.inflation_adjusted:
            amount = expense.amount * ((1 + inflation_rate) ** years_active)
        else:
            amount = expense.amount

        total += amount

    return total


def _apply_life_events(
    year: int,
    income_sources: list[IncomeSource],
    expenses: list[Expense],
    assets: dict[str, float],
    events: list[LifeEvent],
    inflation_rate: float,
    start_year: int,
) -> tuple[list[IncomeSource], list[Expense], dict[str, float], float]:
    """Apply life events for a given year.

    Returns updated income sources, expenses, assets, and any windfall amount.
    """
    windfall = 0.0

    for event in events:
        if event.year != year:
            continue

        if event.type == "income_change":
            # Find and update the income source or add new
            source_name = event.details.get("source_name", "Primary Job")
            new_amount = event.details.get("new_amount", 0)
            growth_rate = event.details.get("growth_rate", 0.03)

            found = False
            for i, source in enumerate(income_sources):
                if source.name == source_name:
                    income_sources[i] = IncomeSource(
                        name=source_name,
                        amount=new_amount,
                        growth_rate=growth_rate,
                        start_year=year,
                        end_year=source.end_year,
                    )
                    found = True
                    break

            if not found:
                income_sources.append(
                    IncomeSource(
                        name=source_name,
                        amount=new_amount,
                        growth_rate=growth_rate,
                        start_year=year,
                    )
                )

        elif event.type == "expense_change":
            expense_name = event.details.get("expense_name", "New Expense")
            new_amount = event.details.get("new_amount", 0)
            inflation_adjusted = event.details.get("inflation_adjusted", True)

            found = False
            for i, expense in enumerate(expenses):
                if expense.name == expense_name:
                    expenses[i] = Expense(
                        name=expense_name,
                        amount=new_amount,
                        inflation_adjusted=inflation_adjusted,
                        start_year=year,
                        end_year=expense.end_year,
                    )
                    found = True
                    break

            if not found:
                expenses.append(
                    Expense(
                        name=expense_name,
                        amount=new_amount,
                        inflation_adjusted=inflation_adjusted,
                        start_year=year,
                    )
                )

        elif event.type == "windfall":
            windfall += event.details.get("amount", 0)
            target_asset = event.details.get("target_asset")
            if target_asset and target_asset in assets:
                assets[target_asset] += event.details.get("amount", 0)
                windfall = 0  # Already allocated

        elif event.type == "asset_purchase":
            asset_name = event.details.get("asset_name", "New Asset")
            cost = event.details.get("cost", 0)
            asset_value = event.details.get("asset_value", cost)

            # Deduct from taxable or specified source
            source_asset = event.details.get("source_asset", "Taxable Brokerage")
            if source_asset in assets:
                assets[source_asset] = max(0, assets[source_asset] - cost)

            assets[asset_name] = asset_value

        elif event.type == "child":
            annual_cost = event.details.get("annual_cost", 20000)
            child_name = event.details.get("name", f"Child (born {year})")
            end_year = event.details.get("end_year", year + 18)

            expenses.append(
                Expense(
                    name=child_name,
                    amount=annual_cost,
                    inflation_adjusted=True,
                    start_year=year,
                    end_year=end_year,
                )
            )

        elif event.type == "retirement":
            # End all income sources
            for i, source in enumerate(income_sources):
                if source.end_year is None:
                    income_sources[i] = IncomeSource(
                        name=source.name,
                        amount=source.amount,
                        growth_rate=source.growth_rate,
                        start_year=source.start_year,
                        end_year=year - 1,
                    )

    return income_sources, expenses, assets, windfall


def _allocate_savings(
    savings: float,
    assets: dict[str, float],
    asset_configs: list[Asset],
    age: int,
    gross_income: float,
    limit_401k: float,
    limit_401k_catchup: float,
    limit_hsa: float,
    limit_ira: float,
    limit_ira_catchup: float,
    roth_phase_out_start: float,
    roth_phase_out_end: float,
) -> dict[str, float]:
    """Allocate savings to various accounts.

    Priority: 401k (to limit) -> HSA -> Roth IRA (if eligible) -> Taxable
    """
    remaining = savings
    if remaining <= 0:
        return assets

    # Create lookup for asset configs
    config_by_name = {a.name: a for a in asset_configs}

    # 401k contribution (inflation-adjusted limit)
    effective_401k_limit = limit_401k
    if age >= 50:
        effective_401k_limit += limit_401k_catchup

    for name, config in config_by_name.items():
        if config.type == "401k" and remaining > 0:
            contribution = min(remaining, effective_401k_limit)
            assets[name] = assets.get(name, 0) + contribution
            remaining -= contribution
            break

    # HSA contribution (inflation-adjusted limit)
    for name, config in config_by_name.items():
        if config.type == "hsa" and remaining > 0:
            contribution = min(remaining, limit_hsa)
            assets[name] = assets.get(name, 0) + contribution
            remaining -= contribution
            break

    # Roth IRA contribution (with income phase-out check)
    base_ira_limit = limit_ira
    if age >= 50:
        base_ira_limit += limit_ira_catchup

    # Calculate actual Roth IRA limit based on income
    effective_ira_limit = _calculate_roth_ira_limit(
        income=gross_income,
        base_limit=base_ira_limit,
        phase_out_start=roth_phase_out_start,
        phase_out_end=roth_phase_out_end,
    )

    if effective_ira_limit > 0:
        for name, config in config_by_name.items():
            if config.type == "roth_ira" and remaining > 0:
                contribution = min(remaining, effective_ira_limit)
                assets[name] = assets.get(name, 0) + contribution
                remaining -= contribution
                break

    # Remainder to taxable
    if remaining > 0:
        for name, config in config_by_name.items():
            if config.type == "taxable":
                assets[name] = assets.get(name, 0) + remaining
                remaining = 0
                break

        # If no taxable account, create one
        if remaining > 0:
            assets["Taxable Brokerage"] = assets.get("Taxable Brokerage", 0) + remaining

    return assets


def _apply_returns(
    assets: dict[str, float],
    asset_configs: list[Asset],
    default_return: float,
) -> dict[str, float]:
    """Apply investment returns to all assets."""
    config_by_name = {a.name: a for a in asset_configs}

    for name in assets:
        if name in config_by_name:
            expected_return = config_by_name[name].expected_return
        else:
            expected_return = default_return

        assets[name] *= 1 + expected_return

    return assets


def _calculate_pre_tax_deductions(
    savings: float,
    asset_configs: list[Asset],
    age: int,
    limit_401k: float,
    limit_401k_catchup: float,
    limit_hsa: float,
) -> float:
    """Calculate pre-tax deductions (401k, HSA contributions)."""
    deductions = 0.0

    if savings <= 0:
        return 0.0

    remaining = savings
    config_by_name = {a.name: a for a in asset_configs}

    # 401k contribution (inflation-adjusted limit)
    effective_401k_limit = limit_401k
    if age >= 50:
        effective_401k_limit += limit_401k_catchup

    for name, config in config_by_name.items():
        if config.type == "401k" and remaining > 0:
            contribution = min(remaining, effective_401k_limit)
            deductions += contribution
            remaining -= contribution
            break

    # HSA contribution (inflation-adjusted limit)
    for name, config in config_by_name.items():
        if config.type == "hsa" and remaining > 0:
            contribution = min(remaining, limit_hsa)
            deductions += contribution
            remaining -= contribution
            break

    return deductions


def run_simulation(scenario: Scenario) -> SimulationResult:
    """Run a complete financial simulation.

    Args:
        scenario: The scenario configuration to simulate

    Returns:
        SimulationResult with yearly snapshots and milestones
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
    inflation_rate = scenario.assumptions.inflation_rate

    for year in range(current_year, end_year + 1):
        age = year - scenario.person.birth_year
        years_from_base = max(0, year - BASE_YEAR)

        # Calculate inflation-adjusted values for this year
        adjusted_standard_deduction = inflate_value(
            BASE_STANDARD_DEDUCTION_MFJ, inflation_rate, years_from_base
        )
        adjusted_ss_wage_cap = inflate_value(
            BASE_SS_WAGE_CAP, inflation_rate, years_from_base
        )
        adjusted_401k_limit = inflate_value(
            BASE_401K_LIMIT, inflation_rate, years_from_base
        )
        adjusted_401k_catchup = inflate_value(
            BASE_401K_CATCHUP, inflation_rate, years_from_base
        )
        adjusted_ira_limit = inflate_value(
            BASE_IRA_LIMIT, inflation_rate, years_from_base
        )
        adjusted_ira_catchup = inflate_value(
            BASE_IRA_CATCHUP, inflation_rate, years_from_base
        )
        adjusted_hsa_limit = inflate_value(
            BASE_HSA_LIMIT_FAMILY, inflation_rate, years_from_base
        )
        adjusted_roth_phase_out_start = inflate_value(
            BASE_ROTH_IRA_INCOME_LIMIT_START, inflation_rate, years_from_base
        )
        adjusted_roth_phase_out_end = inflate_value(
            BASE_ROTH_IRA_INCOME_LIMIT_END, inflation_rate, years_from_base
        )
        adjusted_brackets = inflate_brackets(
            FEDERAL_BRACKETS_2024_MFJ, inflation_rate, years_from_base
        )

        # Apply life events for this year
        income_sources, expenses, assets, windfall = _apply_life_events(
            year=year,
            income_sources=income_sources,
            expenses=expenses,
            assets=assets,
            events=scenario.life_events,
            inflation_rate=inflation_rate,
            start_year=current_year,
        )

        # Calculate income
        gross_income = _calculate_gross_income(income_sources, year, current_year)
        gross_income += windfall

        # Calculate expenses
        total_expenses = _calculate_expenses(
            expenses, year, current_year, inflation_rate
        )

        # Estimate savings for pre-tax deduction calculation
        estimated_savings = gross_income * 0.7 - total_expenses  # Rough estimate
        pre_tax_deductions = _calculate_pre_tax_deductions(
            max(0, estimated_savings),
            scenario.assets,
            age,
            adjusted_401k_limit,
            adjusted_401k_catchup,
            adjusted_hsa_limit,
        )

        # Calculate taxes with inflation-adjusted brackets and caps
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

        # Allocate savings to assets (with Roth IRA income check)
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

        # Apply investment returns
        assets = _apply_returns(
            assets=assets,
            asset_configs=scenario.assets,
            default_return=scenario.assumptions.market_return_mean,
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
        if milestones["net_worth_100k"] is None and net_worth >= 100_000:
            milestones["net_worth_100k"] = year
        if milestones["net_worth_500k"] is None and net_worth >= 500_000:
            milestones["net_worth_500k"] = year
        if milestones["net_worth_1m"] is None and net_worth >= 1_000_000:
            milestones["net_worth_1m"] = year
        if milestones["net_worth_2m"] is None and net_worth >= 2_000_000:
            milestones["net_worth_2m"] = year
        if milestones["net_worth_3m"] is None and net_worth >= 3_000_000:
            milestones["net_worth_3m"] = year
        if milestones["net_worth_5m"] is None and net_worth >= 5_000_000:
            milestones["net_worth_5m"] = year
        if milestones["net_worth_10m"] is None and net_worth >= 10_000_000:
            milestones["net_worth_10m"] = year
        if milestones["net_worth_20m"] is None and net_worth >= 20_000_000:
            milestones["net_worth_20m"] = year
        if milestones["net_worth_50m"] is None and net_worth >= 50_000_000:
            milestones["net_worth_50m"] = year
        if milestones["net_worth_100m"] is None and net_worth >= 100_000_000:
            milestones["net_worth_100m"] = year

    # Calculate FIRE milestone based on configured expense basis
    fire_basis = scenario.assumptions.fire_expense_basis

    if fire_basis == "max":
        # Use maximum expenses across all years
        fire_target = max(s.total_expenses for s in snapshots)
    elif fire_basis == "first_year":
        # Use first year's expenses
        fire_target = snapshots[0].total_expenses
    elif fire_basis == "target":
        # Use specified target, or fall back to max if not set
        fire_target = scenario.assumptions.fire_target_expenses
        if fire_target is None:
            fire_target = max(s.total_expenses for s in snapshots)
    else:  # "current" - will be handled per-year below
        fire_target = None

    # Check FIRE and retirement milestones
    for snapshot in snapshots:
        year = snapshot.year
        net_worth = snapshot.net_worth

        # For "current" basis, use that year's expenses
        if fire_basis == "current":
            year_fire_target = snapshot.total_expenses
        else:
            year_fire_target = fire_target

        # Check retirement readiness (25x annual expenses based on fire target)
        if milestones["retirement_ready"] is None and net_worth >= year_fire_target * 25:
            milestones["retirement_ready"] = year

        # FIRE number: 4% safe withdrawal rate covers expenses
        if milestones["fire_number"] is None and net_worth * 0.04 >= year_fire_target:
            milestones["fire_number"] = year

    return SimulationResult(
        scenario_name=scenario.name,
        snapshots=snapshots,
        milestones=milestones,
        fire_target_expenses=fire_target if fire_target else max(s.total_expenses for s in snapshots),
    )

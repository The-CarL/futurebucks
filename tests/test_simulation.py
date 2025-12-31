"""Tests for the simulation engine."""

import pytest
from datetime import datetime

from futurebucks.models import (
    Scenario,
    PersonConfig,
    IncomeSource,
    Asset,
    Expense,
    LifeEvent,
    Assumptions,
)
from futurebucks.simulation import run_simulation


@pytest.fixture
def basic_scenario():
    """Create a basic test scenario."""
    return Scenario(
        name="Test Scenario",
        description="A basic test scenario",
        person=PersonConfig(
            name="Test User",
            birth_year=1990,
            retirement_age=65,
        ),
        income_sources=[
            IncomeSource(
                name="Primary Job",
                amount=100000,
                growth_rate=0.03,
            ),
        ],
        assets=[
            Asset(
                name="Taxable Brokerage",
                type="taxable",
                balance=100000,
                expected_return=0.07,
            ),
            Asset(
                name="401k",
                type="401k",
                balance=50000,
                expected_return=0.07,
            ),
        ],
        expenses=[
            Expense(
                name="Living Expenses",
                amount=50000,
                inflation_adjusted=True,
            ),
        ],
        life_events=[],
        assumptions=Assumptions(),
        simulation_years=10,
    )


class TestSimulationBasics:
    """Basic simulation tests."""

    def test_simulation_runs(self, basic_scenario):
        """Simulation should complete without errors."""
        result = run_simulation(basic_scenario)
        assert result is not None
        assert result.scenario_name == "Test Scenario"

    def test_correct_number_of_snapshots(self, basic_scenario):
        """Should produce correct number of yearly snapshots."""
        result = run_simulation(basic_scenario)
        # +1 because it includes both start and end years
        assert len(result.snapshots) == basic_scenario.simulation_years + 1

    def test_snapshots_have_correct_years(self, basic_scenario):
        """Snapshots should have sequential years."""
        result = run_simulation(basic_scenario)
        current_year = datetime.now().year

        for i, snapshot in enumerate(result.snapshots):
            assert snapshot.year == current_year + i

    def test_age_calculation(self, basic_scenario):
        """Age should be calculated correctly."""
        result = run_simulation(basic_scenario)
        current_year = datetime.now().year
        expected_age = current_year - basic_scenario.person.birth_year

        assert result.snapshots[0].age == expected_age


class TestIncomeCalculation:
    """Tests for income calculations."""

    def test_income_growth(self, basic_scenario):
        """Income should grow at specified rate."""
        result = run_simulation(basic_scenario)

        first_income = result.snapshots[0].gross_income
        second_income = result.snapshots[1].gross_income

        expected_growth = first_income * 1.03
        assert second_income == pytest.approx(expected_growth, rel=0.01)

    def test_multiple_income_sources(self, basic_scenario):
        """Multiple income sources should be summed."""
        basic_scenario.income_sources.append(
            IncomeSource(
                name="Side Job",
                amount=20000,
                growth_rate=0.02,
            )
        )

        result = run_simulation(basic_scenario)
        # First year income should be sum of both sources
        assert result.snapshots[0].gross_income == pytest.approx(120000, rel=0.01)


class TestExpenseCalculation:
    """Tests for expense calculations."""

    def test_expense_inflation(self, basic_scenario):
        """Expenses should inflate at specified rate."""
        result = run_simulation(basic_scenario)

        first_expenses = result.snapshots[0].total_expenses
        second_expenses = result.snapshots[1].total_expenses

        inflation_rate = basic_scenario.assumptions.inflation_rate
        expected = first_expenses * (1 + inflation_rate)
        assert second_expenses == pytest.approx(expected, rel=0.01)

    def test_non_inflation_adjusted_expense(self, basic_scenario):
        """Non-inflation-adjusted expenses should stay constant."""
        basic_scenario.expenses = [
            Expense(
                name="Fixed Expense",
                amount=30000,
                inflation_adjusted=False,
            )
        ]

        result = run_simulation(basic_scenario)
        assert result.snapshots[0].total_expenses == result.snapshots[5].total_expenses


class TestTaxCalculation:
    """Tests for tax calculations."""

    def test_taxes_are_calculated(self, basic_scenario):
        """All tax components should be calculated."""
        result = run_simulation(basic_scenario)
        snapshot = result.snapshots[0]

        assert snapshot.federal_tax > 0
        assert snapshot.state_tax > 0
        assert snapshot.fica_tax > 0
        assert snapshot.total_tax == pytest.approx(
            snapshot.federal_tax + snapshot.state_tax + snapshot.fica_tax
        )

    def test_net_income_calculation(self, basic_scenario):
        """Net income should be gross minus taxes."""
        result = run_simulation(basic_scenario)
        snapshot = result.snapshots[0]

        expected_net = snapshot.gross_income - snapshot.total_tax
        assert snapshot.net_income == pytest.approx(expected_net, rel=0.01)


class TestAssetGrowth:
    """Tests for asset growth."""

    def test_assets_grow(self, basic_scenario):
        """Assets should grow over time."""
        result = run_simulation(basic_scenario)

        first_net_worth = result.snapshots[0].net_worth
        last_net_worth = result.snapshots[-1].net_worth

        assert last_net_worth > first_net_worth

    def test_savings_allocated(self, basic_scenario):
        """Positive savings should increase assets."""
        result = run_simulation(basic_scenario)

        # With $100k income and $50k expenses, there should be savings
        assert result.snapshots[0].savings > 0


class TestLifeEvents:
    """Tests for life events."""

    def test_income_change_event(self, basic_scenario):
        """Income change events should modify income."""
        current_year = datetime.now().year

        basic_scenario.life_events = [
            LifeEvent(
                name="Big Raise",
                year=current_year + 2,
                type="income_change",
                details={
                    "source_name": "Primary Job",
                    "new_amount": 150000,
                    "growth_rate": 0.03,
                },
            )
        ]

        result = run_simulation(basic_scenario)

        # Income before event
        assert result.snapshots[1].gross_income < 110000

        # Income after event should be around $150k
        assert result.snapshots[2].gross_income >= 150000

    def test_child_event(self, basic_scenario):
        """Child event should add expenses."""
        current_year = datetime.now().year

        basic_scenario.life_events = [
            LifeEvent(
                name="First Child",
                year=current_year + 2,
                type="child",
                details={
                    "annual_cost": 15000,
                },
            )
        ]

        result = run_simulation(basic_scenario)

        expenses_before = result.snapshots[1].total_expenses
        expenses_after = result.snapshots[2].total_expenses

        # Expenses should increase by roughly child cost (accounting for inflation)
        assert expenses_after > expenses_before + 10000

    def test_windfall_event(self, basic_scenario):
        """Windfall event should increase net worth."""
        current_year = datetime.now().year

        basic_scenario.life_events = [
            LifeEvent(
                name="Inheritance",
                year=current_year + 2,
                type="windfall",
                details={
                    "amount": 500000,
                    "target_asset": "Taxable Brokerage",
                },
            )
        ]

        result = run_simulation(basic_scenario)

        # Net worth should jump significantly at windfall year
        net_worth_before = result.snapshots[1].net_worth
        net_worth_after = result.snapshots[2].net_worth

        assert net_worth_after > net_worth_before + 400000


class TestMilestones:
    """Tests for milestone tracking."""

    def test_milestones_tracked(self, basic_scenario):
        """Milestones should be tracked."""
        result = run_simulation(basic_scenario)

        # Should have milestone dict
        assert "net_worth_100k" in result.milestones
        assert "net_worth_1m" in result.milestones
        assert "retirement_ready" in result.milestones

    def test_milestone_reached(self, basic_scenario):
        """Reaching a milestone should record the year."""
        # Start with enough to hit $100k milestone in year 1
        basic_scenario.assets[0].balance = 90000

        result = run_simulation(basic_scenario)

        # Should hit $100k milestone
        assert result.milestones["net_worth_100k"] is not None

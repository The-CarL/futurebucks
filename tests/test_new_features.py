"""Tests for new features: category-specific inflation, healthcare, and sensitivity analysis."""

import pytest
from datetime import datetime

from futurebucks.models import (
    Scenario,
    PersonConfig,
    IncomeSource,
    Asset,
    Expense,
    Assumptions,
    CategoryInflationRates,
    HealthcareConfig,
    Distribution,
    DEFAULT_VOLATILITIES,
    MonteCarloConfig,
)
from futurebucks.simulation import run_simulation
from futurebucks.sensitivity import (
    SensitivityAnalyzer,
    run_sensitivity_analysis,
    TornadoResult,
    WhatIfMatrix,
)


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
            life_expectancy=46,  # age 36 + 10 years of simulation
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
    )


class TestCategoryInflationRates:
    """Tests for category-specific inflation rates."""

    def test_default_inflation_rates(self):
        """Default inflation rates should be set correctly."""
        rates = CategoryInflationRates()

        assert rates.general.mean == pytest.approx(0.025, rel=0.01)
        assert rates.healthcare.mean == pytest.approx(0.055, rel=0.01)
        assert rates.education.mean == pytest.approx(0.04, rel=0.01)
        assert rates.housing.mean == pytest.approx(0.03, rel=0.01)

    def test_get_rate_existing_category(self):
        """Should return the correct rate for known categories."""
        rates = CategoryInflationRates()

        general = rates.get_rate("general")
        assert general.mean == pytest.approx(0.025, rel=0.01)

        healthcare = rates.get_rate("healthcare")
        assert healthcare.mean == pytest.approx(0.055, rel=0.01)

    def test_get_rate_unknown_category(self):
        """Should return general rate for unknown categories."""
        rates = CategoryInflationRates()

        unknown = rates.get_rate("unknown_category")
        assert unknown.mean == rates.general.mean

    def test_assumptions_get_inflation_rate(self):
        """Assumptions should provide inflation rate by category."""
        assumptions = Assumptions()

        general = assumptions.get_inflation_rate("general")
        assert general.mean == pytest.approx(0.025, rel=0.01)

        healthcare = assumptions.get_inflation_rate("healthcare")
        assert healthcare.mean == pytest.approx(0.055, rel=0.01)

    def test_legacy_inflation_rate_conversion(self):
        """Legacy inflation_rate should be converted to inflation_rates.general."""
        assumptions = Assumptions(
            inflation_rate=Distribution(mean=0.03, stddev=0.01)
        )

        # Legacy rate should be used as general
        general = assumptions.get_inflation_rate("general")
        assert general.mean == pytest.approx(0.03, rel=0.01)

        # Healthcare should still use default
        healthcare = assumptions.get_inflation_rate("healthcare")
        assert healthcare.mean == pytest.approx(0.055, rel=0.01)

    def test_expense_inflation_category(self, basic_scenario):
        """Expenses should use their specified inflation category."""
        # Add healthcare expense with healthcare inflation
        basic_scenario.expenses.append(
            Expense(
                name="Medical Bills",
                amount=5000,
                inflation_adjusted=True,
                inflation_category="healthcare",
            )
        )

        result = run_simulation(basic_scenario)

        # Just verify simulation runs without error
        assert result is not None
        assert len(result.snapshots) == basic_scenario.simulation_years + 1


class TestHealthcareConfig:
    """Tests for healthcare cost modeling."""

    def test_default_healthcare_config(self):
        """Default healthcare config should have sensible values."""
        config = HealthcareConfig()

        assert config.coverage_type == "family"
        assert config.employer_monthly_cost == 400.0
        assert config.monthly_premium == 1500.0
        assert config.annual_deductible == 6000.0
        assert config.annual_out_of_pocket_max == 12000.0
        assert config.expected_utilization == 0.3

    def test_employer_subsidized_cost(self):
        """Cost during employer coverage should be lower."""
        config = HealthcareConfig(
            employer_subsidized_until=2030,
            employer_monthly_cost=400,
        )

        # Before end of employer coverage
        cost_2025 = config.calculate_annual_cost(year=2025, person_birth_year=1990)
        assert cost_2025 == 400 * 12  # $4,800

    def test_marketplace_cost(self):
        """Cost after employer coverage should be higher."""
        config = HealthcareConfig(
            employer_subsidized_until=2025,
            monthly_premium=1500,
            annual_deductible=6000,
            annual_out_of_pocket_max=12000,
            expected_utilization=0.3,
        )

        # After employer coverage ends
        cost_2030 = config.calculate_annual_cost(year=2030, person_birth_year=1990)

        expected = (1500 * 12) + 6000 + (12000 * 0.3)  # $27,600
        assert cost_2030 == pytest.approx(expected, rel=0.01)

    def test_medicare_eligibility(self):
        """Healthcare cost should be zero after Medicare eligibility (age 65)."""
        config = HealthcareConfig()

        # Person born in 1960, year 2025 = age 65
        cost = config.calculate_annual_cost(year=2025, person_birth_year=1960)
        assert cost == 0.0

        # Person born in 1960, year 2024 = age 64 (still needs coverage)
        cost = config.calculate_annual_cost(year=2024, person_birth_year=1960)
        assert cost > 0.0

    def test_scenario_with_healthcare(self, basic_scenario):
        """Scenario with healthcare config should run correctly."""
        basic_scenario.healthcare = HealthcareConfig(
            employer_subsidized_until=datetime.now().year + 5,
            employer_monthly_cost=400,
            monthly_premium=1500,
        )

        result = run_simulation(basic_scenario)

        # Verify simulation runs
        assert result is not None

        # Healthcare should be included in expenses
        # First year should include healthcare cost
        first_year_expenses = result.snapshots[0].total_expenses
        assert first_year_expenses > 50000  # Base expenses plus healthcare


class TestDefaultVolatilities:
    """Tests for default volatility values."""

    def test_asset_volatilities_exist(self):
        """All asset types should have default volatilities."""
        asset_types = ["401k", "roth_ira", "traditional_ira", "taxable", "hsa", "real_estate", "other"]

        for asset_type in asset_types:
            assert asset_type in DEFAULT_VOLATILITIES
            assert DEFAULT_VOLATILITIES[asset_type] > 0

    def test_inflation_volatilities_exist(self):
        """Inflation categories should have default volatilities."""
        categories = ["inflation_general", "inflation_healthcare", "inflation_education", "inflation_housing"]

        for category in categories:
            assert category in DEFAULT_VOLATILITIES
            assert DEFAULT_VOLATILITIES[category] > 0

    def test_income_growth_volatility(self):
        """Income growth should have a default volatility."""
        assert "income_growth" in DEFAULT_VOLATILITIES
        assert DEFAULT_VOLATILITIES["income_growth"] > 0

    def test_equity_volatility_higher_than_real_estate(self):
        """Equity assets should be more volatile than real estate."""
        assert DEFAULT_VOLATILITIES["401k"] > DEFAULT_VOLATILITIES["real_estate"]
        assert DEFAULT_VOLATILITIES["taxable"] > DEFAULT_VOLATILITIES["real_estate"]


class TestMonteCarloConfigUpdates:
    """Tests for updated Monte Carlo configuration."""

    def test_default_num_simulations(self):
        """Default simulations should be 2000."""
        config = MonteCarloConfig()
        assert config.num_simulations == 2000

    def test_detailed_analysis_mode(self):
        """Detailed analysis mode should use 10000 simulations."""
        config = MonteCarloConfig(detailed_analysis_mode=True)
        assert config.effective_num_simulations == 10000

    def test_normal_mode_uses_configured_value(self):
        """Normal mode should use the configured value."""
        config = MonteCarloConfig(num_simulations=500)
        assert config.effective_num_simulations == 500


class TestSensitivityAnalyzer:
    """Tests for sensitivity analysis."""

    def test_analyzer_creation(self, basic_scenario):
        """Should create analyzer from scenario."""
        analyzer = SensitivityAnalyzer(basic_scenario)
        assert analyzer.scenario == basic_scenario

    def test_resolve_path_simple(self, basic_scenario):
        """Should resolve simple paths."""
        analyzer = SensitivityAnalyzer(basic_scenario)

        # Tuple format: (resolved_path, current_value, is_distribution, item_name)
        resolved = analyzer._resolve_path("start_year")
        assert len(resolved) == 1
        assert resolved[0][1] == basic_scenario.start_year

    def test_resolve_path_nested(self, basic_scenario):
        """Should resolve nested paths."""
        analyzer = SensitivityAnalyzer(basic_scenario)

        # Tuple format: (resolved_path, current_value, is_distribution, item_name)
        resolved = analyzer._resolve_path("assumptions.state_tax_rate")
        assert len(resolved) == 1
        assert resolved[0][1] == pytest.approx(0.0525, rel=0.01)

    def test_resolve_path_wildcard(self, basic_scenario):
        """Should resolve wildcard paths."""
        analyzer = SensitivityAnalyzer(basic_scenario)

        # Tuple format: (resolved_path, current_value, is_distribution, item_name)
        resolved = analyzer._resolve_path("income_sources.*.amount")
        assert len(resolved) == 1
        assert resolved[0][1] == 100000
        assert resolved[0][3] == "Primary Job"  # item_name should be captured

    def test_resolve_path_indexed(self, basic_scenario):
        """Should resolve indexed paths like income_sources[0].amount."""
        analyzer = SensitivityAnalyzer(basic_scenario)

        # Tuple format: (resolved_path, current_value, is_distribution, item_name)
        resolved = analyzer._resolve_path("income_sources[0].amount")
        assert len(resolved) == 1
        assert resolved[0][1] == 100000
        assert resolved[0][3] == "Primary Job"  # item_name should be captured

        # Test assets too
        resolved = analyzer._resolve_path("assets[0].balance")
        assert len(resolved) == 1
        assert resolved[0][1] == 100000
        assert resolved[0][3] == "Taxable Brokerage"

    def test_tornado_analysis_runs(self, basic_scenario):
        """Tornado analysis should complete."""
        result = run_sensitivity_analysis(
            scenario=basic_scenario,
            target_metric="final_net_worth",
            variation_pct=0.10,
            top_n=5,
        )

        assert isinstance(result, TornadoResult)
        assert result.metric_name == "final_net_worth"
        assert result.base_value > 0
        assert len(result.sensitivities) <= 5

    def test_tornado_results_sorted_by_impact(self, basic_scenario):
        """Results should be sorted by impact (descending)."""
        result = run_sensitivity_analysis(
            scenario=basic_scenario,
            target_metric="final_net_worth",
            variation_pct=0.20,
            top_n=5,
        )

        if len(result.sensitivities) > 1:
            for i in range(len(result.sensitivities) - 1):
                assert result.sensitivities[i].impact >= result.sensitivities[i + 1].impact

    def test_sensitivity_result_structure(self, basic_scenario):
        """Sensitivity results should have correct structure."""
        result = run_sensitivity_analysis(
            scenario=basic_scenario,
            target_metric="final_net_worth",
            variation_pct=0.20,
            top_n=3,
        )

        for sens in result.sensitivities:
            assert sens.variable_path
            assert sens.variable_label
            assert sens.base_value > 0 or sens.base_value == 0
            assert sens.low_value != sens.high_value
            assert sens.impact >= 0

    def test_what_if_matrix(self, basic_scenario):
        """What-if matrix analysis should work."""
        analyzer = SensitivityAnalyzer(basic_scenario)

        result = analyzer.run_what_if_matrix(
            variable1_path="simulation_years",
            variable1_range=(5, 15, 3),
            variable2_path="assumptions.state_tax_rate",
            variable2_range=(0.03, 0.07, 3),
            target_metric="final_net_worth",
        )

        assert isinstance(result, WhatIfMatrix)
        assert len(result.variable1_values) == 3
        assert len(result.variable2_values) == 3
        assert len(result.cells) == 9  # 3 x 3

    def test_fire_year_metric(self, basic_scenario):
        """Should be able to analyze FIRE year metric."""
        result = run_sensitivity_analysis(
            scenario=basic_scenario,
            target_metric="fire_year",
            variation_pct=0.20,
            top_n=3,
        )

        assert result.metric_name == "fire_year"
        # Base value should be a year or large number if not achieved
        assert result.base_value >= datetime.now().year


class TestConservativeDefaults:
    """Tests for conservative defaults mode."""

    def test_conservative_mode_flag(self):
        """Conservative mode flag should be configurable."""
        assumptions = Assumptions(use_conservative_defaults=True)
        assert assumptions.use_conservative_defaults is True

        assumptions = Assumptions(use_conservative_defaults=False)
        assert assumptions.use_conservative_defaults is False

    def test_conservative_mode_in_scenario(self, basic_scenario):
        """Conservative mode should be usable in scenario."""
        basic_scenario.assumptions.use_conservative_defaults = True

        # Just verify it doesn't break anything
        result = run_simulation(basic_scenario)
        assert result is not None


class TestIntegration:
    """Integration tests for all new features working together."""

    def test_full_scenario_with_all_features(self):
        """Scenario using all new features should work correctly."""
        scenario = Scenario(
            name="Full Feature Test",
            description="Test all new features",
            person=PersonConfig(
                name="Test User",
                birth_year=1985,
                retirement_age=60,
                life_expectancy=61,  # age 41 + 20 years = 61
            ),
            income_sources=[
                IncomeSource(
                    name="Primary Job",
                    amount=120000,
                    growth_rate=Distribution(mean=0.03, stddev=0.02),
                ),
            ],
            assets=[
                Asset(
                    name="401k",
                    type="401k",
                    balance=200000,
                    expected_return=Distribution(mean=0.07, stddev=0.15),
                ),
                Asset(
                    name="Taxable",
                    type="taxable",
                    balance=100000,
                    expected_return=0.07,  # Will use DEFAULT_VOLATILITIES
                ),
            ],
            expenses=[
                Expense(
                    name="Living",
                    amount=60000,
                    inflation_adjusted=True,
                    inflation_category="general",
                ),
                Expense(
                    name="Medical",
                    amount=5000,
                    inflation_adjusted=True,
                    inflation_category="healthcare",
                ),
            ],
            healthcare=HealthcareConfig(
                employer_subsidized_until=datetime.now().year + 5,
                employer_monthly_cost=400,
                monthly_premium=1800,
            ),
            assumptions=Assumptions(
                use_conservative_defaults=False,
            ),
            life_events=[],
        )

        result = run_simulation(scenario)

        # Verify simulation completes (simulation_years computed from life_expectancy)
        assert result is not None
        assert len(result.snapshots) == scenario.simulation_years + 1

        # Verify net worth grows
        assert result.snapshots[-1].net_worth > result.snapshots[0].net_worth

        # Run sensitivity analysis
        sensitivity = run_sensitivity_analysis(
            scenario=scenario,
            target_metric="final_net_worth",
            variation_pct=0.15,
            top_n=5,
        )

        assert len(sensitivity.sensitivities) > 0
        assert sensitivity.base_value > 0

"""Tests for Monte Carlo simulation."""

import pytest
import numpy as np
from datetime import datetime

from futurebucks.models import (
    Scenario,
    PersonConfig,
    IncomeSource,
    Asset,
    Expense,
    Assumptions,
    Distribution,
    MonteCarloConfig,
)
from futurebucks.monte_carlo import (
    SampledScenario,
    run_monte_carlo,
    run_simulation_with_samples,
)


@pytest.fixture
def basic_scenario():
    """Create a basic test scenario with distributions."""
    return Scenario(
        name="Test Monte Carlo",
        description="A test scenario for Monte Carlo",
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
                growth_rate={"mean": 0.03, "stddev": 0.02},
            ),
        ],
        assets=[
            Asset(
                name="Taxable Brokerage",
                type="taxable",
                balance=100000,
                expected_return={"mean": 0.07, "stddev": 0.15},
            ),
            Asset(
                name="401k",
                type="401k",
                balance=50000,
                expected_return={"mean": 0.07, "stddev": 0.15},
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
        assumptions=Assumptions(
            inflation_rate={"mean": 0.025, "stddev": 0.01},
        ),
    )


class TestDistribution:
    """Tests for Distribution type coercion."""

    def test_scalar_coercion(self):
        """Scalar values should be coerced to Distribution."""
        source = IncomeSource(
            name="Test",
            amount=50000,
            growth_rate=0.03,  # Scalar
        )
        assert isinstance(source.growth_rate, Distribution)
        assert source.growth_rate.mean == 0.03
        assert source.growth_rate.stddev == 0.0

    def test_dict_coercion(self):
        """Dict values should be coerced to Distribution."""
        source = IncomeSource(
            name="Test",
            amount=50000,
            growth_rate={"mean": 0.03, "stddev": 0.02},
        )
        assert isinstance(source.growth_rate, Distribution)
        assert source.growth_rate.mean == 0.03
        assert source.growth_rate.stddev == 0.02

    def test_distribution_passthrough(self):
        """Distribution objects should pass through unchanged."""
        dist = Distribution(mean=0.05, stddev=0.01)
        source = IncomeSource(
            name="Test",
            amount=50000,
            growth_rate=dist,
        )
        assert source.growth_rate.mean == 0.05
        assert source.growth_rate.stddev == 0.01

    def test_distribution_sampling_zero_stddev(self):
        """Distribution with zero stddev should return mean."""
        dist = Distribution(mean=0.07, stddev=0.0)
        rng = np.random.default_rng(42)

        samples = [dist.sample(rng) for _ in range(100)]
        assert all(s == 0.07 for s in samples)

    def test_distribution_sampling_with_stddev(self):
        """Distribution with stddev should sample from normal distribution."""
        dist = Distribution(mean=0.07, stddev=0.15)
        rng = np.random.default_rng(42)

        samples = [dist.sample(rng) for _ in range(1000)]

        # Should be approximately normal around the mean
        assert np.mean(samples) == pytest.approx(0.07, abs=0.02)
        assert np.std(samples) == pytest.approx(0.15, abs=0.02)


class TestSampledScenario:
    """Tests for SampledScenario class."""

    def test_sampled_scenario_creation(self, basic_scenario):
        """SampledScenario should create successfully."""
        rng = np.random.default_rng(42)
        sampled = SampledScenario(basic_scenario, rng)

        assert sampled.num_years == basic_scenario.simulation_years + 1

    def test_inflation_samples(self, basic_scenario):
        """Should sample inflation rates for each year."""
        rng = np.random.default_rng(42)
        sampled = SampledScenario(basic_scenario, rng)

        rates = [sampled.get_inflation_rate(i) for i in range(5)]

        # Should have variance
        assert len(set(rates)) > 1

        # Should be around the mean
        assert np.mean(rates) == pytest.approx(0.025, abs=0.02)

    def test_asset_return_samples(self, basic_scenario):
        """Should sample asset returns for each year."""
        rng = np.random.default_rng(42)
        sampled = SampledScenario(basic_scenario, rng)

        returns = [sampled.get_asset_return("Taxable Brokerage", i) for i in range(5)]

        # Should have variance
        assert len(set(returns)) > 1

        # Should be around the mean
        assert np.mean(returns) == pytest.approx(0.07, abs=0.1)

    def test_income_growth_samples(self, basic_scenario):
        """Should sample income growth for each year."""
        rng = np.random.default_rng(42)
        sampled = SampledScenario(basic_scenario, rng)

        growth = [sampled.get_income_growth("Primary Job", i) for i in range(5)]

        # Should have variance
        assert len(set(growth)) > 1


class TestMonteCarloSimulation:
    """Tests for Monte Carlo simulation."""

    def test_run_monte_carlo(self, basic_scenario):
        """Monte Carlo simulation should complete successfully."""
        config = MonteCarloConfig(num_simulations=10, seed=42)
        result = run_monte_carlo(basic_scenario, config)

        assert result.num_simulations == 10
        assert result.scenario_name == "Test Monte Carlo"

    def test_monte_carlo_snapshots(self, basic_scenario):
        """Should produce correct number of snapshots."""
        config = MonteCarloConfig(num_simulations=10, seed=42)
        result = run_monte_carlo(basic_scenario, config)

        assert len(result.snapshots) == basic_scenario.simulation_years + 1

    def test_monte_carlo_has_variance(self, basic_scenario):
        """Monte Carlo results should show variance across simulations."""
        config = MonteCarloConfig(num_simulations=50, seed=42)
        result = run_monte_carlo(basic_scenario, config)

        final_snapshot = result.snapshots[-1]

        # Percentiles should be different
        assert final_snapshot.net_worth_p10 < final_snapshot.net_worth_p50
        assert final_snapshot.net_worth_p50 < final_snapshot.net_worth_p90

        # Should have non-zero standard deviation
        assert final_snapshot.net_worth_stddev > 0

    def test_monte_carlo_reproducibility(self, basic_scenario):
        """Same seed should produce same results."""
        config1 = MonteCarloConfig(num_simulations=10, seed=42)
        config2 = MonteCarloConfig(num_simulations=10, seed=42)

        result1 = run_monte_carlo(basic_scenario, config1)
        result2 = run_monte_carlo(basic_scenario, config2)

        assert result1.snapshots[-1].net_worth_p50 == result2.snapshots[-1].net_worth_p50

    def test_different_seeds_different_results(self, basic_scenario):
        """Different seeds should produce different results."""
        config1 = MonteCarloConfig(num_simulations=20, seed=42)
        config2 = MonteCarloConfig(num_simulations=20, seed=123)

        result1 = run_monte_carlo(basic_scenario, config1)
        result2 = run_monte_carlo(basic_scenario, config2)

        # Medians should differ (though could theoretically be the same by chance)
        assert result1.snapshots[-1].net_worth_p50 != result2.snapshots[-1].net_worth_p50

    def test_milestone_probabilities(self, basic_scenario):
        """Should calculate milestone probabilities."""
        config = MonteCarloConfig(num_simulations=20, seed=42)
        result = run_monte_carlo(basic_scenario, config)

        assert len(result.milestone_probabilities) > 0

        for prob in result.milestone_probabilities:
            assert 0 <= prob.probability <= 1

    def test_final_net_worth_distribution(self, basic_scenario):
        """Should store final net worth distribution."""
        config = MonteCarloConfig(num_simulations=20, seed=42)
        result = run_monte_carlo(basic_scenario, config)

        assert len(result.final_net_worth_distribution) == 20


class TestBackwardCompatibility:
    """Tests for backward compatibility with scalar values."""

    def test_scalar_values_still_work(self):
        """Scenarios with scalar values should still work and get default volatility."""
        scenario = Scenario(
            name="Scalar Test",
            description="Test with scalar values",
            person=PersonConfig(
                name="Test User",
                birth_year=1990,
                retirement_age=65,
                life_expectancy=41,  # age 36 + 5 years of simulation
            ),
            income_sources=[
                IncomeSource(
                    name="Job",
                    amount=80000,
                    growth_rate=0.03,  # Scalar - will get default 2% stddev
                ),
            ],
            assets=[
                Asset(
                    name="Savings",
                    type="taxable",
                    balance=10000,
                    expected_return=0.05,  # Scalar - will get market_return_stddev
                ),
            ],
            expenses=[
                Expense(
                    name="Expenses",
                    amount=40000,
                    inflation_adjusted=True,
                ),
            ],
            life_events=[],
            assumptions=Assumptions(
                inflation_rate=0.025,  # Scalar - will get default 1% stddev
            ),
        )

        config = MonteCarloConfig(num_simulations=10, seed=42)
        result = run_monte_carlo(scenario, config)

        assert result.num_simulations == 10

        # Scalar values now get default volatility, so there should be variance
        final_snapshot = result.snapshots[-1]
        assert final_snapshot.net_worth_p10 != final_snapshot.net_worth_p90
        assert final_snapshot.net_worth_stddev > 0

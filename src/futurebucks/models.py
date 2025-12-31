"""Pydantic models for financial simulation configuration and state."""

from typing import Annotated, Literal, Union
import numpy as np
from pydantic import BaseModel, Field, BeforeValidator


class Distribution(BaseModel):
    """A distribution for stochastic parameters.

    Supports multiple distribution types:
    - normal: Standard normal distribution (default)
    - lognormal: Lognormal distribution (good for returns, always positive)
    - uniform: Uniform distribution between min and max
    """

    type: Literal["normal", "lognormal", "uniform"] = "normal"
    mean: float
    stddev: float = Field(default=0.0, ge=0.0)
    min: float | None = Field(default=None, description="Minimum value (for uniform)")
    max: float | None = Field(default=None, description="Maximum value (for uniform)")

    def sample(self, rng: np.random.Generator) -> float:
        """Sample a value from this distribution."""
        if self.stddev == 0 and self.type != "uniform":
            return self.mean

        if self.type == "normal":
            return float(rng.normal(self.mean, self.stddev))
        elif self.type == "lognormal":
            # Convert mean/stddev to lognormal parameters
            # For lognormal, we want the actual mean to be self.mean
            # mu = ln(mean^2 / sqrt(var + mean^2))
            # sigma = sqrt(ln(1 + var/mean^2))
            if self.mean <= 0:
                return self.mean  # Can't do lognormal with non-positive mean
            var = self.stddev ** 2
            mu = np.log(self.mean**2 / np.sqrt(var + self.mean**2))
            sigma = np.sqrt(np.log(1 + var / self.mean**2))
            return float(rng.lognormal(mu, sigma))
        elif self.type == "uniform":
            low = self.min if self.min is not None else (self.mean - self.stddev * 1.732)
            high = self.max if self.max is not None else (self.mean + self.stddev * 1.732)
            return float(rng.uniform(low, high))
        else:
            return self.mean

    def __float__(self) -> float:
        """Return the mean for backward compatibility with float operations."""
        return self.mean


def _coerce_to_distribution(v: Union[float, int, dict, Distribution]) -> Distribution:
    """Validator to convert scalar or dict to Distribution."""
    if isinstance(v, Distribution):
        return v
    if isinstance(v, (int, float)):
        return Distribution(mean=float(v), stddev=0.0)
    if isinstance(v, dict):
        return Distribution(**v)
    raise ValueError(f"Cannot convert {type(v)} to Distribution")


# Type alias for fields that can be either float or Distribution
StochasticFloat = Annotated[Distribution, BeforeValidator(_coerce_to_distribution)]


class PersonConfig(BaseModel):
    """Configuration for a person in the simulation."""

    name: str
    birth_year: int
    retirement_age: int = 65


class IncomeSource(BaseModel):
    """An income source (job, rental income, etc.)."""

    name: str
    amount: float = Field(description="Annual income amount")
    growth_rate: StochasticFloat = Field(
        default_factory=lambda: Distribution(mean=0.03, stddev=0.0),
        description="Annual raise percentage (can be distribution)",
    )
    start_year: int | None = None
    end_year: int | None = None


class Asset(BaseModel):
    """A financial asset (investment account, property, etc.)."""

    name: str
    type: Literal["taxable", "401k", "roth_ira", "traditional_ira", "hsa", "real_estate", "other"]
    balance: float
    expected_return: StochasticFloat = Field(
        default_factory=lambda: Distribution(mean=0.07, stddev=0.0),
        description="Expected annual return (can be distribution)",
    )
    contribution_limit: float | None = Field(
        default=None, description="Annual contribution limit (None for unlimited)"
    )


class Expense(BaseModel):
    """A recurring expense."""

    name: str
    amount: float = Field(description="Annual expense amount")
    inflation_adjusted: bool = True
    start_year: int | None = None
    end_year: int | None = None


class LifeEvent(BaseModel):
    """A life event that affects finances."""

    name: str
    year: int
    type: Literal["income_change", "expense_change", "windfall", "asset_purchase", "child", "retirement"]
    details: dict = Field(default_factory=dict, description="Event-specific details")


class Assumptions(BaseModel):
    """Economic and tax assumptions for the simulation."""

    inflation_rate: StochasticFloat = Field(
        default_factory=lambda: Distribution(mean=0.025, stddev=0.0),
        description="Annual inflation rate (can be distribution)",
    )
    market_return_mean: float = 0.07  # Kept for reference/default
    market_return_stddev: float = 0.15  # Kept for reference/default
    state: str = "NC"
    state_tax_rate: float = 0.0525
    federal_standard_deduction: float = 14600  # 2024 single
    contribution_401k_limit: float = 23000  # 2024
    contribution_401k_catchup: float = 7500  # age 50+
    contribution_ira_limit: float = 7000  # 2024
    contribution_ira_catchup: float = 1000  # age 50+
    contribution_hsa_limit: float = 4150  # 2024 single
    # FIRE calculation settings
    fire_expense_basis: Literal["max", "first_year", "current", "target"] = Field(
        default="max",
        description="How to calculate FIRE target: 'max' (highest expenses in simulation), "
        "'first_year' (starting expenses), 'current' (each year's expenses), "
        "'target' (use fire_target_expenses value)",
    )
    fire_target_expenses: float | None = Field(
        default=None,
        description="Target annual expenses for FIRE calculation (only used if fire_expense_basis='target')",
    )


class MonteCarloConfig(BaseModel):
    """Configuration for Monte Carlo simulation."""

    enabled: bool = False
    num_simulations: int = Field(default=1000, ge=10, le=10000)
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    percentiles: list[int] = Field(default=[10, 25, 50, 75, 90])


class Scenario(BaseModel):
    """A complete financial scenario configuration."""

    name: str
    description: str = ""
    person: PersonConfig
    income_sources: list[IncomeSource] = Field(default_factory=list)
    assets: list[Asset] = Field(default_factory=list)
    expenses: list[Expense] = Field(default_factory=list)
    life_events: list[LifeEvent] = Field(default_factory=list)
    assumptions: Assumptions = Field(default_factory=Assumptions)
    simulation_years: int = 30
    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)


class YearlySnapshot(BaseModel):
    """A snapshot of financial state for a single year."""

    year: int
    age: int
    gross_income: float
    federal_tax: float
    state_tax: float
    fica_tax: float
    total_tax: float
    net_income: float
    total_expenses: float
    savings: float
    assets: dict[str, float] = Field(description="Asset name -> balance")
    net_worth: float
    cumulative_taxes_paid: float = 0.0


class SimulationResult(BaseModel):
    """Complete results from a simulation run."""

    scenario_name: str
    snapshots: list[YearlySnapshot]
    milestones: dict[str, int | None] = Field(
        default_factory=dict,
        description="Milestone name -> year achieved (e.g., 'net_worth_1m': 2028)",
    )
    fire_target_expenses: float = Field(
        default=0.0,
        description="The annual expenses used for FIRE calculation (based on fire_expense_basis)",
    )


class MonteCarloSnapshot(BaseModel):
    """Aggregated statistics for a single year across all Monte Carlo runs."""

    year: int
    age: int
    # Net worth percentiles
    net_worth_p10: float
    net_worth_p25: float
    net_worth_p50: float  # median
    net_worth_p75: float
    net_worth_p90: float
    net_worth_mean: float
    net_worth_stddev: float
    # Other aggregated metrics
    gross_income_median: float
    total_expenses_median: float
    savings_median: float


class MilestoneProbability(BaseModel):
    """Probability of reaching a milestone across Monte Carlo runs."""

    milestone: str
    target_value: float | None = None
    probability: float  # 0.0 to 1.0
    median_year: int | None = None  # Year when 50% of runs achieve it
    p10_year: int | None = None  # Year when 10% of runs achieve it
    p90_year: int | None = None  # Year when 90% of runs achieve it


class MonteCarloResult(BaseModel):
    """Results from Monte Carlo simulation."""

    scenario_name: str
    num_simulations: int
    snapshots: list[MonteCarloSnapshot]
    milestone_probabilities: list[MilestoneProbability]
    final_net_worth_distribution: list[float]  # All final net worth values

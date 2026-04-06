"""Pydantic models for financial simulation configuration and state."""

from typing import Annotated, Literal, Union
import logging
import numpy as np
from pydantic import BaseModel, Field, BeforeValidator, model_validator

logger = logging.getLogger(__name__)


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
    life_expectancy: int = Field(
        default=95,
        description="Expected lifespan for retirement planning (used by presentation layer, "
        "not the core simulation which runs to IRS retirement age)",
    )


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
    inflation_category: str = Field(
        default="general",
        description="Inflation category: 'general', 'healthcare', 'education', etc.",
    )
    start_year: int | None = None
    end_year: int | None = None


class LifeEvent(BaseModel):
    """A life event that affects finances."""

    name: str
    year: int
    type: Literal["income_change", "expense_change", "windfall", "asset_purchase", "child", "retirement"]
    details: dict = Field(default_factory=dict, description="Event-specific details")


class CategoryInflationRates(BaseModel):
    """Category-specific inflation rates.

    Different expense categories often inflate at different rates:
    - Healthcare typically inflates at 5-6% annually (faster than general)
    - Education inflates at 4-5% annually
    - General goods/services at 2-3%
    """

    general: StochasticFloat = Field(
        default_factory=lambda: Distribution(mean=0.025, stddev=0.01),
        description="General inflation rate (2.5% historical average)",
    )
    healthcare: StochasticFloat = Field(
        default_factory=lambda: Distribution(mean=0.055, stddev=0.015),
        description="Healthcare inflation (historically 5-6% annually)",
    )
    education: StochasticFloat = Field(
        default_factory=lambda: Distribution(mean=0.04, stddev=0.01),
        description="Education cost inflation (historically 4-5%)",
    )
    housing: StochasticFloat = Field(
        default_factory=lambda: Distribution(mean=0.03, stddev=0.015),
        description="Housing cost inflation",
    )

    def get_rate(self, category: str) -> Distribution:
        """Get inflation rate for a category, defaulting to general if not found."""
        if hasattr(self, category):
            return getattr(self, category)
        logger.warning(f"Unknown inflation category '{category}', using general rate")
        return self.general


class Assumptions(BaseModel):
    """Economic and tax assumptions for the simulation."""

    model_config = {"extra": "ignore"}  # Ignore legacy YAML fields

    # Support both old single inflation_rate and new category-specific rates
    inflation_rate: StochasticFloat | None = Field(
        default=None,
        description="DEPRECATED: Use inflation_rates instead. Kept for backward compatibility.",
    )
    inflation_rates: CategoryInflationRates = Field(
        default_factory=CategoryInflationRates,
        description="Category-specific inflation rates",
    )
    market_return_mean: float = 0.07  # Kept for reference/default
    market_return_stddev: float = 0.15  # Kept for reference/default
    state: str = "NC"
    state_tax_rate: float = 0.0525
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
    # IRS retirement age settings (for penalty-free retirement account access)
    irs_retirement_age: float = Field(
        default=59.5,
        description="Current IRS retirement age for penalty-free 401k/IRA withdrawals",
    )
    irs_retirement_age_increase_per_decade: float = Field(
        default=1.0,
        description="Expected increase in IRS retirement age per decade (e.g., 1.0 means 59.5 -> 60.5 over 10 years)",
    )
    # Cash buffer strategy
    cash_buffer_months: float = Field(
        default=2.0,
        ge=0.0,
        description="Target cash buffer as months of expenses. High-risk tolerance = 1-2 months, conservative = 3-6 months.",
    )
    # Monte Carlo mode options
    use_conservative_defaults: bool = Field(
        default=False,
        description="If True, reduces return means by 1% and increases stddevs by 20% for stress testing",
    )

    @model_validator(mode="after")
    def handle_legacy_inflation_rate(self) -> "Assumptions":
        """Convert legacy single inflation_rate to category-specific rates."""
        if self.inflation_rate is not None:
            # Use the legacy rate as the general rate
            self.inflation_rates = CategoryInflationRates(
                general=self.inflation_rate,
                healthcare=Distribution(mean=0.055, stddev=0.015),  # Default healthcare
                education=Distribution(mean=0.04, stddev=0.01),  # Default education
                housing=Distribution(mean=0.03, stddev=0.015),  # Default housing
            )
            logger.info("Converted legacy inflation_rate to inflation_rates.general")
        return self

    def get_inflation_rate(self, category: str = "general") -> Distribution:
        """Get inflation rate for a category."""
        return self.inflation_rates.get_rate(category)

    def get_irs_retirement_age(self, years_from_now: int = 0) -> float:
        """Get the projected IRS retirement age for a future year.

        Args:
            years_from_now: Number of years in the future

        Returns:
            Projected IRS retirement age (increases over time)
        """
        decades = years_from_now / 10.0
        return self.irs_retirement_age + (decades * self.irs_retirement_age_increase_per_decade)

    def get_years_until_irs_access(self, current_age: float) -> float:
        """Calculate years until a person can access retirement accounts.

        The IRS retirement age increases over time, so we need to find when
        the person's age will catch up to the moving target.

        Math: Person is eligible when age + t = irs_age + (t/10) * increase_per_decade
        Solving: t = (irs_age - current_age) / (1 - increase_per_decade/10)

        Args:
            current_age: Person's current age

        Returns:
            Years until IRS retirement age eligibility (0 if already eligible)
        """
        if current_age >= self.irs_retirement_age:
            return 0.0

        # Rate at which IRS age increases per year
        increase_per_year = self.irs_retirement_age_increase_per_decade / 10.0

        # If IRS age increases >= 1 year per year, person never catches up
        if increase_per_year >= 1.0:
            return float("inf")

        # Solve for t: current_age + t = irs_age + t * increase_per_year
        # t * (1 - increase_per_year) = irs_age - current_age
        # t = (irs_age - current_age) / (1 - increase_per_year)
        years_until = (self.irs_retirement_age - current_age) / (1 - increase_per_year)

        return max(0.0, years_until)


class HealthcareConfig(BaseModel):
    """Healthcare cost configuration for pre-Medicare years.

    Models healthcare expenses from current age until Medicare eligibility (age 65).
    Supports transition from employer-subsidized coverage to marketplace/COBRA.

    Annual cost calculation:
    - During employer coverage: employer_monthly_cost * 12
    - After employer coverage: (monthly_premium * 12) + annual_deductible +
                               (annual_out_of_pocket_max * expected_utilization)
    """

    coverage_type: Literal["individual", "family"] = Field(
        default="family",
        description="Type of healthcare coverage",
    )
    # Employer-subsidized period
    employer_subsidized_until: int | None = Field(
        default=None,
        description="Year when employer coverage ends (None if no employer coverage)",
    )
    employer_monthly_cost: float = Field(
        default=400.0,
        description="Monthly cost while employer-subsidized",
    )
    # Post-employer coverage (ACA marketplace, COBRA, etc.)
    monthly_premium: float = Field(
        default=1500.0,
        description="Monthly premium for marketplace/COBRA coverage",
    )
    annual_deductible: float = Field(
        default=6000.0,
        description="Annual deductible amount",
    )
    annual_out_of_pocket_max: float = Field(
        default=12000.0,
        description="Annual out-of-pocket maximum",
    )
    expected_utilization: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Expected utilization as fraction of OOP max (0.0-1.0)",
    )

    def calculate_annual_cost(self, year: int, person_birth_year: int) -> float:
        """Calculate annual healthcare cost for a given year.

        Args:
            year: The simulation year
            person_birth_year: Birth year of the person (to calculate age)

        Returns:
            Annual healthcare cost, or 0 if age >= 65 (Medicare eligible)
        """
        age = year - person_birth_year

        # No healthcare modeling after Medicare eligibility
        if age >= 65:
            return 0.0

        # Check if still in employer-subsidized period
        if self.employer_subsidized_until and year < self.employer_subsidized_until:
            return self.employer_monthly_cost * 12

        # Full marketplace/COBRA cost
        return (
            (self.monthly_premium * 12)
            + self.annual_deductible
            + (self.annual_out_of_pocket_max * self.expected_utilization)
        )


# Default volatilities for Monte Carlo simulation
# Based on historical data and financial planning best practices
DEFAULT_VOLATILITIES = {
    # Asset types - based on historical volatility of underlying asset classes
    "401k": 0.15,           # Equity-heavy retirement accounts (S&P 500 ~15% annual vol)
    "roth_ira": 0.15,       # Similar to 401k
    "traditional_ira": 0.15,
    "taxable": 0.15,        # Typically equity-heavy
    "hsa": 0.12,            # Often more conservative allocation
    "real_estate": 0.08,    # Less volatile but illiquid (Case-Shiller ~5-10%)
    "other": 0.10,          # Mixed/unknown assets

    # Income volatility - job raises are somewhat predictable
    "income_growth": 0.03,  # Based on variance in annual raise distributions

    # Inflation volatility - based on historical CPI variation
    "inflation_general": 0.01,      # General CPI volatility
    "inflation_healthcare": 0.015,  # Healthcare CPI more volatile
    "inflation_education": 0.01,
    "inflation_housing": 0.015,
}


class MonteCarloConfig(BaseModel):
    """Configuration for Monte Carlo simulation."""

    enabled: bool = False
    num_simulations: int = Field(
        default=2000,  # Increased from 1000 for better tail coverage
        ge=10,
        le=100000,
        description="Number of simulations (2000 default, 10000+ for detailed analysis)",
    )
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    percentiles: list[int] = Field(default=[10, 25, 50, 75, 90])
    detailed_analysis_mode: bool = Field(
        default=False,
        description="If True, runs 10000 simulations for better statistical coverage",
    )

    @property
    def effective_num_simulations(self) -> int:
        """Get actual number of simulations to run."""
        if self.detailed_analysis_mode:
            return 10000
        return self.num_simulations


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
    healthcare: HealthcareConfig | None = Field(
        default=None,
        description="Healthcare cost configuration (pre-Medicare)",
    )
    start_year: int = Field(
        default=2026,
        description="Year to start the simulation (e.g., 2026)",
    )
    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)

    @property
    def simulation_years(self) -> int:
        """Compute simulation years from start_year to IRS retirement age.

        The simulation runs until the person can access retirement accounts
        penalty-free (IRS retirement age, which increases over time).
        """
        current_age = self.start_year - self.person.birth_year
        years_until_irs = self.assumptions.get_years_until_irs_access(float(current_age))
        return int(years_until_irs)


class YearlySnapshot(BaseModel):
    """A snapshot of financial state for a single year."""

    year: int
    age: int
    gross_income: float
    federal_tax: float
    state_tax: float
    fica_tax: float
    capital_gains_tax: float = Field(
        default=0.0,
        description="Capital gains tax from selling taxable investments to cover expenses",
    )
    total_tax: float
    net_income: float
    total_expenses: float
    savings: float
    assets_sold: float = Field(
        default=0.0,
        description="Total taxable assets sold this year to cover expenses",
    )
    assets: dict[str, float] = Field(description="Asset name -> balance")
    net_worth: float
    accessible_net_worth: float = Field(
        default=0.0,
        description="Net worth excluding retirement accounts (401k, IRA, HSA) - accessible before IRS retirement age",
    )
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
    probability: float  # 0.0 to 1.0 - probability of ever reaching during simulation
    probability_by_target: float | None = None  # Probability of reaching by target year
    target_year: int | None = None  # The target year (e.g., retirement year)
    target_age: int | None = None  # The target age for context
    median_year: int | None = None  # Year when 50% of runs achieve it
    p10_year: int | None = None  # Year when 10% of runs achieve it (early achievers)
    p90_year: int | None = None  # Year when 90% of runs achieve it (late achievers)


class MonteCarloResult(BaseModel):
    """Results from Monte Carlo simulation."""

    scenario_name: str
    num_simulations: int
    snapshots: list[MonteCarloSnapshot]
    milestone_probabilities: list[MilestoneProbability]
    final_net_worth_distribution: list[float]  # All final net worth values

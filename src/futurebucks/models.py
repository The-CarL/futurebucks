"""Pydantic models for financial simulation configuration and state."""

from typing import Literal
from pydantic import BaseModel, Field


class PersonConfig(BaseModel):
    """Configuration for a person in the simulation."""

    name: str
    birth_year: int
    retirement_age: int = 65


class IncomeSource(BaseModel):
    """An income source (job, rental income, etc.)."""

    name: str
    amount: float = Field(description="Annual income amount")
    growth_rate: float = Field(default=0.03, description="Annual raise percentage")
    start_year: int | None = None
    end_year: int | None = None


class Asset(BaseModel):
    """A financial asset (investment account, property, etc.)."""

    name: str
    type: Literal["taxable", "401k", "roth_ira", "traditional_ira", "hsa", "real_estate", "other"]
    balance: float
    expected_return: float = Field(default=0.07, description="Expected annual return")
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

    inflation_rate: float = 0.025
    market_return_mean: float = 0.07
    market_return_stddev: float = 0.15
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

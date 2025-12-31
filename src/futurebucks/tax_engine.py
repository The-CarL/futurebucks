"""Tax calculation engine for federal, state, and FICA taxes."""

from dataclasses import dataclass

# 2024 Federal Tax Brackets (Married Filing Jointly)
FEDERAL_BRACKETS_2024_MFJ = [
    (23200, 0.10),
    (94300, 0.12),
    (201050, 0.22),
    (383900, 0.24),
    (487450, 0.32),
    (731200, 0.35),
    (float("inf"), 0.37),
]

# Base values for 2024 (used for inflation adjustment)
BASE_YEAR = 2024
BASE_STANDARD_DEDUCTION_MFJ = 29200
BASE_SS_WAGE_CAP = 168600
BASE_401K_LIMIT = 23000
BASE_401K_CATCHUP = 7500
BASE_IRA_LIMIT = 7000
BASE_IRA_CATCHUP = 1000
BASE_HSA_LIMIT_FAMILY = 8300
BASE_ROTH_IRA_INCOME_LIMIT_START = 230000  # MFJ phase-out starts
BASE_ROTH_IRA_INCOME_LIMIT_END = 240000    # MFJ fully phased out

# FICA rates (these don't change with inflation)
SOCIAL_SECURITY_RATE = 0.062
MEDICARE_RATE = 0.0145
MEDICARE_ADDITIONAL_RATE = 0.009
MEDICARE_ADDITIONAL_THRESHOLD = 200000


def inflate_value(base_value: float, inflation_rate: float, years: int) -> float:
    """Apply inflation to a base value."""
    return base_value * ((1 + inflation_rate) ** years)


def inflate_brackets(
    brackets: list[tuple[float, float]],
    inflation_rate: float,
    years: int,
) -> list[tuple[float, float]]:
    """Apply inflation to tax bracket thresholds."""
    multiplier = (1 + inflation_rate) ** years
    return [
        (limit * multiplier if limit != float("inf") else float("inf"), rate)
        for limit, rate in brackets
    ]


@dataclass
class TaxBreakdown:
    """Breakdown of tax liability."""

    federal_tax: float
    state_tax: float
    fica_tax: float
    total_tax: float
    effective_rate: float


def calculate_federal_tax(
    taxable_income: float,
    brackets: list[tuple[float, float]] | None = None,
) -> float:
    """Calculate federal income tax using progressive brackets.

    Args:
        taxable_income: Income after deductions
        brackets: List of (upper_limit, rate) tuples. Uses 2024 MFJ brackets if None.

    Returns:
        Federal tax owed
    """
    if brackets is None:
        brackets = FEDERAL_BRACKETS_2024_MFJ

    if taxable_income <= 0:
        return 0.0

    tax = 0.0
    prev_limit = 0.0

    for upper_limit, rate in brackets:
        if taxable_income <= prev_limit:
            break

        bracket_income = min(taxable_income, upper_limit) - prev_limit
        tax += bracket_income * rate
        prev_limit = upper_limit

    return tax


def calculate_state_tax(taxable_income: float, state_rate: float) -> float:
    """Calculate state income tax using flat rate.

    Args:
        taxable_income: Income after deductions
        state_rate: Flat state tax rate (e.g., 0.0525 for NC)

    Returns:
        State tax owed
    """
    if taxable_income <= 0:
        return 0.0
    return taxable_income * state_rate


def calculate_fica_tax(
    gross_income: float,
    ss_wage_cap: float | None = None,
) -> float:
    """Calculate FICA taxes (Social Security + Medicare).

    Args:
        gross_income: Gross employment income
        ss_wage_cap: Social Security wage cap (uses base 2024 value if None)

    Returns:
        Total FICA tax owed
    """
    if gross_income <= 0:
        return 0.0

    if ss_wage_cap is None:
        ss_wage_cap = BASE_SS_WAGE_CAP

    # Social Security (capped)
    ss_income = min(gross_income, ss_wage_cap)
    social_security = ss_income * SOCIAL_SECURITY_RATE

    # Medicare (no cap, but additional tax above threshold)
    medicare = gross_income * MEDICARE_RATE
    if gross_income > MEDICARE_ADDITIONAL_THRESHOLD:
        medicare += (gross_income - MEDICARE_ADDITIONAL_THRESHOLD) * MEDICARE_ADDITIONAL_RATE

    return social_security + medicare


def calculate_taxes(
    gross_income: float,
    pre_tax_deductions: float = 0.0,
    standard_deduction: float = 29200,
    state_rate: float = 0.0525,
    federal_brackets: list[tuple[float, float]] | None = None,
    ss_wage_cap: float | None = None,
) -> TaxBreakdown:
    """Calculate complete tax breakdown.

    Args:
        gross_income: Total gross income
        pre_tax_deductions: 401k, HSA, and other pre-tax deductions
        standard_deduction: Federal standard deduction
        state_rate: State income tax rate
        federal_brackets: Federal tax brackets (uses 2024 MFJ if None)
        ss_wage_cap: Social Security wage cap (uses base 2024 value if None)

    Returns:
        TaxBreakdown with all tax components
    """
    # Taxable income for federal/state
    taxable_income = max(0, gross_income - pre_tax_deductions - standard_deduction)

    # State typically uses similar taxable income (simplified)
    state_taxable = max(0, gross_income - pre_tax_deductions - standard_deduction)

    federal = calculate_federal_tax(taxable_income, federal_brackets)
    state = calculate_state_tax(state_taxable, state_rate)
    fica = calculate_fica_tax(gross_income, ss_wage_cap)

    total = federal + state + fica
    effective_rate = total / gross_income if gross_income > 0 else 0.0

    return TaxBreakdown(
        federal_tax=federal,
        state_tax=state,
        fica_tax=fica,
        total_tax=total,
        effective_rate=effective_rate,
    )


def calculate_capital_gains_tax(
    gains: float,
    ordinary_income: float,
    is_long_term: bool = True,
) -> float:
    """Calculate capital gains tax.

    Args:
        gains: Capital gains amount
        ordinary_income: Ordinary income (affects bracket)
        is_long_term: Whether gains are long-term (held > 1 year)

    Returns:
        Capital gains tax owed
    """
    if gains <= 0:
        return 0.0

    if not is_long_term:
        # Short-term gains taxed as ordinary income
        return calculate_federal_tax(ordinary_income + gains) - calculate_federal_tax(ordinary_income)

    # 2024 Long-term capital gains brackets (single)
    total_income = ordinary_income + gains

    if total_income <= 47025:
        return 0.0
    elif total_income <= 518900:
        # 15% bracket
        taxable_at_15 = min(gains, total_income - 47025)
        return taxable_at_15 * 0.15
    else:
        # 20% bracket (simplified)
        taxable_at_15 = max(0, 518900 - 47025 - ordinary_income)
        taxable_at_20 = gains - taxable_at_15
        return taxable_at_15 * 0.15 + taxable_at_20 * 0.20

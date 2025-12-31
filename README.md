# FutureBucks

A life-event-driven financial simulation tool. Model your financial future with year-by-year projections, tax calculations, and scenario analysis.

## Features

- **Year-by-year simulation** - Project net worth, income, expenses, and taxes over 30+ years
- **Life event modeling** - Job changes, children, home purchases, windfalls, retirement
- **Tax engine** - Federal brackets (MFJ), state tax, FICA, Social Security cap
- **Inflation adjustment** - Tax brackets, contribution limits, and expenses adjust for inflation
- **Retirement account logic** - 401k, Roth IRA (with income limits), HSA, Traditional IRA
- **Roth IRA income phase-out** - Automatically blocks contributions when income exceeds limits
- **Scenario comparison** - Compare multiple what-if scenarios side by side
- **FIRE tracking** - Track progress toward financial independence (4% rule)
- **Interactive UI** - Streamlit-based dashboard with Plotly charts

## Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/futurebucks.git
cd futurebucks

# Install dependencies (requires Python 3.11+ and uv)
uv sync

# Run the app
uv run streamlit run src/futurebucks/app.py
```

## Creating Scenarios

Scenarios are defined in YAML files in the `scenarios/` directory. Here's a minimal example:

```yaml
name: "My Scenario"
description: "A simple financial projection"

person:
  name: "Your Name"
  birth_year: 1990
  retirement_age: 60

income_sources:
  - name: "Primary Job"
    amount: 150000
    growth_rate: 0.03

assets:
  - name: "401k"
    type: "401k"
    balance: 100000
    expected_return: 0.07
  - name: "Taxable Brokerage"
    type: "taxable"
    balance: 50000
    expected_return: 0.07

expenses:
  - name: "Living Expenses"
    amount: 60000
    inflation_adjusted: true

life_events:
  - name: "First child"
    year: 2028
    type: "child"
    details:
      name: "Child expenses"
      annual_cost: 20000
      end_year: 2046

assumptions:
  inflation_rate: 0.025
  market_return_mean: 0.07
  state_tax_rate: 0.05

simulation_years: 30
```

## Life Event Types

| Type | Description | Details |
|------|-------------|---------|
| `income_change` | Job change, raise, layoff | `source_name`, `new_amount`, `growth_rate` |
| `expense_change` | New or modified expense | `expense_name`, `new_amount`, `inflation_adjusted` |
| `windfall` | One-time cash event | `amount`, `target_asset` |
| `asset_purchase` | Buy property/asset | `asset_name`, `cost`, `asset_value`, `source_asset` |
| `child` | New child with expenses | `name`, `annual_cost`, `end_year` |
| `retirement` | End all income sources | - |

## Asset Types

| Type | Tax Treatment |
|------|---------------|
| `taxable` | Regular brokerage, taxed on gains |
| `401k` | Pre-tax contributions, taxed on withdrawal |
| `roth_ira` | Post-tax, tax-free growth (income limits apply) |
| `traditional_ira` | Pre-tax contributions |
| `hsa` | Triple tax-advantaged |
| `real_estate` | Home equity |
| `other` | Vehicles, collectibles, etc. |

## Inflation-Adjusted Values

The simulation automatically adjusts for inflation (based on `assumptions.inflation_rate`):

- Federal tax brackets
- Standard deduction
- 401k contribution limits ($23,000 base in 2024)
- IRA contribution limits ($7,000 base in 2024)
- HSA contribution limits ($8,300 base in 2024)
- Social Security wage cap ($168,600 base in 2024)
- Roth IRA income phase-out limits ($230K-$240K MFJ in 2024)
- Any expense marked `inflation_adjusted: true`

## Roth IRA Income Limits

The simulation enforces Roth IRA income limits (Married Filing Jointly):

- **Phase-out starts**: $230,000 MAGI (2024, inflation-adjusted)
- **Fully phased out**: $240,000 MAGI (2024, inflation-adjusted)

If your income exceeds these limits, Roth IRA contributions are automatically blocked and redirected to taxable brokerage.

## FIRE Calculation

FIRE (Financial Independence, Retire Early) is calculated when your net worth × 4% covers your target expenses.

The expense basis is configurable via `assumptions.fire_expense_basis`:

| Value | Description |
|-------|-------------|
| `max` (default) | Uses maximum expenses across all simulation years |
| `first_year` | Uses the first year's expenses as baseline |
| `current` | Uses each year's actual expenses (original behavior) |
| `target` | Uses `fire_target_expenses` value you specify |

Example configuration:

```yaml
assumptions:
  fire_expense_basis: "target"
  fire_target_expenses: 120000  # Retire on $120K/year
```

Using `max` (default) prevents misleading results where crisis-level expense cuts would show early FIRE.

## Sample Scenarios

The `samples/` directory includes example scenarios to get you started:

| Scenario | Description |
|----------|-------------|
| `example_base.yaml` | Dual-income household with career growth and two children |
| `example_bull_market.yaml` | Optimistic scenario with strong market returns |
| `example_bear_market.yaml` | Pessimistic scenario with weak market performance |

To use a sample, copy it to `scenarios/` and customize:

```bash
cp samples/example_base.yaml scenarios/my_scenario.yaml
```

Then edit the file with your own financial details.

**Note:** The `scenarios/` directory is gitignored to protect your personal financial data. Only files in `samples/` are committed to the repository.

## Project Structure

```
futurebucks/
├── src/futurebucks/
│   ├── app.py           # Streamlit UI
│   ├── models.py        # Pydantic data models
│   ├── simulation.py    # Core simulation engine
│   ├── tax_engine.py    # Tax calculations
│   ├── scenarios.py     # YAML loading/saving
│   └── charts.py        # Plotly visualizations
├── samples/             # Example scenario files (committed)
├── scenarios/           # Your personal scenarios (gitignored)
├── tests/               # Test suite
└── pyproject.toml       # Project config
```

## Running Tests

```bash
uv run pytest -v
```

## Limitations

This tool is for **educational and planning purposes only**. It does not constitute financial or tax advice.

Known simplifications:
- Uses flat state tax rates (no state-specific brackets)
- Does not model required minimum distributions (RMDs)
- Does not model Social Security benefits
- Does not model backdoor Roth conversions
- Capital gains are not explicitly tracked year-over-year
- Does not model estate taxes or inheritance

## License

MIT

## Contributing

Contributions welcome! Please open an issue or PR.

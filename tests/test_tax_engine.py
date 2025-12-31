"""Tests for the tax engine module."""

import pytest
from futurebucks.tax_engine import (
    calculate_federal_tax,
    calculate_state_tax,
    calculate_fica_tax,
    calculate_taxes,
    calculate_capital_gains_tax,
    BASE_SS_WAGE_CAP,
)


class TestFederalTax:
    """Tests for federal tax calculations."""

    def test_zero_income(self):
        """Zero income should result in zero tax."""
        assert calculate_federal_tax(0) == 0

    def test_negative_income(self):
        """Negative income should result in zero tax."""
        assert calculate_federal_tax(-1000) == 0

    def test_first_bracket(self):
        """Income in first bracket should be taxed at 10%."""
        # $10,000 income should be taxed at 10%
        tax = calculate_federal_tax(10000)
        assert tax == pytest.approx(1000, rel=0.01)

    def test_second_bracket(self):
        """Income spanning two brackets (MFJ)."""
        # $30,000 taxable income
        # First $23,200 at 10% = $2,320
        # Remaining $6,800 at 12% = $816
        # Total = $3,136
        tax = calculate_federal_tax(30000)
        assert tax == pytest.approx(3136, rel=0.01)

    def test_higher_income(self):
        """Test higher income tax calculation."""
        # $150,000 taxable income
        tax = calculate_federal_tax(150000)
        # Should be progressive through multiple brackets
        assert tax > 0
        assert tax < 150000 * 0.37  # Less than max rate on all income


class TestStateTax:
    """Tests for state tax calculations."""

    def test_zero_income(self):
        """Zero income should result in zero tax."""
        assert calculate_state_tax(0, 0.05) == 0

    def test_flat_rate(self):
        """State tax should apply flat rate."""
        tax = calculate_state_tax(100000, 0.05)
        assert tax == pytest.approx(5000)

    def test_nc_rate(self):
        """Test North Carolina rate (5.25%)."""
        tax = calculate_state_tax(100000, 0.0525)
        assert tax == pytest.approx(5250)


class TestFicaTax:
    """Tests for FICA tax calculations."""

    def test_zero_income(self):
        """Zero income should result in zero tax."""
        assert calculate_fica_tax(0) == 0

    def test_below_ss_cap(self):
        """Income below Social Security cap."""
        income = 100000
        tax = calculate_fica_tax(income)
        expected_ss = income * 0.062
        expected_medicare = income * 0.0145
        assert tax == pytest.approx(expected_ss + expected_medicare)

    def test_above_ss_cap(self):
        """Income above Social Security cap should cap SS portion."""
        income = 200000
        tax = calculate_fica_tax(income)
        expected_ss = BASE_SS_WAGE_CAP * 0.062
        expected_medicare = income * 0.0145
        # Additional Medicare tax above $200k threshold
        assert tax >= expected_ss + expected_medicare

    def test_additional_medicare_tax(self):
        """Test additional Medicare tax above $200k."""
        income = 250000
        tax = calculate_fica_tax(income)
        expected_ss = BASE_SS_WAGE_CAP * 0.062
        base_medicare = income * 0.0145
        additional_medicare = (income - 200000) * 0.009
        assert tax == pytest.approx(expected_ss + base_medicare + additional_medicare)


class TestCalculateTaxes:
    """Tests for the complete tax calculation."""

    def test_complete_calculation(self):
        """Test complete tax breakdown."""
        result = calculate_taxes(
            gross_income=150000,
            pre_tax_deductions=20000,
            standard_deduction=14600,
            state_rate=0.05,
        )

        assert result.federal_tax > 0
        assert result.state_tax > 0
        assert result.fica_tax > 0
        assert result.total_tax == pytest.approx(
            result.federal_tax + result.state_tax + result.fica_tax
        )
        assert 0 < result.effective_rate < 1

    def test_pre_tax_deductions_reduce_taxes(self):
        """Pre-tax deductions should reduce federal and state taxes."""
        base = calculate_taxes(
            gross_income=100000,
            pre_tax_deductions=0,
        )

        with_deductions = calculate_taxes(
            gross_income=100000,
            pre_tax_deductions=20000,
        )

        assert with_deductions.federal_tax < base.federal_tax
        assert with_deductions.state_tax < base.state_tax
        # FICA is not affected by pre-tax deductions
        assert with_deductions.fica_tax == base.fica_tax


class TestCapitalGainsTax:
    """Tests for capital gains tax calculations."""

    def test_zero_gains(self):
        """Zero gains should result in zero tax."""
        assert calculate_capital_gains_tax(0, 50000) == 0

    def test_negative_gains(self):
        """Negative gains should result in zero tax."""
        assert calculate_capital_gains_tax(-1000, 50000) == 0

    def test_long_term_0_percent_bracket(self):
        """Low income should have 0% long-term capital gains."""
        # Under $47,025 total income
        tax = calculate_capital_gains_tax(10000, 30000)
        assert tax == pytest.approx(0)

    def test_long_term_15_percent_bracket(self):
        """Middle income should have 15% long-term capital gains."""
        # Total income of $100,000 (in 15% bracket)
        tax = calculate_capital_gains_tax(20000, 80000)
        # All gains should be taxed at 15%
        assert tax == pytest.approx(20000 * 0.15)

    def test_short_term_gains(self):
        """Short-term gains should be taxed as ordinary income."""
        tax = calculate_capital_gains_tax(10000, 50000, is_long_term=False)
        # Should be the marginal tax on additional $10k of ordinary income
        assert tax > 0

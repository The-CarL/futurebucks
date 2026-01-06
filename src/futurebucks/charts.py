"""Plotly and Altair chart functions for financial visualization."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
import pandas as pd

from futurebucks.models import SimulationResult, MonteCarloResult
import numpy as np


def format_currency(value: float) -> str:
    """Format a number as currency."""
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    elif abs(value) >= 1_000:
        return f"${value / 1_000:.0f}K"
    else:
        return f"${value:.0f}"


def net_worth_chart(
    result: SimulationResult,
    life_events: list | None = None,
    show_milestones: bool = True,
) -> go.Figure:
    """Create a net worth over time chart with milestone and life event markers.

    Args:
        result: Simulation result to visualize
        life_events: Optional list of LifeEvent objects to display
        show_milestones: Whether to show milestone annotations

    Returns:
        Plotly figure
    """
    years = [s.year for s in result.snapshots]
    net_worth = [s.net_worth for s in result.snapshots]
    ages = [s.age for s in result.snapshots]
    max_net_worth = max(net_worth)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=years,
            y=net_worth,
            mode="lines+markers",
            name="Net Worth",
            line=dict(width=3, color="#2E86AB"),
            marker=dict(size=6),
            hovertemplate="Year: %{x}<br>Age: %{customdata}<br>Net Worth: $%{y:,.0f}<extra></extra>",
            customdata=ages,
        )
    )

    # Add life event markers
    if life_events:
        event_icons = {
            "income_change": ("diamond", "#17A2B8", "Job Change"),
            "expense_change": ("square", "#FFC107", "Expense"),
            "windfall": ("star", "#28A745", "Windfall"),
            "asset_purchase": ("pentagon", "#6F42C1", "Purchase"),
            "child": ("circle", "#E83E8C", "Child"),
            "retirement": ("hexagon", "#20C997", "Retirement"),
        }

        # Group events by year to avoid overlapping
        events_by_year: dict[int, list] = {}
        for event in life_events:
            if event.year not in events_by_year:
                events_by_year[event.year] = []
            events_by_year[event.year].append(event)

        for year, year_events in events_by_year.items():
            if year < years[0] or year > years[-1]:
                continue

            idx = year - years[0]
            if idx < 0 or idx >= len(net_worth):
                continue

            # Get the most significant event for this year for the marker
            # Priority: child > windfall > income_change > asset_purchase > expense_change
            priority = ["child", "windfall", "income_change", "asset_purchase", "expense_change", "retirement"]
            primary_event = min(year_events, key=lambda e: priority.index(e.type) if e.type in priority else 99)

            symbol, color, type_label = event_icons.get(primary_event.type, ("circle", "#6C757D", "Event"))

            # Create hover text with all events for this year
            hover_lines = [f"<b>{year}</b>"]
            for evt in year_events:
                hover_lines.append(f"• {evt.name}")

            fig.add_trace(
                go.Scatter(
                    x=[year],
                    y=[net_worth[idx]],
                    mode="markers",
                    name=primary_event.name,
                    marker=dict(size=14, color=color, symbol=symbol, line=dict(width=2, color="white")),
                    hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
                    showlegend=False,
                )
            )

            # Add annotation for major events (child, windfall, asset_purchase)
            if primary_event.type in ["child", "windfall", "asset_purchase"]:
                # Shorten the name for annotation
                short_name = primary_event.name
                if len(short_name) > 20:
                    short_name = short_name[:17] + "..."

                fig.add_annotation(
                    x=year,
                    y=net_worth[idx],
                    text=short_name,
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=0.8,
                    arrowwidth=1,
                    arrowcolor=color,
                    font=dict(size=9, color=color),
                    ax=0,
                    ay=-35,
                    bgcolor="rgba(255,255,255,0.8)",
                )

    if show_milestones:
        milestones = result.milestones

        # Define all milestone thresholds with their values
        milestone_config = [
            ("net_worth_1m", 1_000_000, "$1M"),
            ("net_worth_2m", 2_000_000, "$2M"),
            ("net_worth_3m", 3_000_000, "$3M"),
            ("net_worth_5m", 5_000_000, "$5M"),
            ("net_worth_10m", 10_000_000, "$10M"),
            ("net_worth_20m", 20_000_000, "$20M"),
            ("net_worth_50m", 50_000_000, "$50M"),
            ("net_worth_100m", 100_000_000, "$100M"),
        ]

        # Add horizontal reference lines for relevant milestones
        for key, value, label in milestone_config:
            if value <= max_net_worth * 1.1:  # Show if within range
                fig.add_hline(
                    y=value,
                    line_dash="dot",
                    line_color="rgba(100, 100, 100, 0.3)",
                    annotation_text=label,
                    annotation_position="right",
                    annotation_font_size=10,
                    annotation_font_color="rgba(100, 100, 100, 0.7)",
                )

        # Mark FIRE milestone with special annotation
        if milestones.get("fire_number"):
            year = milestones["fire_number"]
            idx = year - years[0]
            if 0 <= idx < len(net_worth):
                fig.add_annotation(
                    x=year,
                    y=net_worth[idx],
                    text="FIRE!",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#28A745",
                    font=dict(color="#28A745", size=12, family="Arial Black"),
                    ax=40,
                    ay=-50,
                    bgcolor="rgba(40, 167, 69, 0.1)",
                    bordercolor="#28A745",
                    borderwidth=1,
                )

    fig.update_layout(
        title="Net Worth Over Time (with Life Events)",
        xaxis_title="Year",
        yaxis_title="Net Worth",
        yaxis_tickprefix="$",
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def income_expenses_chart(result: SimulationResult) -> go.Figure:
    """Create an income vs expenses chart.

    Args:
        result: Simulation result to visualize

    Returns:
        Plotly figure
    """
    years = [s.year for s in result.snapshots]
    gross_income = [s.gross_income for s in result.snapshots]
    net_income = [s.net_income for s in result.snapshots]
    expenses = [s.total_expenses for s in result.snapshots]
    savings = [s.savings for s in result.snapshots]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=years,
            y=gross_income,
            mode="lines",
            name="Gross Income",
            line=dict(width=2, color="#28A745"),
            hovertemplate="$%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=net_income,
            mode="lines",
            name="Net Income (after tax)",
            line=dict(width=2, color="#17A2B8"),
            hovertemplate="$%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=expenses,
            mode="lines",
            name="Expenses",
            line=dict(width=2, color="#DC3545"),
            hovertemplate="$%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=savings,
            mode="lines",
            name="Savings",
            line=dict(width=2, color="#6F42C1", dash="dash"),
            hovertemplate="$%{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Income, Expenses & Savings Over Time",
        xaxis_title="Year",
        yaxis_title="Amount",
        yaxis_tickprefix="$",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def asset_allocation_chart(result: SimulationResult) -> go.Figure:
    """Create an asset allocation over time chart.

    Args:
        result: Simulation result to visualize

    Returns:
        Plotly figure
    """
    years = [s.year for s in result.snapshots]

    # Get all unique asset names
    all_assets = set()
    for snapshot in result.snapshots:
        all_assets.update(snapshot.assets.keys())

    # Build traces for each asset
    fig = go.Figure()

    colors = [
        "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
        "#95C623", "#5C4D7D", "#2E294E", "#541388", "#D90368",
    ]

    for i, asset_name in enumerate(sorted(all_assets)):
        values = [s.assets.get(asset_name, 0) for s in result.snapshots]
        fig.add_trace(
            go.Scatter(
                x=years,
                y=values,
                mode="lines",
                name=asset_name,
                stackgroup="assets",
                line=dict(width=0.5, color=colors[i % len(colors)]),
                fillcolor=colors[i % len(colors)],
                hovertemplate=f"{asset_name}: $%{{y:,.0f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Asset Allocation Over Time",
        xaxis_title="Year",
        yaxis_title="Value",
        yaxis_tickprefix="$",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def tax_breakdown_chart(result: SimulationResult) -> go.Figure:
    """Create a tax breakdown over time chart.

    Args:
        result: Simulation result to visualize

    Returns:
        Plotly figure
    """
    years = [s.year for s in result.snapshots]
    federal = [s.federal_tax for s in result.snapshots]
    state = [s.state_tax for s in result.snapshots]
    fica = [s.fica_tax for s in result.snapshots]
    cap_gains = [s.capital_gains_tax for s in result.snapshots]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=years,
            y=federal,
            name="Federal Tax",
            marker_color="#DC3545",
            hovertemplate="$%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=years,
            y=state,
            name="State Tax",
            marker_color="#FFC107",
            hovertemplate="$%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=years,
            y=fica,
            name="FICA",
            marker_color="#17A2B8",
            hovertemplate="$%{y:,.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=years,
            y=cap_gains,
            name="Capital Gains",
            marker_color="#6F42C1",
            hovertemplate="$%{y:,.0f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Tax Breakdown Over Time",
        xaxis_title="Year",
        yaxis_title="Tax Amount",
        yaxis_tickprefix="$",
        barmode="stack",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def comparison_chart(results: list[SimulationResult]) -> go.Figure:
    """Create a net worth comparison chart for multiple scenarios.

    Args:
        results: List of simulation results to compare

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    colors = [
        "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
        "#95C623", "#5C4D7D", "#2E294E", "#541388", "#D90368",
    ]

    for i, result in enumerate(results):
        years = [s.year for s in result.snapshots]
        net_worth = [s.net_worth for s in result.snapshots]

        fig.add_trace(
            go.Scatter(
                x=years,
                y=net_worth,
                mode="lines",
                name=result.scenario_name,
                line=dict(width=2, color=colors[i % len(colors)]),
                hovertemplate=f"{result.scenario_name}<br>Year: %{{x}}<br>Net Worth: $%{{y:,.0f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Net Worth Comparison",
        xaxis_title="Year",
        yaxis_title="Net Worth",
        yaxis_tickprefix="$",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def savings_rate_chart(result: SimulationResult) -> go.Figure:
    """Create a savings rate over time chart.

    Args:
        result: Simulation result to visualize

    Returns:
        Plotly figure
    """
    years = [s.year for s in result.snapshots]
    savings_rate = [
        (s.savings / s.gross_income * 100) if s.gross_income > 0 else 0
        for s in result.snapshots
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=years,
            y=savings_rate,
            mode="lines+markers",
            name="Savings Rate",
            line=dict(width=2, color="#28A745"),
            marker=dict(size=6),
            hovertemplate="Year: %{x}<br>Savings Rate: %{y:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Savings Rate Over Time",
        xaxis_title="Year",
        yaxis_title="Savings Rate (%)",
        yaxis_ticksuffix="%",
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def summary_dashboard(result: SimulationResult) -> go.Figure:
    """Create a summary dashboard with multiple charts.

    Args:
        result: Simulation result to visualize

    Returns:
        Plotly figure with subplots
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Net Worth",
            "Income & Expenses",
            "Asset Allocation",
            "Tax Breakdown",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    years = [s.year for s in result.snapshots]

    # Net Worth
    fig.add_trace(
        go.Scatter(
            x=years,
            y=[s.net_worth for s in result.snapshots],
            mode="lines",
            name="Net Worth",
            line=dict(width=2, color="#2E86AB"),
        ),
        row=1,
        col=1,
    )

    # Income & Expenses
    fig.add_trace(
        go.Scatter(
            x=years,
            y=[s.net_income for s in result.snapshots],
            mode="lines",
            name="Net Income",
            line=dict(width=2, color="#28A745"),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=[s.total_expenses for s in result.snapshots],
            mode="lines",
            name="Expenses",
            line=dict(width=2, color="#DC3545"),
        ),
        row=1,
        col=2,
    )

    # Asset Allocation - simplified for dashboard
    all_assets = set()
    for snapshot in result.snapshots:
        all_assets.update(snapshot.assets.keys())

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#95C623"]
    for i, asset_name in enumerate(sorted(all_assets)[:5]):
        values = [s.assets.get(asset_name, 0) for s in result.snapshots]
        fig.add_trace(
            go.Scatter(
                x=years,
                y=values,
                mode="lines",
                name=asset_name,
                stackgroup="assets",
                line=dict(width=0.5, color=colors[i % len(colors)]),
            ),
            row=2,
            col=1,
        )

    # Tax Breakdown
    fig.add_trace(
        go.Bar(
            x=years,
            y=[s.federal_tax for s in result.snapshots],
            name="Federal",
            marker_color="#DC3545",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=years,
            y=[s.state_tax for s in result.snapshots],
            name="State",
            marker_color="#FFC107",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=700,
        showlegend=True,
        template="plotly_white",
        barmode="stack",
    )

    return fig


def synced_dashboard(
    result: SimulationResult,
    life_events: list | None = None,
) -> go.Figure:
    """Create a dashboard with synchronized crosshairs across all charts.

    Args:
        result: Simulation result to visualize
        life_events: Optional list of LifeEvent objects to display

    Returns:
        Plotly figure with subplots and shared x-axis hover
    """
    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=(
            "Net Worth",
            "Income, Expenses & Savings",
            "Asset Allocation",
            "Tax Breakdown",
        ),
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.3, 0.25, 0.25, 0.2],
    )

    years = [s.year for s in result.snapshots]
    ages = [s.age for s in result.snapshots]

    # Row 1: Net Worth
    fig.add_trace(
        go.Scatter(
            x=years,
            y=[s.net_worth for s in result.snapshots],
            mode="lines+markers",
            name="Net Worth",
            line=dict(width=3, color="#2E86AB"),
            marker=dict(size=4),
            customdata=ages,
            hovertemplate="Net Worth: $%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add FIRE milestone marker
    fire_year = result.milestones.get("fire_number")
    if fire_year:
        idx = fire_year - years[0]
        if 0 <= idx < len(result.snapshots):
            fig.add_trace(
                go.Scatter(
                    x=[fire_year],
                    y=[result.snapshots[idx].net_worth],
                    mode="markers+text",
                    name="FIRE",
                    marker=dict(size=15, color="#28A745", symbol="star"),
                    text=["FIRE!"],
                    textposition="top center",
                    textfont=dict(color="#28A745", size=10, family="Arial Black"),
                    hovertemplate="FIRE Achieved!<br>Year: %{x}<br>Net Worth: $%{y:,.0f}<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    # Add life event markers on net worth chart
    if life_events:
        event_colors = {
            "income_change": "#17A2B8",
            "expense_change": "#FFC107",
            "windfall": "#28A745",
            "asset_purchase": "#6F42C1",
            "child": "#E83E8C",
            "retirement": "#20C997",
        }

        for event in life_events:
            if event.year < years[0] or event.year > years[-1]:
                continue
            idx = event.year - years[0]
            if idx < 0 or idx >= len(result.snapshots):
                continue

            color = event_colors.get(event.type, "#6C757D")
            fig.add_trace(
                go.Scatter(
                    x=[event.year],
                    y=[result.snapshots[idx].net_worth],
                    mode="markers",
                    name=event.name,
                    marker=dict(size=10, color=color, line=dict(width=1, color="white")),
                    hovertemplate=f"{event.name}<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    # Row 2: Income & Expenses
    fig.add_trace(
        go.Scatter(
            x=years,
            y=[s.gross_income for s in result.snapshots],
            mode="lines",
            name="Gross Income",
            line=dict(width=2, color="#28A745"),
            hovertemplate="Gross: $%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=[s.net_income for s in result.snapshots],
            mode="lines",
            name="Net Income",
            line=dict(width=2, color="#17A2B8"),
            hovertemplate="Net: $%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=[s.total_expenses for s in result.snapshots],
            mode="lines",
            name="Expenses",
            line=dict(width=2, color="#DC3545"),
            hovertemplate="Expenses: $%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=years,
            y=[s.savings for s in result.snapshots],
            mode="lines",
            name="Savings",
            line=dict(width=2, color="#6F42C1", dash="dash"),
            hovertemplate="Savings: $%{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Row 3: Asset Allocation (stacked area)
    all_assets = set()
    for snapshot in result.snapshots:
        all_assets.update(snapshot.assets.keys())

    colors = [
        "#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
        "#95C623", "#5C4D7D", "#2E294E", "#541388", "#D90368",
    ]

    for i, asset_name in enumerate(sorted(all_assets)):
        values = [s.assets.get(asset_name, 0) for s in result.snapshots]
        fig.add_trace(
            go.Scatter(
                x=years,
                y=values,
                mode="lines",
                name=asset_name,
                stackgroup="assets",
                line=dict(width=0.5, color=colors[i % len(colors)]),
                fillcolor=colors[i % len(colors)],
                hovertemplate=f"{asset_name}: $%{{y:,.0f}}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    # Row 4: Tax Breakdown (stacked bar)
    fig.add_trace(
        go.Bar(
            x=years,
            y=[s.federal_tax for s in result.snapshots],
            name="Federal Tax",
            marker_color="#DC3545",
            hovertemplate="Federal: $%{y:,.0f}<extra></extra>",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=years,
            y=[s.state_tax for s in result.snapshots],
            name="State Tax",
            marker_color="#FFC107",
            hovertemplate="State: $%{y:,.0f}<extra></extra>",
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=years,
            y=[s.fica_tax for s in result.snapshots],
            name="FICA",
            marker_color="#17A2B8",
            hovertemplate="FICA: $%{y:,.0f}<extra></extra>",
        ),
        row=4,
        col=1,
    )

    # Update layout
    fig.update_layout(
        height=1000,
        showlegend=True,
        template="plotly_white",
        barmode="stack",
        hovermode="x unified",
        spikedistance=-1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    # Configure each axis with spikes for crosshair effect
    spike_config = dict(
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="gray",
        spikedash="solid",
    )

    # Update all x-axes with spike configuration
    for i in range(1, 5):
        fig.update_xaxes(**spike_config, row=i, col=1)
        fig.update_yaxes(tickprefix="$", **spike_config, row=i, col=1)

    # Only show x-axis label on bottom chart
    fig.update_xaxes(title_text="Year", row=4, col=1)

    return fig


def altair_synced_dashboard(
    result: SimulationResult,
    life_events: list | None = None,
) -> alt.VConcatChart:
    """Create a dashboard with synchronized crosshairs using Altair.

    Args:
        result: Simulation result to visualize
        life_events: Optional list of LifeEvent objects to display

    Returns:
        Altair vertically concatenated chart with linked hover
    """
    # Prepare main data
    data = []
    for s in result.snapshots:
        row = {
            "Year": s.year,
            "Age": s.age,
            "Net Worth": s.net_worth,
            "Gross Income": s.gross_income,
            "Net Income": s.net_income,
            "Expenses": s.total_expenses,
            "Savings": s.savings,
            "Federal Tax": s.federal_tax,
            "State Tax": s.state_tax,
            "FICA": s.fica_tax,
            "Total Tax": s.total_tax,
        }
        # Add assets
        for asset_name, balance in s.assets.items():
            row[asset_name] = balance
        data.append(row)

    df = pd.DataFrame(data)

    # Get asset columns
    asset_cols = [col for col in df.columns if col not in [
        "Year", "Age", "Net Worth", "Gross Income", "Net Income",
        "Expenses", "Savings", "Federal Tax", "State Tax", "FICA", "Total Tax"
    ]]

    # Create shared selection for synchronized hover - only add params once
    hover = alt.selection_point(
        name="hover",
        fields=["Year"],
        on="pointerover",
        nearest=True,
        empty=False,
    )

    # Vertical rule that follows hover (without adding params - will be added to first chart only)
    def make_rule(add_params=False):
        rule = alt.Chart(df).mark_rule(color="gray", strokeWidth=1).encode(
            x="Year:Q",
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        )
        if add_params:
            rule = rule.add_params(hover)
        return rule

    # Common x-axis configuration
    x_axis = alt.X("Year:Q", axis=alt.Axis(format="d", title=None))
    x_axis_bottom = alt.X("Year:Q", axis=alt.Axis(format="d", title="Year"))

    # 1. Net Worth Chart
    net_worth_line = alt.Chart(df).mark_line(color="#2E86AB", strokeWidth=2).encode(
        x=x_axis,
        y=alt.Y("Net Worth:Q", axis=alt.Axis(format="$,.0f", title="Net Worth")),
    )

    net_worth_points = alt.Chart(df).mark_circle(color="#2E86AB", size=60).encode(
        x=x_axis,
        y=alt.Y("Net Worth:Q"),
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("Year:Q", format="d"),
            alt.Tooltip("Age:Q"),
            alt.Tooltip("Net Worth:Q", format="$,.0f"),
        ],
    )

    # Add FIRE marker if achieved
    fire_year = result.milestones.get("fire_number")
    if fire_year:
        fire_df = df[df["Year"] == fire_year]
        fire_marker = alt.Chart(fire_df).mark_point(
            color="#28A745", size=200, shape="star", filled=True
        ).encode(
            x="Year:Q",
            y="Net Worth:Q",
        )
        fire_text = alt.Chart(fire_df).mark_text(
            dy=-15, color="#28A745", fontWeight="bold", fontSize=12
        ).encode(
            x="Year:Q",
            y="Net Worth:Q",
            text=alt.value("FIRE!"),
        )
        net_worth_chart = alt.layer(net_worth_line, net_worth_points, make_rule(add_params=True), fire_marker, fire_text)
    else:
        net_worth_chart = alt.layer(net_worth_line, net_worth_points, make_rule(add_params=True))

    # Add life event markers
    if life_events:
        event_data = []
        for event in life_events:
            if event.year >= df["Year"].min() and event.year <= df["Year"].max():
                nw = df[df["Year"] == event.year]["Net Worth"].values
                if len(nw) > 0:
                    event_data.append({
                        "Year": event.year,
                        "Net Worth": nw[0],
                        "Event": event.name,
                        "Type": event.type,
                    })

        if event_data:
            event_df = pd.DataFrame(event_data)
            event_markers = alt.Chart(event_df).mark_point(
                size=100, filled=True, strokeWidth=1, stroke="white"
            ).encode(
                x="Year:Q",
                y="Net Worth:Q",
                color=alt.Color("Type:N", scale=alt.Scale(
                    domain=["income_change", "expense_change", "windfall", "asset_purchase", "child", "retirement"],
                    range=["#17A2B8", "#FFC107", "#28A745", "#6F42C1", "#E83E8C", "#20C997"]
                ), legend=None),
                tooltip=["Year:Q", "Event:N", "Type:N"],
            )
            net_worth_chart = alt.layer(net_worth_chart, event_markers)

    net_worth_chart = net_worth_chart.properties(height=200, title="Net Worth")

    # 2. Income & Expenses Chart
    income_df = df.melt(
        id_vars=["Year"],
        value_vars=["Gross Income", "Net Income", "Expenses", "Savings"],
        var_name="Category",
        value_name="Amount",
    )

    income_lines = alt.Chart(income_df).mark_line(strokeWidth=2).encode(
        x=x_axis,
        y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f", title="Amount")),
        color=alt.Color("Category:N", scale=alt.Scale(
            domain=["Gross Income", "Net Income", "Expenses", "Savings"],
            range=["#28A745", "#17A2B8", "#DC3545", "#6F42C1"]
        )),
        strokeDash=alt.condition(
            alt.datum.Category == "Savings",
            alt.value([5, 5]),
            alt.value([0]),
        ),
    )

    income_points = alt.Chart(df).mark_circle(size=60, opacity=0).encode(
        x=x_axis,
        y=alt.Y("Gross Income:Q"),
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip("Year:Q", format="d"),
            alt.Tooltip("Gross Income:Q", format="$,.0f"),
            alt.Tooltip("Net Income:Q", format="$,.0f"),
            alt.Tooltip("Expenses:Q", format="$,.0f"),
            alt.Tooltip("Savings:Q", format="$,.0f"),
        ],
    )

    income_chart = alt.layer(income_lines, income_points, make_rule()).properties(
        height=150, title="Income, Expenses & Savings"
    )

    # 3. Asset Allocation Chart (stacked area)
    if asset_cols:
        asset_df = df.melt(
            id_vars=["Year"],
            value_vars=asset_cols,
            var_name="Asset",
            value_name="Balance",
        )

        asset_area = alt.Chart(asset_df).mark_area().encode(
            x=x_axis,
            y=alt.Y("Balance:Q", axis=alt.Axis(format="$,.0f", title="Value"), stack="zero"),
            color=alt.Color("Asset:N", scale=alt.Scale(scheme="category10")),
            tooltip=[
                alt.Tooltip("Year:Q", format="d"),
                alt.Tooltip("Asset:N"),
                alt.Tooltip("Balance:Q", format="$,.0f"),
            ],
        )

        asset_chart = alt.layer(asset_area, make_rule()).properties(
            height=150, title="Asset Allocation"
        )
    else:
        asset_chart = alt.Chart(df).mark_text(text="No assets").properties(height=150)

    # 4. Tax Breakdown Chart (stacked bar)
    tax_df = df.melt(
        id_vars=["Year"],
        value_vars=["Federal Tax", "State Tax", "FICA"],
        var_name="Tax Type",
        value_name="Amount",
    )

    tax_bars = alt.Chart(tax_df).mark_bar().encode(
        x=x_axis_bottom,
        y=alt.Y("Amount:Q", axis=alt.Axis(format="$,.0f", title="Tax Amount"), stack="zero"),
        color=alt.Color("Tax Type:N", scale=alt.Scale(
            domain=["Federal Tax", "State Tax", "FICA"],
            range=["#DC3545", "#FFC107", "#17A2B8"]
        )),
        tooltip=[
            alt.Tooltip("Year:Q", format="d"),
            alt.Tooltip("Tax Type:N"),
            alt.Tooltip("Amount:Q", format="$,.0f"),
        ],
    )

    tax_chart = alt.layer(tax_bars, make_rule()).properties(
        height=120, title="Tax Breakdown"
    )

    # Combine all charts vertically
    combined = alt.vconcat(
        net_worth_chart,
        income_chart,
        asset_chart,
        tax_chart,
        spacing=10,
    ).resolve_scale(
        color="independent"
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        grid=True,
        gridColor="#f0f0f0",
    )

    return combined


def monte_carlo_fan_chart(
    result: MonteCarloResult,
    show_median: bool = True,
    show_mean: bool = False,
) -> go.Figure:
    """Create a fan chart showing percentile bands over time.

    Args:
        result: Monte Carlo simulation result
        show_median: Show the 50th percentile line
        show_mean: Show the mean line

    Returns:
        Plotly figure with fan chart
    """
    years = [s.year for s in result.snapshots]
    ages = [s.age for s in result.snapshots]

    fig = go.Figure()

    # Add separate traces for each percentile boundary line
    # P90 line (top of outer band)
    fig.add_trace(go.Scatter(
        x=years,
        y=[s.net_worth_p90 for s in result.snapshots],
        mode='lines',
        line=dict(color='rgba(46, 134, 171, 0.3)', width=1),
        name='90th percentile',
        showlegend=False,
        hovertemplate="90th: $%{y:,.0f}<extra></extra>",
    ))

    # P10 line (bottom of outer band)
    fig.add_trace(go.Scatter(
        x=years,
        y=[s.net_worth_p10 for s in result.snapshots],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(46, 134, 171, 0.15)',
        line=dict(color='rgba(46, 134, 171, 0.3)', width=1),
        name='10th-90th percentile',
        hovertemplate="10th: $%{y:,.0f}<extra></extra>",
    ))

    # P75 line (top of inner band)
    fig.add_trace(go.Scatter(
        x=years,
        y=[s.net_worth_p75 for s in result.snapshots],
        mode='lines',
        line=dict(color='rgba(46, 134, 171, 0.5)', width=1),
        name='75th percentile',
        showlegend=False,
        hovertemplate="75th: $%{y:,.0f}<extra></extra>",
    ))

    # P25 line (bottom of inner band)
    fig.add_trace(go.Scatter(
        x=years,
        y=[s.net_worth_p25 for s in result.snapshots],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(46, 134, 171, 0.3)',
        line=dict(color='rgba(46, 134, 171, 0.5)', width=1),
        name='25th-75th percentile',
        hovertemplate="25th: $%{y:,.0f}<extra></extra>",
    ))

    # Median line
    if show_median:
        fig.add_trace(go.Scatter(
            x=years,
            y=[s.net_worth_p50 for s in result.snapshots],
            mode='lines',
            name='Median (50th)',
            line=dict(color='#2E86AB', width=3),
            customdata=ages,
            hovertemplate=(
                "Year: %{x}<br>"
                "Age: %{customdata}<br>"
                "Median Net Worth: $%{y:,.0f}<extra></extra>"
            ),
        ))

    # Mean line (optional)
    if show_mean:
        fig.add_trace(go.Scatter(
            x=years,
            y=[s.net_worth_mean for s in result.snapshots],
            mode='lines',
            name='Mean',
            line=dict(color='#A23B72', width=2, dash='dash'),
            hovertemplate="Mean: $%{y:,.0f}<extra></extra>",
        ))

    # Add percentile annotations at end
    last_snap = result.snapshots[-1]
    annotations_y = [
        (last_snap.net_worth_p90, "90th", "#2E86AB"),
        (last_snap.net_worth_p50, "50th", "#2E86AB"),
        (last_snap.net_worth_p10, "10th", "#2E86AB"),
    ]

    for y_val, label, color in annotations_y:
        fig.add_annotation(
            x=years[-1],
            y=y_val,
            text=f"{label}: ${y_val:,.0f}",
            showarrow=False,
            xanchor='left',
            xshift=10,
            font=dict(size=10, color=color),
        )

    fig.update_layout(
        title=f"Net Worth Projection ({result.num_simulations} simulations)",
        xaxis_title="Year",
        yaxis_title="Net Worth",
        yaxis_tickprefix="$",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def final_net_worth_histogram(
    result: MonteCarloResult,
    bins: int = 30,
) -> go.Figure:
    """Create a histogram of net worth distribution at end of simulation.

    Args:
        result: Monte Carlo simulation result
        bins: Number of histogram bins

    Returns:
        Plotly figure with histogram
    """
    values = result.final_net_worth_distribution

    # Get the final year/age from snapshots for the title
    final_snapshot = result.snapshots[-1] if result.snapshots else None
    if final_snapshot:
        title = f"Net Worth at Age {final_snapshot.age} ({final_snapshot.year})"
    else:
        title = "Net Worth Distribution (End of Simulation)"

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=bins,
        name='Net Worth',
        marker=dict(
            color='#2E86AB',
            line=dict(color='#1a5276', width=1),
        ),
        opacity=0.85,
    ))

    # Add vertical lines for key percentiles
    p50 = float(np.percentile(values, 50))
    p10 = float(np.percentile(values, 10))
    p90 = float(np.percentile(values, 90))

    for val, label, color in [
        (p10, '10th', '#DC3545'),
        (p50, 'Median', '#28A745'),
        (p90, '90th', '#17A2B8'),
    ]:
        fig.add_vline(
            x=val,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{label}: ${val/1_000_000:.1f}M",
            annotation_position="top",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Net Worth",
        yaxis_title="Frequency",
        xaxis_tickprefix="$",
        template="plotly_white",
        bargap=0.05,
    )

    return fig


def milestone_probability_chart(
    result: MonteCarloResult,
    use_target_probability: bool = True,
) -> go.Figure:
    """Create a chart showing milestone probabilities by retirement age.

    Args:
        result: Monte Carlo simulation result
        use_target_probability: If True, show probability by target retirement age

    Returns:
        Plotly figure with milestone probabilities
    """
    milestones = result.milestone_probabilities

    fig = go.Figure()

    # Use probability_by_target if available and requested
    def get_prob(m):
        if use_target_probability and m.probability_by_target is not None:
            return m.probability_by_target
        return m.probability

    # Bar chart for probabilities
    colors = []
    for m in milestones:
        prob = get_prob(m)
        if prob > 0.7:
            colors.append('#28A745')  # Green
        elif prob > 0.3:
            colors.append('#FFC107')  # Yellow
        else:
            colors.append('#DC3545')  # Red

    # Build labels with target values
    labels = []
    for m in milestones:
        if m.milestone == "fire_number":
            labels.append("FIRE")
        elif m.target_value:
            labels.append(f"${m.target_value / 1_000_000:.0f}M")
        else:
            labels.append(m.milestone.replace('_', ' ').title())

    probs = [get_prob(m) for m in milestones]

    fig.add_trace(go.Bar(
        x=labels,
        y=[p * 100 for p in probs],
        marker_color=colors,
        text=[f"{p:.0%}" for p in probs],
        textposition='outside',
        hovertemplate=(
            "%{x}<br>"
            "Probability: %{y:.1f}%<br>"
            "<extra></extra>"
        ),
    ))

    # Build title with target age if available
    target_age = milestones[0].target_age if milestones else None
    if target_age and use_target_probability:
        title = f"Probability by Age {target_age}"
    else:
        title = "Milestone Probabilities"

    fig.update_layout(
        title=title,
        xaxis_title="Milestone",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 105],
        template="plotly_white",
    )

    return fig


def milestone_timing_table(
    result: MonteCarloResult,
    birth_year: int | None = None,
) -> pd.DataFrame:
    """Create a DataFrame showing milestone timing statistics.

    Args:
        result: Monte Carlo simulation result
        birth_year: Birth year to convert years to ages (optional)

    Returns:
        DataFrame with milestone timing data
    """
    def year_to_age(year: int | None) -> str:
        if year is None:
            return '-'
        if birth_year:
            return f"Age {year - birth_year}"
        return str(year)

    def format_range(p10: int | None, p90: int | None) -> str:
        if p10 is None or p90 is None:
            return '-'
        if birth_year:
            return f"Ages {p10 - birth_year}-{p90 - birth_year}"
        return f"{p10}-{p90}"

    data = []
    for m in result.milestone_probabilities:
        # Use probability_by_target if available (by retirement age)
        prob = m.probability_by_target if m.probability_by_target is not None else m.probability

        # Build milestone label
        if m.milestone == "fire_number":
            label = "FIRE (25x expenses)"
        elif m.target_value:
            label = f"${m.target_value / 1_000_000:.0f}M Net Worth"
        else:
            label = m.milestone.replace('_', ' ').title()

        data.append({
            'Milestone': label,
            f'Prob by Age {m.target_age}' if m.target_age else 'Probability': f"{prob:.0%}",
            'Typical Range': format_range(m.p10_year, m.p90_year),
            'Median': year_to_age(m.median_year),
        })
    return pd.DataFrame(data)


# =============================================================================
# Sensitivity Analysis Charts
# =============================================================================


def tornado_chart(
    result: "TornadoResult",
    color_positive: str = "#28A745",
    color_negative: str = "#DC3545",
) -> go.Figure:
    """Create a tornado chart showing variable sensitivity.

    Args:
        result: TornadoResult from sensitivity analysis
        color_positive: Color for positive impacts
        color_negative: Color for negative impacts

    Returns:
        Plotly figure with tornado chart
    """
    from futurebucks.sensitivity import TornadoResult  # Avoid circular import

    fig = go.Figure()

    # Sort by impact (already sorted, but ensure order)
    sensitivities = sorted(result.sensitivities, key=lambda x: x.impact, reverse=True)

    labels = []
    low_deltas = []
    high_deltas = []

    for sens in sensitivities:
        labels.append(sens.variable_label)
        # Calculate delta from base
        low_deltas.append(sens.low_result - sens.base_result)
        high_deltas.append(sens.high_result - sens.base_result)

    # Determine if metric is something where higher is better (net worth) or lower is better (fire year)
    # For net worth: positive delta = good (green), negative delta = bad (red)
    # For fire year: positive delta = bad (red, later FIRE), negative delta = good (green, earlier FIRE)
    is_lower_better = result.metric_name == "fire_year"

    def get_bar_color(delta: float) -> str:
        """Get color based on whether outcome is better or worse than baseline."""
        if is_lower_better:
            # Lower is better (fire_year) - negative delta is good
            return color_positive if delta < 0 else color_negative
        else:
            # Higher is better (net_worth) - positive delta is good
            return color_positive if delta > 0 else color_negative

    # Create bar traces for low and high values
    # Low value bars (extend left from base)
    fig.add_trace(go.Bar(
        y=labels,
        x=low_deltas,
        orientation='h',
        name='Low (-20%)',
        marker_color=[get_bar_color(d) for d in low_deltas],
        customdata=[
            f"${abs(d):,.0f}" if result.metric_name == "final_net_worth" else f"{abs(d):.1f}"
            for d in low_deltas
        ],
        hovertemplate=(
            "%{y}<br>"
            "Low setting impact: %{customdata}<br>"
            "<extra></extra>"
        ),
    ))

    # High value bars (extend right from base)
    fig.add_trace(go.Bar(
        y=labels,
        x=high_deltas,
        orientation='h',
        name='High (+20%)',
        marker_color=[get_bar_color(d) for d in high_deltas],
        customdata=[
            f"${abs(d):,.0f}" if result.metric_name == "final_net_worth" else f"{abs(d):.1f}"
            for d in high_deltas
        ],
        hovertemplate=(
            "%{y}<br>"
            "High setting impact: %{customdata}<br>"
            "<extra></extra>"
        ),
    ))

    # Add vertical line at base value
    fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=2)

    # Format x-axis label based on metric
    if result.metric_name == "final_net_worth":
        xaxis_title = "Impact on Final Net Worth"
        xaxis_tickprefix = "$"
    elif result.metric_name == "fire_year":
        xaxis_title = "Impact on FIRE Year"
        xaxis_tickprefix = ""
    else:
        xaxis_title = f"Impact on {result.metric_name.replace('_', ' ').title()}"
        xaxis_tickprefix = ""

    fig.update_layout(
        title=f"Sensitivity Analysis: {result.metric_name.replace('_', ' ').title()}",
        xaxis_title=xaxis_title,
        yaxis_title="Variable",
        xaxis_tickprefix=xaxis_tickprefix,
        barmode='overlay',
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=max(400, len(sensitivities) * 35 + 100),  # Dynamic height
    )

    # Reverse y-axis so highest impact is at top
    fig.update_yaxes(autorange="reversed")

    return fig


def sensitivity_heatmap(
    result: "WhatIfMatrix",
    colorscale: str = "RdYlGn",
) -> go.Figure:
    """Create a heatmap showing what-if analysis results.

    Args:
        result: WhatIfMatrix from sensitivity analysis
        colorscale: Plotly colorscale name

    Returns:
        Plotly figure with heatmap
    """
    from futurebucks.sensitivity import WhatIfMatrix  # Avoid circular import

    # Build 2D matrix from cells
    n_rows = len(result.variable2_values)
    n_cols = len(result.variable1_values)

    # Create matrix
    z_matrix = np.zeros((n_rows, n_cols))
    cell_lookup = {(c.var1_value, c.var2_value): c.result for c in result.cells}

    for i, v2 in enumerate(result.variable2_values):
        for j, v1 in enumerate(result.variable1_values):
            z_matrix[i, j] = cell_lookup.get((v1, v2), result.base_result)

    # Format axis labels
    def format_value(v):
        if abs(v) >= 1000:
            return f"${v/1000:.0f}K" if v > 0 else f"-${abs(v)/1000:.0f}K"
        elif abs(v) < 1:
            return f"{v:.1%}"
        else:
            return f"{v:.0f}"

    x_labels = [format_value(v) for v in result.variable1_values]
    y_labels = [format_value(v) for v in result.variable2_values]

    # Reverse colorscale for fire_year (lower is better)
    if result.metric_name == "fire_year":
        colorscale = "RdYlGn_r"

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        colorbar=dict(
            title=result.metric_name.replace("_", " ").title(),
            tickprefix="$" if result.metric_name == "final_net_worth" else "",
        ),
        hovertemplate=(
            f"{result.variable1_label}: %{{x}}<br>"
            f"{result.variable2_label}: %{{y}}<br>"
            f"Result: $%{{z:,.0f}}<extra></extra>"
            if result.metric_name == "final_net_worth" else
            f"{result.variable1_label}: %{{x}}<br>"
            f"{result.variable2_label}: %{{y}}<br>"
            f"Result: %{{z}}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=f"What-If Analysis: {result.metric_name.replace('_', ' ').title()}",
        xaxis_title=result.variable1_label,
        yaxis_title=result.variable2_label,
        template="plotly_white",
    )

    return fig


def spider_chart(
    results: list["SensitivityResult"],
    metric_name: str = "final_net_worth",
) -> go.Figure:
    """Create a spider/radar chart showing relative sensitivity.

    Args:
        results: List of SensitivityResult from tornado analysis
        metric_name: Name of the metric being analyzed

    Returns:
        Plotly figure with spider chart
    """
    from futurebucks.sensitivity import SensitivityResult  # Avoid circular import

    if not results:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return fig

    # Take top N results (max 8 for readability)
    results = sorted(results, key=lambda x: x.impact, reverse=True)[:8]

    # Calculate relative impacts (normalize to 0-100 scale)
    max_impact = max(r.impact for r in results) if results else 1
    normalized_impacts = [r.impact / max_impact * 100 for r in results]

    labels = [r.variable_label for r in results]

    # Close the spider by repeating first point
    labels_closed = labels + [labels[0]]
    impacts_closed = normalized_impacts + [normalized_impacts[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=impacts_closed,
        theta=labels_closed,
        fill='toself',
        name='Impact',
        fillcolor='rgba(46, 134, 171, 0.3)',
        line=dict(color='#2E86AB', width=2),
        hovertemplate=(
            "%{theta}<br>"
            "Relative Impact: %{r:.1f}%<br>"
            "<extra></extra>"
        ),
    ))

    # Add markers for each point
    fig.add_trace(go.Scatterpolar(
        r=normalized_impacts,
        theta=labels,
        mode='markers',
        marker=dict(size=10, color='#2E86AB'),
        showlegend=False,
        customdata=[
            f"${r.impact:,.0f}" if metric_name == "final_net_worth" else f"{r.impact:.2f}"
            for r in results
        ],
        hovertemplate=(
            "%{theta}<br>"
            "Actual Impact: %{customdata}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 105],
                ticksuffix="%",
            ),
        ),
        title=f"Sensitivity Spider Chart: {metric_name.replace('_', ' ').title()}",
        showlegend=False,
        template="plotly_white",
    )

    return fig


def sensitivity_summary_table(
    results: list["SensitivityResult"],
    metric_name: str = "final_net_worth",
) -> pd.DataFrame:
    """Create a summary DataFrame of sensitivity results.

    Args:
        results: List of SensitivityResult from tornado analysis
        metric_name: Name of the metric being analyzed

    Returns:
        DataFrame with sensitivity summary
    """
    from futurebucks.sensitivity import SensitivityResult  # Avoid circular import

    data = []
    for r in results:
        if metric_name == "final_net_worth":
            base_fmt = f"${r.base_result:,.0f}"
            low_fmt = f"${r.low_result:,.0f}"
            high_fmt = f"${r.high_result:,.0f}"
            impact_fmt = f"${r.impact:,.0f}"
        elif metric_name == "fire_year":
            base_fmt = f"{int(r.base_result)}"
            low_fmt = f"{int(r.low_result)}"
            high_fmt = f"{int(r.high_result)}"
            impact_fmt = f"{r.impact:.1f} years"
        else:
            base_fmt = f"{r.base_result:.2f}"
            low_fmt = f"{r.low_result:.2f}"
            high_fmt = f"{r.high_result:.2f}"
            impact_fmt = f"{r.impact:.2f}"

        data.append({
            "Variable": r.variable_label,
            "Base Value": f"{r.base_value:,.2f}" if r.base_value < 100 else f"{r.base_value:,.0f}",
            "Low (-20%)": r.low_value,
            "High (+20%)": r.high_value,
            f"Result at Low": low_fmt,
            f"Result at High": high_fmt,
            "Impact": impact_fmt,
        })

    return pd.DataFrame(data)

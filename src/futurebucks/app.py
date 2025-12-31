"""Streamlit application for FutureBucks financial simulation."""

import streamlit as st
import pandas as pd

from futurebucks.models import SimulationResult
from futurebucks.scenarios import (
    list_scenarios,
    list_sample_scenarios,
    load_scenario,
    save_scenario,
    copy_samples_to_scenarios,
    scenario_to_yaml,
    yaml_to_scenario,
)
from futurebucks.simulation import run_simulation
from futurebucks.charts import (
    net_worth_chart,
    income_expenses_chart,
    asset_allocation_chart,
    tax_breakdown_chart,
    comparison_chart,
    savings_rate_chart,
)


st.set_page_config(
    page_title="FutureBucks",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)


def format_currency(value: float) -> str:
    """Format a number as currency."""
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    else:
        return f"${value:.0f}"


def init_session_state():
    """Initialize session state variables."""
    if "loaded_scenario" not in st.session_state:
        st.session_state.loaded_scenario = None
    if "simulation_result" not in st.session_state:
        st.session_state.simulation_result = None
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = []
    if "yaml_editor_content" not in st.session_state:
        st.session_state.yaml_editor_content = ""


def sidebar():
    """Render the sidebar."""
    st.sidebar.title("FutureBucks")
    st.sidebar.caption("Life-Event Financial Simulator")

    st.sidebar.divider()

    # Load sample scenarios button
    if st.sidebar.button("Load Sample Scenarios", width="stretch"):
        copied = copy_samples_to_scenarios()
        if copied:
            st.sidebar.success(f"Copied {len(copied)} sample scenarios")
        else:
            st.sidebar.info("No new samples to copy")
        st.rerun()

    # Scenario selector
    scenarios = list_scenarios()

    if not scenarios:
        st.sidebar.warning("No scenarios found. Click 'Load Sample Scenarios' to get started.")
        sample_scenarios = list_sample_scenarios()
        if sample_scenarios:
            st.sidebar.caption(f"{len(sample_scenarios)} sample scenarios available")
        return

    st.sidebar.subheader("Select Scenario")
    selected = st.sidebar.selectbox(
        "Scenario file",
        scenarios,
        label_visibility="collapsed",
    )

    if st.sidebar.button("Load Scenario", width="stretch"):
        try:
            scenario = load_scenario(selected)
            st.session_state.loaded_scenario = scenario
            st.session_state.yaml_editor_content = scenario_to_yaml(scenario)
            # Run simulation immediately
            result = run_simulation(scenario)
            st.session_state.simulation_result = result
            st.sidebar.success(f"Loaded: {scenario.name}")
        except Exception as e:
            st.sidebar.error(f"Error loading scenario: {e}")

    st.sidebar.divider()

    # Scenario comparison
    st.sidebar.subheader("Compare Scenarios")
    comparison_scenarios = st.sidebar.multiselect(
        "Select scenarios to compare",
        scenarios,
        label_visibility="collapsed",
    )

    if st.sidebar.button("Compare", width="stretch") and comparison_scenarios:
        results = []
        for filename in comparison_scenarios:
            try:
                scenario = load_scenario(filename)
                result = run_simulation(scenario)
                results.append(result)
            except Exception as e:
                st.sidebar.error(f"Error with {filename}: {e}")
        st.session_state.comparison_results = results
        st.sidebar.success(f"Loaded {len(results)} scenarios for comparison")

    st.sidebar.divider()

    # Info
    st.sidebar.caption(
        "**Note:** This tool is for planning purposes only and should not be "
        "considered financial or tax advice. Consult a professional for "
        "personalized guidance."
    )


def tab_results(result: SimulationResult):
    """Render the single scenario results tab."""
    scenario = st.session_state.loaded_scenario

    first_snapshot = result.snapshots[0]
    last_snapshot = result.snapshots[-1]

    # Life Events Timeline - Prominent display at top
    if scenario.life_events:
        st.subheader("Life Events Timeline")

        # Sort events by year
        sorted_events = sorted(scenario.life_events, key=lambda e: e.year)

        # Create a timeline table
        events_data = []
        event_icons = {
            "income_change": "💼",
            "expense_change": "💰",
            "windfall": "🎉",
            "asset_purchase": "🏠",
            "child": "👶",
            "retirement": "🏖️",
        }

        for event in sorted_events:
            age = event.year - scenario.person.birth_year
            icon = event_icons.get(event.type, "📌")
            events_data.append({
                "Year": event.year,
                "Age": age,
                "Event": f"{icon} {event.name}",
                "Type": event.type.replace("_", " ").title(),
            })

        st.dataframe(
            pd.DataFrame(events_data),
            width="stretch",
            hide_index=True,
            height=min(400, len(events_data) * 35 + 38),
        )

        st.divider()

    # Key Milestones
    st.subheader("Key Milestones")

    # FIRE milestone highlight
    fire_year = result.milestones.get("fire_number")
    retirement_year = result.milestones.get("retirement_ready")

    if fire_year:
        fire_age = fire_year - scenario.person.birth_year
        st.success(f"**FIRE Achieved: {fire_year} (Age {fire_age})** - Your investments can cover your expenses with a 4% withdrawal rate!")

    # Major milestones in columns
    milestone_cols = st.columns(4)

    key_milestones = [
        ("net_worth_1m", "$1M"),
        ("net_worth_2m", "$2M"),
        ("net_worth_5m", "$5M"),
        ("net_worth_10m", "$10M"),
        ("net_worth_20m", "$20M"),
        ("net_worth_50m", "$50M"),
        ("net_worth_100m", "$100M"),
    ]

    # Filter to only show relevant milestones (achieved or within reach)
    relevant_milestones = []
    for key, label in key_milestones:
        year = result.milestones.get(key)
        if year:
            age = year - scenario.person.birth_year
            relevant_milestones.append((label, year, age, True))
        elif last_snapshot.net_worth >= float(label.replace("$", "").replace("M", "")) * 500_000:
            # Show if we're at least halfway there
            relevant_milestones.append((label, None, None, False))

    # Display up to 4 key milestones
    for i, (label, year, age, achieved) in enumerate(relevant_milestones[:4]):
        with milestone_cols[i]:
            if achieved:
                st.metric(label, f"{year}", f"Age {age}")
            else:
                st.metric(label, "Not reached", "")

    st.divider()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Starting Net Worth",
            format_currency(first_snapshot.net_worth),
        )
    with col2:
        st.metric(
            "Final Net Worth",
            format_currency(last_snapshot.net_worth),
        )
    with col3:
        growth = last_snapshot.net_worth - first_snapshot.net_worth
        st.metric(
            "Total Growth",
            format_currency(growth),
        )
    with col4:
        st.metric(
            "Cumulative Taxes",
            format_currency(last_snapshot.cumulative_taxes_paid),
        )

    st.divider()

    # Charts
    st.plotly_chart(net_worth_chart(result, life_events=scenario.life_events), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(income_expenses_chart(result), use_container_width=True)

    with col2:
        st.plotly_chart(asset_allocation_chart(result), use_container_width=True)

    st.plotly_chart(tax_breakdown_chart(result), use_container_width=True)

    st.divider()

    # All Milestones Table
    with st.expander("View All Milestones"):
        milestone_data = []
        all_milestone_labels = {
            "net_worth_100k": "Net Worth $100K",
            "net_worth_500k": "Net Worth $500K",
            "net_worth_1m": "Net Worth $1M",
            "net_worth_2m": "Net Worth $2M",
            "net_worth_3m": "Net Worth $3M",
            "net_worth_5m": "Net Worth $5M",
            "net_worth_10m": "Net Worth $10M",
            "net_worth_20m": "Net Worth $20M",
            "net_worth_50m": "Net Worth $50M",
            "net_worth_100m": "Net Worth $100M",
            "fire_number": "FIRE (4% rule)",
            "retirement_ready": "Retirement Ready (25x expenses)",
        }

        for key, label in all_milestone_labels.items():
            year = result.milestones.get(key)
            if year:
                age = year - scenario.person.birth_year
                milestone_data.append({"Milestone": label, "Year": str(year), "Age": str(age), "Status": "Achieved"})
            else:
                milestone_data.append({"Milestone": label, "Year": "-", "Age": "-", "Status": "Not reached"})

        st.dataframe(
            pd.DataFrame(milestone_data),
            width="stretch",
            hide_index=True,
        )

    # Detailed yearly table
    with st.expander("View Yearly Details"):
        df_data = []
        for snapshot in result.snapshots:
            df_data.append({
                "Year": snapshot.year,
                "Age": snapshot.age,
                "Gross Income": f"${snapshot.gross_income:,.0f}",
                "Total Tax": f"${snapshot.total_tax:,.0f}",
                "Net Income": f"${snapshot.net_income:,.0f}",
                "Expenses": f"${snapshot.total_expenses:,.0f}",
                "Savings": f"${snapshot.savings:,.0f}",
                "Net Worth": f"${snapshot.net_worth:,.0f}",
            })

        st.dataframe(
            pd.DataFrame(df_data),
            width="stretch",
            hide_index=True,
        )


def tab_editor():
    """Render the scenario editor tab."""
    st.subheader("Scenario Editor")

    if not st.session_state.loaded_scenario:
        st.info("Load a scenario from the sidebar to edit it.")
        return

    # YAML editor
    yaml_content = st.text_area(
        "Edit scenario YAML",
        value=st.session_state.yaml_editor_content,
        height=500,
        key="yaml_editor",
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Validate", width="stretch"):
            try:
                scenario = yaml_to_scenario(yaml_content)
                st.success(f"Valid scenario: {scenario.name}")
                st.session_state.yaml_editor_content = yaml_content
            except Exception as e:
                st.error(f"Invalid YAML: {e}")

    with col2:
        if st.button("Run Simulation", width="stretch"):
            try:
                scenario = yaml_to_scenario(yaml_content)
                st.session_state.loaded_scenario = scenario
                st.session_state.yaml_editor_content = yaml_content
                result = run_simulation(scenario)
                st.session_state.simulation_result = result
                st.success("Simulation complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # Save scenario
    st.subheader("Save Scenario")

    filename = st.text_input(
        "Filename (without .yaml extension)",
        value=st.session_state.loaded_scenario.name.lower().replace(" ", "_") if st.session_state.loaded_scenario else "",
    )

    if st.button("Save", width="stretch"):
        if not filename:
            st.error("Please enter a filename")
        else:
            try:
                scenario = yaml_to_scenario(yaml_content)
                filepath = save_scenario(scenario, f"{filename}.yaml")
                st.success(f"Saved to {filepath}")
            except Exception as e:
                st.error(f"Error saving: {e}")


def tab_comparison():
    """Render the scenario comparison tab."""
    st.subheader("Scenario Comparison")

    if not st.session_state.comparison_results:
        st.info("Select scenarios to compare in the sidebar.")
        return

    results = st.session_state.comparison_results

    # Net worth comparison chart
    st.plotly_chart(comparison_chart(results), width="stretch")

    st.divider()

    # Side-by-side milestone comparison
    st.subheader("Milestone Comparison")

    milestone_labels = {
        "net_worth_1m": "$1M",
        "net_worth_2m": "$2M",
        "net_worth_5m": "$5M",
        "net_worth_10m": "$10M",
        "net_worth_20m": "$20M",
        "net_worth_50m": "$50M",
        "fire_number": "FIRE",
    }

    # Build comparison table
    comparison_data = {"Milestone": list(milestone_labels.values())}

    for result in results:
        col_data = []
        for key in milestone_labels:
            year = result.milestones.get(key)
            col_data.append(str(year) if year else "-")
        comparison_data[result.scenario_name] = col_data

    st.dataframe(
        pd.DataFrame(comparison_data),
        width="stretch",
        hide_index=True,
    )

    st.divider()

    # Final net worth comparison
    st.subheader("Final Net Worth Comparison")

    final_data = []
    for result in results:
        last = result.snapshots[-1]
        final_data.append({
            "Scenario": result.scenario_name,
            "Final Net Worth": format_currency(last.net_worth),
            "Final Year": last.year,
            "Final Age": last.age,
            "Total Taxes Paid": format_currency(last.cumulative_taxes_paid),
        })

    st.dataframe(
        pd.DataFrame(final_data),
        width="stretch",
        hide_index=True,
    )


def main():
    """Main application entry point."""
    init_session_state()
    sidebar()

    # Main content area
    if st.session_state.loaded_scenario:
        st.title(st.session_state.loaded_scenario.name)
        st.caption(st.session_state.loaded_scenario.description)
    else:
        st.title("FutureBucks")
        st.caption("Life-event-driven financial simulation tool")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Results", "Editor", "Comparison"])

    with tab1:
        if st.session_state.simulation_result:
            tab_results(st.session_state.simulation_result)
        else:
            st.info("Load and run a scenario to see results.")

    with tab2:
        tab_editor()

    with tab3:
        tab_comparison()


if __name__ == "__main__":
    main()

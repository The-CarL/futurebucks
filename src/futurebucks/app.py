"""Streamlit application for FutureBucks financial simulation."""

import streamlit as st
import pandas as pd

from futurebucks.models import SimulationResult, MonteCarloConfig, MonteCarloResult
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
from futurebucks.monte_carlo import run_monte_carlo
from futurebucks.charts import (
    net_worth_chart,
    income_expenses_chart,
    asset_allocation_chart,
    tax_breakdown_chart,
    comparison_chart,
    savings_rate_chart,
    monte_carlo_fan_chart,
    final_net_worth_histogram,
    milestone_probability_chart,
    milestone_timing_table,
    tornado_chart,
    sensitivity_heatmap,
    spider_chart,
    sensitivity_summary_table,
)
from futurebucks.sensitivity import SensitivityAnalyzer, run_sensitivity_analysis


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
    if "mc_result" not in st.session_state:
        st.session_state.mc_result = None
    if "mc_enabled" not in st.session_state:
        st.session_state.mc_enabled = False
    if "mc_num_sims" not in st.session_state:
        st.session_state.mc_num_sims = 500
    if "sensitivity_result" not in st.session_state:
        st.session_state.sensitivity_result = None
    if "what_if_result" not in st.session_state:
        st.session_state.what_if_result = None


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

            # Run Monte Carlo if enabled
            if st.session_state.mc_enabled:
                config = MonteCarloConfig(num_simulations=st.session_state.mc_num_sims)
                mc_result = run_monte_carlo(scenario, config)
                st.session_state.mc_result = mc_result
            else:
                st.session_state.mc_result = None

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

    # Monte Carlo settings
    st.sidebar.subheader("Monte Carlo")
    st.session_state.mc_enabled = st.sidebar.toggle(
        "Enable Monte Carlo",
        value=st.session_state.mc_enabled,
        help="Run multiple simulations with stochastic parameters",
    )

    if st.session_state.mc_enabled:
        st.session_state.mc_num_sims = st.sidebar.slider(
            "Number of simulations",
            min_value=100,
            max_value=15000,
            value=st.session_state.mc_num_sims,
            step=100,
            help="More simulations = more accurate results but slower (10k+ recommended for detailed analysis)",
        )

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


def tab_monte_carlo():
    """Render the Monte Carlo results tab."""
    st.subheader("Monte Carlo Analysis")

    if not st.session_state.mc_result:
        if not st.session_state.mc_enabled:
            st.info("Enable Monte Carlo in the sidebar and reload a scenario to see probabilistic analysis.")
        else:
            st.info("Load a scenario to run Monte Carlo simulation.")
        return

    mc_result: MonteCarloResult = st.session_state.mc_result
    scenario = st.session_state.loaded_scenario

    # Summary metrics
    st.markdown(f"**{mc_result.num_simulations} simulations run**")

    final_snapshot = mc_result.snapshots[-1]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "10th Percentile",
            format_currency(final_snapshot.net_worth_p10),
            help="Pessimistic outcome - only 10% of simulations ended lower",
        )
    with col2:
        st.metric(
            "Median (50th)",
            format_currency(final_snapshot.net_worth_p50),
            help="Middle outcome - half ended higher, half lower",
        )
    with col3:
        st.metric(
            "90th Percentile",
            format_currency(final_snapshot.net_worth_p90),
            help="Optimistic outcome - only 10% of simulations ended higher",
        )
    with col4:
        st.metric(
            "Mean ± Std Dev",
            format_currency(final_snapshot.net_worth_mean),
            f"± {format_currency(final_snapshot.net_worth_stddev)}",
        )

    st.divider()

    # Milestone Probabilities
    st.subheader("Milestone Probabilities")
    retirement_age = scenario.person.retirement_age
    st.caption(f"Showing probability of reaching milestones by retirement age {retirement_age}")

    for prob in mc_result.milestone_probabilities:
        # Build milestone label
        if prob.milestone == "fire_number":
            label = "FIRE (25x expenses)"
        else:
            label = prob.milestone.replace("_", " ").title()
            if prob.target_value:
                label = f"{label} (${prob.target_value / 1_000_000:.0f}M)"

        # Use probability_by_target as primary metric (by retirement age)
        display_prob = prob.probability_by_target if prob.probability_by_target is not None else prob.probability

        col1, col2 = st.columns([3, 1])
        with col1:
            # Show probability by retirement age
            if prob.target_age and prob.probability_by_target is not None:
                progress_text = f"{label} by age {prob.target_age}: **{display_prob:.0%}**"
            else:
                progress_text = f"{label}: **{display_prob:.0%}**"

            st.progress(display_prob, text=progress_text)

        with col2:
            # Show when it's likely achieved (range)
            if prob.p10_year and prob.p90_year:
                p10_age = prob.p10_year - scenario.person.birth_year
                p90_age = prob.p90_year - scenario.person.birth_year
                median_age = prob.median_year - scenario.person.birth_year if prob.median_year else None

                if p10_age == p90_age:
                    st.caption(f"Typically age {p10_age}")
                elif median_age:
                    st.caption(f"Ages {p10_age}-{p90_age} (median {median_age})")
                else:
                    st.caption(f"Ages {p10_age}-{p90_age}")
            elif prob.probability == 0:
                st.caption("Not reached")
            else:
                st.caption("—")

    st.divider()

    # Fan Chart
    st.subheader("Net Worth Projections")
    st.plotly_chart(monte_carlo_fan_chart(mc_result), use_container_width=True)

    # Histogram and Milestone Chart side by side
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(final_net_worth_histogram(mc_result), use_container_width=True)

    with col2:
        st.plotly_chart(milestone_probability_chart(mc_result), use_container_width=True)

    # Detailed timing table
    with st.expander("Milestone Timing Details"):
        timing_df = milestone_timing_table(mc_result, birth_year=scenario.person.birth_year)
        st.dataframe(timing_df, use_container_width=True, hide_index=True)


def tab_sensitivity():
    """Render the Sensitivity Analysis tab."""
    st.subheader("Sensitivity Analysis")

    if not st.session_state.loaded_scenario:
        st.info("Load a scenario from the sidebar to run sensitivity analysis.")
        return

    scenario = st.session_state.loaded_scenario

    # Controls
    st.markdown("Analyze which variables have the most impact on your financial outcomes.")

    col1, col2, col3 = st.columns(3)

    with col1:
        target_metric = st.selectbox(
            "Target Metric",
            ["final_net_worth", "fire_year", "final_savings_rate"],
            format_func=lambda x: {
                "final_net_worth": "Final Net Worth",
                "fire_year": "FIRE Year",
                "final_savings_rate": "Final Savings Rate",
            }.get(x, x),
            help="The metric to analyze sensitivity for",
        )

    with col2:
        variation_pct = st.slider(
            "Variation %",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="How much to vary each parameter (e.g., ±20%)",
        ) / 100

    with col3:
        top_n = st.slider(
            "Top Variables",
            min_value=5,
            max_value=15,
            value=10,
            help="Number of top variables to show",
        )

    # Run Tornado Analysis
    if st.button("Run Tornado Analysis", type="primary"):
        with st.spinner("Running sensitivity analysis... This may take a moment."):
            progress_bar = st.progress(0)

            def update_progress(completed, total):
                progress_bar.progress(completed / total)

            try:
                result = run_sensitivity_analysis(
                    scenario=scenario,
                    target_metric=target_metric,
                    variation_pct=variation_pct,
                    top_n=top_n,
                    progress_callback=update_progress,
                )
                st.session_state.sensitivity_result = result
                progress_bar.empty()
                st.success(f"Analysis complete! Analyzed {len(result.sensitivities)} variables.")
            except Exception as e:
                st.error(f"Error running analysis: {e}")

    # Display results
    if st.session_state.sensitivity_result:
        result = st.session_state.sensitivity_result

        st.divider()

        # Summary
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"**Base {result.metric_name.replace('_', ' ').title()}:** ")
            if result.metric_name == "final_net_worth":
                st.markdown(f"### {format_currency(result.base_value)}")
            elif result.metric_name == "fire_year":
                st.markdown(f"### {int(result.base_value)}")
            else:
                st.markdown(f"### {result.base_value:.2%}")

        with col2:
            # Top impact variable
            if result.sensitivities:
                top_var = result.sensitivities[0]
                st.markdown("**Most Impactful Variable:**")
                st.markdown(f"**{top_var.variable_label}**")

        st.divider()

        # Tornado Chart
        st.subheader("Tornado Chart")
        st.caption("Shows how varying each input affects the outcome. Longer bars = more impact.")
        st.plotly_chart(tornado_chart(result), use_container_width=True)

        # Spider Chart
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Spider Chart")
            st.caption("Relative impact of each variable (normalized to 100%)")
            st.plotly_chart(
                spider_chart(result.sensitivities, result.metric_name),
                use_container_width=True,
            )

        with col2:
            st.subheader("Key Insights")
            st.markdown("**Top 5 Most Impactful Variables:**")

            for i, sens in enumerate(result.sensitivities[:5], 1):
                # Determine direction
                if sens.high_result > sens.low_result:
                    direction = "Higher values increase outcome"
                else:
                    direction = "Higher values decrease outcome"

                if result.metric_name == "final_net_worth":
                    impact_str = format_currency(sens.impact)
                elif result.metric_name == "fire_year":
                    impact_str = f"{sens.impact:.1f} years"
                else:
                    impact_str = f"{sens.impact:.2%}"

                st.markdown(f"{i}. **{sens.variable_label}**: ±{impact_str}")
                st.caption(f"   {direction}")

        # Detailed Table
        with st.expander("View Detailed Results"):
            df = sensitivity_summary_table(result.sensitivities, result.metric_name)
            st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()

    # What-If Analysis Section
    st.subheader("What-If Analysis")
    st.caption("Explore how two variables interact by creating a heatmap of outcomes.")

    # Build dynamic variable list from the actual scenario
    available_vars = []

    # Add income sources with actual names and values
    for i, source in enumerate(scenario.income_sources):
        available_vars.append((
            f"income_sources[{i}].amount",
            f"Income: {source.name} (${source.amount:,.0f}/yr)",
        ))
        # Also add growth rate
        growth_rate = source.growth_rate.mean if hasattr(source.growth_rate, 'mean') else source.growth_rate
        available_vars.append((
            f"income_sources[{i}].growth_rate",
            f"Income Growth: {source.name} ({growth_rate:.1%}/yr)",
        ))

    # Add assets with actual names and values
    for i, asset in enumerate(scenario.assets):
        available_vars.append((
            f"assets[{i}].balance",
            f"Asset: {asset.name} (${asset.balance:,.0f})",
        ))
        exp_return = asset.expected_return.mean if hasattr(asset.expected_return, 'mean') else asset.expected_return
        available_vars.append((
            f"assets[{i}].expected_return",
            f"Return Rate: {asset.name} ({exp_return:.1%}/yr)",
        ))

    # Add expenses with actual names and values
    for i, expense in enumerate(scenario.expenses):
        available_vars.append((
            f"expenses[{i}].amount",
            f"Expense: {expense.name} (${expense.amount:,.0f}/yr)",
        ))

    # Add assumptions
    available_vars.append((
        "assumptions.state_tax_rate",
        f"State Tax Rate ({scenario.assumptions.state_tax_rate:.1%})",
    ))
    general_inflation = scenario.assumptions.get_inflation_rate("general").mean
    available_vars.append((
        "assumptions.inflation_rates.general",
        f"General Inflation Rate ({general_inflation:.1%}/yr)",
    ))

    # Add simulation config
    available_vars.append((
        "simulation_years",
        f"Simulation Years ({scenario.simulation_years} years)",
    ))

    # Initialize analyzer
    analyzer = SensitivityAnalyzer(scenario)

    # Initialize variables with defaults
    var1_min, var1_max, var1_steps = 0.0, 1.0, 5
    var2_min, var2_max, var2_steps = 0.0, 1.0, 5
    can_run_what_if = False

    col1, col2 = st.columns(2)

    with col1:
        var1_idx = st.selectbox(
            "Variable 1 (X-axis)",
            range(len(available_vars)),
            format_func=lambda i: available_vars[i][1],
        )
        var1_path, var1_label = available_vars[var1_idx]

        # Get current value to set default range
        # Tuple format: (resolved_path, current_value, is_distribution, item_name)
        resolved1 = analyzer._resolve_path(var1_path)
        if resolved1:
            base_val1 = resolved1[0][1]  # current_value is at index 1
            if base_val1 != 0:
                var1_min = st.number_input(
                    f"{var1_label} Min",
                    value=float(base_val1 * 0.5),
                    format="%.2f",
                )
                var1_max = st.number_input(
                    f"{var1_label} Max",
                    value=float(base_val1 * 1.5),
                    format="%.2f",
                )
            else:
                var1_min = st.number_input(f"{var1_label} Min", value=0.0, format="%.2f")
                var1_max = st.number_input(f"{var1_label} Max", value=1.0, format="%.2f")
            var1_steps = st.slider("Steps", min_value=3, max_value=10, value=5, key="var1_steps")
            can_run_what_if = True
        else:
            st.warning(f"Could not resolve path: {var1_path}")

    with col2:
        # Default to a different variable than var1 if possible
        default_var2_idx = min(1, len(available_vars) - 1) if len(available_vars) > 1 else 0
        # Try to pick an asset if var1 is income, or vice versa
        if var1_idx < len(available_vars):
            var1_type = available_vars[var1_idx][1].split(":")[0] if ":" in available_vars[var1_idx][1] else ""
            for i, (_, label) in enumerate(available_vars):
                if i != var1_idx and ":" in label and label.split(":")[0] != var1_type:
                    default_var2_idx = i
                    break

        var2_idx = st.selectbox(
            "Variable 2 (Y-axis)",
            range(len(available_vars)),
            format_func=lambda i: available_vars[i][1],
            index=default_var2_idx,
        )
        var2_path, var2_label = available_vars[var2_idx]

        # Tuple format: (resolved_path, current_value, is_distribution, item_name)
        resolved2 = analyzer._resolve_path(var2_path)
        if resolved2:
            base_val2 = resolved2[0][1]  # current_value is at index 1
            if base_val2 != 0:
                var2_min = st.number_input(
                    f"{var2_label} Min",
                    value=float(base_val2 * 0.5),
                    format="%.2f",
                )
                var2_max = st.number_input(
                    f"{var2_label} Max",
                    value=float(base_val2 * 1.5),
                    format="%.2f",
                )
            else:
                var2_min = st.number_input(f"{var2_label} Min", value=0.0, format="%.2f")
                var2_max = st.number_input(f"{var2_label} Max", value=1.0, format="%.2f")
            var2_steps = st.slider("Steps", min_value=3, max_value=10, value=5, key="var2_steps")
        else:
            st.warning(f"Could not resolve path: {var2_path}")
            can_run_what_if = False

    what_if_metric = st.selectbox(
        "What-If Target Metric",
        ["final_net_worth", "fire_year"],
        format_func=lambda x: {
            "final_net_worth": "Final Net Worth",
            "fire_year": "FIRE Year",
        }.get(x, x),
        key="what_if_metric",
    )

    if st.button("Run What-If Analysis", disabled=not can_run_what_if):
        with st.spinner(f"Running what-if analysis ({var1_steps * var2_steps} simulations)..."):
            progress_bar = st.progress(0)

            def update_progress(completed, total):
                progress_bar.progress(completed / total)

            # Extract clean labels for axis (remove the current value in parentheses)
            # "Income: Primary Job ($120,000/yr)" -> "Income: Primary Job"
            def clean_label(label: str) -> str:
                if "(" in label:
                    return label.split("(")[0].strip()
                return label

            try:
                what_if_result = analyzer.run_what_if_matrix(
                    variable1_path=var1_path,
                    variable1_range=(var1_min, var1_max, var1_steps),
                    variable2_path=var2_path,
                    variable2_range=(var2_min, var2_max, var2_steps),
                    target_metric=what_if_metric,
                    progress_callback=update_progress,
                    variable1_label=clean_label(var1_label),
                    variable2_label=clean_label(var2_label),
                )
                st.session_state.what_if_result = what_if_result
                progress_bar.empty()
                st.success("What-if analysis complete!")
            except Exception as e:
                st.error(f"Error: {e}")

    # Display what-if results
    if st.session_state.what_if_result:
        what_if_result = st.session_state.what_if_result

        st.plotly_chart(
            sensitivity_heatmap(what_if_result),
            use_container_width=True,
        )

        st.caption(
            f"Heatmap shows {what_if_result.metric_name.replace('_', ' ').title()} "
            f"for different combinations of {what_if_result.variable1_label} and {what_if_result.variable2_label}. "
            f"Green = better outcomes, Red = worse outcomes."
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Results", "Monte Carlo", "Sensitivity", "Editor", "Comparison"])

    with tab1:
        if st.session_state.simulation_result:
            tab_results(st.session_state.simulation_result)
        else:
            st.info("Load and run a scenario to see results.")

    with tab2:
        tab_monte_carlo()

    with tab3:
        tab_sensitivity()

    with tab4:
        tab_editor()

    with tab5:
        tab_comparison()


if __name__ == "__main__":
    main()

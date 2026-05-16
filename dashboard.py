import streamlit as st
import pandas as pd
import plotly.express as px

# Import the pipeline functions you already wrote
from pipeline import run_pipeline

# ─── 1. PAGE CONFIGURATION ────────────────────────────────────────────────────
st.set_page_config(
    page_title="BAGFLOW Dashboard",
    page_icon="✈️",
    layout="wide"
)

st.title("BAGFLOW: Airport Data Insights")
st.markdown("Interactive dashboard for Vanderlande baggage handling analysis.")

# ─── 2. DATA LOADING ──────────────────────────────────────────────────────────
# The @st.cache_data decorator tells Streamlit to run the pipeline only once
# and store the result in memory.
@st.cache_data
def load():
    return run_pipeline()

df, kpis = load()

# ─── 3. SIDEBAR FILTERS ───────────────────────────────────────────────────────
st.sidebar.header("Filters")

# Convert date column to datetime for the slider to work correctly
df["date"] = pd.to_datetime(df["date"])

# FILTER 1: Date range
min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# FILTER 2: Terminal
# sorted() ensures the options appear in alphabetical order.
all_terminals = sorted(df["terminal"].astype(str).unique())
selected_terminals = st.sidebar.multiselect(
    "Terminal",
    options=all_terminals,
    default=all_terminals
)

# FILTER 3: Zone
all_zones = sorted(df["zone"].astype(str).unique())
selected_zones = st.sidebar.multiselect(
    "Zone",
    options=all_zones,
    default=all_zones
)

# FILTER 4: Process step
all_processes = sorted(df["process"].astype(str).unique())
selected_processes = st.sidebar.multiselect(
    "Process step",
    options=all_processes,
    default=all_processes
)

# ─── APPLY FILTERS TO THE DATAFRAME ──────────────────────────────────────────
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range[0]

filtered_df = df[
    (df["date"].dt.date >= start_date) &
    (df["date"].dt.date <= end_date) &
    (df["terminal"].astype(str).isin(selected_terminals)) &
    (df["zone"].astype(str).isin(selected_zones)) &
    (df["process"].astype(str).isin(selected_processes))
]
# Show the user how many rows are currently visible
st.sidebar.markdown(f"Showing **{len(filtered_df):,}** of **{len(df):,}** events")

# ─── 3. KPI SCORECARDS ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("System Overview")

# Create four equal columns for the four scorecards
col1, col2, col3, col4 = st.columns(4)

# SCORECARD 1: Total unique bags processed
total_bags = filtered_df["bag_id"].nunique()
col1.metric(
    label="Total Bags Processed",
    value=f"{total_bags:,}"
)
# SCORECARD 2: Overall system success rate
success_rate = filtered_df["success_flag"].mean()
col2.metric(
    label="System Success Rate",
    value=f"{success_rate:.1%}"
)
# SCORECARD 3: Jam frequency rate
jam_rate = filtered_df["jam"].astype(int).mean()
col3.metric(
    label="Jam Frequency Rate",
    value=f"{jam_rate:.1%}"
)
# SCORECARD 4: Manual intervention rate
intervention_rate = filtered_df["intervene"].astype(int).mean()
col4.metric(
    label="Manual Intervention Rate",
    value=f"{intervention_rate:.1%}"
)
# ─── 4. VISUALISATION PANELS ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("Detailed Analysis")

# ── ROW 1: EFFICIENCY AND FAILURE ANALYSIS ────────────────────────────────────
st.markdown("#### Process Efficiency and Failure Analysis")
row1_col1, row1_col2 = st.columns(2)

# CHART 1: Throughput trend (line chart)
# Shows the number of bag events per day over the three-month period.
with row1_col1:
    st.markdown("**Bag Throughput Over Time**")
    throughput = (
        filtered_df.groupby("date")["bag_id"]
        .count()
        .reset_index()
        .rename(columns={"bag_id": "event_count"})
    )
    fig1 = px.line(
        throughput,
        x="date",
        y="event_count",
        labels={"date": "Date", "event_count": "Number of Events"},
        template="plotly_white"
    )
    st.plotly_chart(fig1, use_container_width=True)

# CHART 2: Failure distribution by process step (horizontal bar chart)
with row1_col2:
    st.markdown("**Failure Distribution by Process Step**")
    failures = (
        filtered_df[filtered_df["success_flag"] == 0]
        .groupby("process")["bag_id"]
        .count()
        .reset_index()
        .rename(columns={"bag_id": "failure_count", "process": "Process Step"})
        .sort_values("failure_count", ascending=True)
    )
    fig2 = px.bar(
        failures,
        x="failure_count",
        y="Process Step",
        orientation="h",
        labels={"failure_count": "Number of Failures"},
        template="plotly_white",
        color="failure_count",
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── ROW 2: DELAY AND PRIORITY ─────────────────────────────────────────────────
st.markdown("#### Delay Analysis and Priority Bag Handling")
row2_col1, row2_col2 = st.columns(2)

# CHART 3: Delay comparison over time (stacked area chart)
with row2_col1:
    st.markdown("**Average Delays Over Time**")
    delays = (
        filtered_df.groupby("date")
        .agg(
            avg_screen_delay=("screen_delay", "mean"),
            avg_vehicle_delay=("vehicle_delay", "mean")
        )
        .reset_index()
    )
    fig3 = px.area(
        delays,
        x="date",
        y=["avg_screen_delay", "avg_vehicle_delay"],
        labels={
            "date": "Date",
            "value": "Average Delay (min)",
            "variable": "Delay Type"
        },
        template="plotly_white"
    )
    # Rename the legend entries for readability
    fig3.for_each_trace(lambda t: t.update(
        name="Screen Delay" if t.name == "avg_screen_delay" else "Vehicle Delay"
    ))
    st.plotly_chart(fig3, use_container_width=True)

# CHART 4: Priority bag outcomes (donut chart)
with row2_col2:
    st.markdown("**Priority Bag Handling Outcomes**")
    priority_data = filtered_df[filtered_df["priority"] == True]
    priority_outcomes = (
        priority_data.groupby("result")["bag_id"]
        .count()
        .reset_index()
        .rename(columns={"bag_id": "count"})
    )
    # Map boolean result values to readable labels
    priority_outcomes["outcome"] = priority_outcomes["result"].map(
        {True: "Successful", False: "Failed"}
    )
    fig4 = px.pie(
        priority_outcomes,
        values="count",
        names="outcome",
        hole=0.5,
        template="plotly_white",
        color_discrete_sequence=["#2ecc71", "#e74c3c"]
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── ROW 3: EQUIPMENT HEALTH ───────────────────────────────────────────────────
st.markdown("#### Equipment Health Monitoring")

# CHART 5: Sensor anomaly scatter plot
st.markdown("**Sensor Vibration vs Temperature**")
equipment = (
    filtered_df.groupby("sensor")
    .agg(
        avg_vibration=("vibration", "mean"),
        avg_temp=("temp", "mean"),
        avg_speed=("speed", "mean")
    )
    .reset_index()
)
# Fill NaN values in avg_speed with 0 so Plotly can use it as marker size
equipment["avg_speed"] = equipment["avg_speed"].fillna(0)

fig5 = px.scatter(
    equipment,
    x="avg_temp",
    y="avg_vibration",
    size="avg_speed",
    color="sensor",
    labels={
        "avg_temp": "Average Temperature (°C)",
        "avg_vibration": "Average Vibration",
        "avg_speed": "Average Speed",
        "sensor": "Sensor"
    },
    template="plotly_white"
)
st.plotly_chart(fig5, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("BAGFLOW Airport Data Insights | Andrea Menegato | ICT Semester 1 – Project 3")
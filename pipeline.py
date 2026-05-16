import pandas as pd

# ─── STAGE 1: DATA INGESTION ───────────────────────────────────────────────────
def load_data(filepath):
    """
    Loads the raw CSV file and performs initial type parsing.
    Returns a DataFrame ready for cleaning.
    """
    df = pd.read_csv(filepath, sep=";")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%m/%y %H:%M")
    df["date"] = df["timestamp"].dt.date
    return df

# ─── STAGE 2: DATA CLEANING ────────────────────────────────────────────────────
def clean_data(df):
    """
    Handles null values, fixes column types, and removes duplicates.
    Returns a cleaned DataFrame.
    """

    #Remove fully duplicate rows
    before = len(df)
    df = df.drop_duplicates(keep="first")
    after = len(df)
    print(f"Duplicates removed: {before - after}")

    #Convert binary integer columns to boolean (True/False)
    binary_columns = ["jam", "intervene", "priority", "result", "rfid_ok"]
    for col in binary_columns:
        df[col] = df[col].astype(bool)

    #Convert text columns to the "category" type
    category_columns = ["terminal", "zone", "process", "sensor", "flight"]
    for col in category_columns:
        df[col] = df[col].astype("category")

    #Document the remaining NaN values without removing them
    print("Remaining NaN values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    return df

# ─── STAGE 3: DATA PREPROCESSING ──────────────────────────────────────────────
def preprocess_data(df):
    """
    Creates derived columns needed for KPI computation.
    Does not modify existing columns.
    Returns the enriched DataFrame.
    """

    # Step 1: Create a numeric success flag (1 = success, 0 = failure)
    df["success_flag"] = df["result"].astype(int)

    # Step 2: Create a combined delay flag
    df["has_delay"] = df["screen_delay"].notna() | df["vehicle_delay"].notna()

    # Step 3: Create a human-readable priority label
    df["priority_label"] = df["priority"].map(
        {True: "Priority", False: "Standard"}
    )

    # Step 4: Extract the hour from the timestamp as a standalone column
    df["hour_of_day"] = df["timestamp"].dt.hour

    # Step 5: Print a summary to verify the new columns were created
    print("New columns added in preprocessing:")
    new_cols = ["success_flag", "has_delay", "priority_label", "hour_of_day"]
    print(df[new_cols].head(10))
    print("\nDataFrame shape after preprocessing:", df.shape)

    return df

# ─── STAGE 4 + 5: KPI COMPUTATION ─────────────────────────────────────────────
def compute_kpis(df):
    """
    Computes all five KPI groups defined in the SMART objectives.
    Returns a dictionary where each key is a named KPI result.
    """

    kpis = {}

    # ── KPI 1: PROCESS EFFICIENCY ─────────────────────────────────────────────
    # Total number of unique bags processed across the entire dataset
    kpis["total_bags"] = df["bag_id"].nunique()

    # Total number of events recorded per day (bag throughput over time)
    kpis["throughput_by_day"] = (
        df.groupby("date")["bag_id"]
        .count()
        .reset_index()
        .rename(columns={"bag_id": "event_count"})
    )

    # Distribution of events across the four process steps
    kpis["events_by_process"] = (
        df.groupby("process")["bag_id"]
        .count()
        .reset_index()
        .rename(columns={"bag_id": "event_count"})
    )

    # ── KPI 2: RELIABILITY AND FAILURE ANALYSIS ───────────────────────────────
    # Overall system success rate across all events
    kpis["system_success_rate"] = df["success_flag"].mean()

    # RFID read success rate
    kpis["rfid_success_rate"] = df["rfid_ok"].astype(int).mean()

    # Jam frequency and manual intervention rate
    kpis["jam_rate"] = df["jam"].astype(int).mean()
    kpis["intervention_rate"] = df["intervene"].astype(int).mean()

    # Failure count grouped by process step
    kpis["failure_by_process"] = (
        df[df["success_flag"] == 0]
        .groupby("process")["bag_id"]
        .count()
        .reset_index()
        .rename(columns={"bag_id": "failure_count"})
        .sort_values("failure_count", ascending=False)
    )

    # ── KPI 3: DELAY ANALYSIS ─────────────────────────────────────────────────
    # Average screen delay and vehicle delay grouped by date
    kpis["delays_by_day"] = (
        df.groupby("date")
        .agg(
            avg_screen_delay=("screen_delay", "mean"),
            avg_vehicle_delay=("vehicle_delay", "mean")
        )
        .reset_index()
    )
    # Average delays broken down by terminal for comparison
    kpis["delays_by_terminal"] = (
        df.groupby("terminal")
        .agg(
            avg_screen_delay=("screen_delay", "mean"),
            avg_vehicle_delay=("vehicle_delay", "mean")
        )
        .reset_index()
    )
    # ── KPI 4: EQUIPMENT HEALTH MONITORING ───────────────────────────────────
    # Average conveyor speed, vibration, and temperature per sensor
    kpis["equipment_by_sensor"] = (
        df.groupby("sensor")
        .agg(
            avg_speed=("speed", "mean"),
            avg_vibration=("vibration", "mean"),
            avg_temp=("temp", "mean")
        )
        .reset_index()
    )
    # ── KPI 5: PRIORITY BAG HANDLING ─────────────────────────────────────────
    # Success rate for priority bags versus standard bags
    kpis["success_by_priority"] = (
        df.groupby("priority_label")["success_flag"]
        .mean()
        .reset_index()
        .rename(columns={"success_flag": "success_rate"})
    )

    # Distribution of result outcomes for priority bags only
    kpis["priority_results"] = (
        df[df["priority"] == True]
        .groupby("result")["bag_id"]
        .count()
        .reset_index()
        .rename(columns={"bag_id": "count", "result": "outcome"})
    )
    # Print a summary of all computed KPIs
    print("\nKPIs computed successfully:")
    for key, value in kpis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}")
        elif isinstance(value, int):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: DataFrame with {len(value)} rows")

    return kpis

# ─── MAIN FUNCTION ─────────────────────────────────────────────────────────────
def run_pipeline(filepath="data/baggage_handling_dataset.csv"):
    df = load_data(filepath)
    df = clean_data(df)
    df = preprocess_data(df)
    kpis = compute_kpis(df)
    return df, kpis

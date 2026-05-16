import pandas as pd

# --- EXERCISE 1: Loading and start looking at the values
# Load the dataset
df = pd.read_csv("data/baggage_handling_dataset.csv", sep=";")
# Print the first 5 rows
print(df.head())
# How many rows and columns does the dataset have?
print(df.shape)
# What are the column names?
print(df.columns.tolist())
# What data type is each column?
print(df.dtypes)


# --- EXERCISE 2: Cleaning and type conversion ---

# Convert timestamp from text to a real datetime object
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%m/%y %H:%M")

# Verify the conversion worked
print("Timestamp dtype after conversion:", df["timestamp"].dtype)

# Now extract the date only (without the time) into a new column
df["date"] = df["timestamp"].dt.date

# Print a few rows to see the new column
print(df[["timestamp", "date"]].head())

# Check how many NaN values exist in each column
print("\nMissing values per column:")
print(df.isnull().sum())

# --- EXERCISE 3: Filtering and Grouping ---

# FILTERING
# Select only rows where the process is check-in
checkin_df = df[df["process"] == "checkin"]
print("Total check-in events:", len(checkin_df))

# Select only rows where a jam occurred (jam = 1)
jams_df = df[df["jam"] == 1]
print("Total jam events:", len(jams_df))

# Select only priority bags
priority_df = df[df["priority"] == 1]
print("Total priority bag events:", len(priority_df))

# GROUPING
# Count how many events occurred per process step
events_per_process = df.groupby("process")["bag_id"].count()
print("\nEvents per process step:")
print(events_per_process)

# Calculate the average screen_delay per terminal
# We use dropna() to ignore the NaN values we just found
avg_screen_delay = df.dropna(subset=["screen_delay"]).groupby("terminal")["screen_delay"].mean()
print("\nAverage screen delay per terminal:")
print(avg_screen_delay)

# Count how many unique bags exist in the entire dataset
unique_bags = df["bag_id"].nunique()
print("\nTotal unique bags processed:", unique_bags)
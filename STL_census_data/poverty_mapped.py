import pandas as pd
import folium
import os

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define CSV filenames
csv_paths = {
    "poverty": os.path.join(script_dir, "Poverty_Status_Below_1.csv"),
    "income": os.path.join(script_dir, "Median_Household_Income.csv")
}

# Load datasets
df_poverty = pd.read_csv(csv_paths["poverty"])
df_income = pd.read_csv(csv_paths["income"])

# 🛠️ Check if 'GEOID' is correct, or use 'GEOID_Data'
merge_column = "GEOID" if "GEOID" in df_poverty.columns else "GEOID_Data"

# Drop duplicate columns before merging
columns_to_drop = ["Total_Poverty_Status", "Median_Household_Income"]  # Drop from second file
df_income = df_income.drop(columns=columns_to_drop, errors="ignore")

# Merge datasets on 'GEOID' or 'GEOID_Data'
df = df_poverty.merge(df_income, on=merge_column, how="inner")

# Explicitly select the correct latitude and longitude columns
df["INTPTLAT"] = df["INTPTLAT_x"]
df["INTPTLON"] = df["INTPTLON_x"]

# Print merged column names to verify
print("\n✅ Merged DataFrame Columns:")
print(df.columns.tolist())

# Required columns check
required_columns = ["INTPTLAT", "INTPTLON", merge_column, "Total_Poverty_Status", "Median_Household_Income"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"❌ Missing required columns: {', '.join(missing_columns)}")

# Create a map centered around the mean latitude and longitude
map_center = [df["INTPTLAT"].mean(), df["INTPTLON"].mean()]
poverty_map = folium.Map(location=map_center, zoom_start=10)

# Save the map to an HTML file
output_file = os.path.join(script_dir, "poverty_choropleth_map.html")
poverty_map.save(output_file)

print(f"\n🎉 Map has been saved as {output_file}. Open it in a browser to view.")

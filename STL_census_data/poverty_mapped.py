import pandas as pd
import folium
import os
import tkinter as tk
from tkinter import filedialog

# Prompt user to select the CSV file
root = tk.Tk()
root.withdraw()  # Hide the root window
file_path = filedialog.askopenfilename(title="Select CSV file")

if not file_path:
    raise FileNotFoundError("No file selected. Please select the correct CSV file.")

# Ensure the file exists before proceeding
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# Load the CSV file
df = pd.read_csv(file_path)

# Check if required columns exist
required_columns = [
    "INTPTLAT", "INTPTLON", "NAMELSAD", "Total_Poverty_Status", "POV_UNDER_1",
    "Median_Household_Income", "Race_Total", "Race_White", "Race_Black",
    "Race_American_Indian_Alaskan", "Race_Asian", "Race_Native_Hawaiian_Pacific",
    "Race_Other", "Race_2_or_more", "Race_2_or_more_inc_other",
    "Race_2_or_more_ex_other", "Vacant_Housing_Units", "GEOID"
]

missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

# Prompt user to select GeoJSON file
geojson_path = filedialog.askopenfilename(title="Select GeoJSON file for boundaries")

if not geojson_path:
    raise FileNotFoundError("No GeoJSON file selected. Please select the correct file.")

if not os.path.exists(geojson_path):
    raise FileNotFoundError(f"File not found: {geojson_path}")

# Create a map centered around the mean latitude and longitude
map_center = [df["INTPTLAT"].mean(), df["INTPTLON"].mean()]
poverty_map = folium.Map(location=map_center, zoom_start=10)

# Add a choropleth map
folium.Choropleth(
    geo_data=geojson_path,
    name="choropleth",
    data=df,
    columns=["GEOID", "Median_Household_Income"],
    key_on="feature.properties.GEOID",
    fill_color="YlOrRd",  # Yellow to Red gradient
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Median Household Income ($)"
).add_to(poverty_map)

# Add layer control
folium.LayerControl().add_to(poverty_map)

# Save the map to an HTML file
output_file = os.path.join(os.path.dirname(file_path), "poverty_choropleth_map.html")
poverty_map.save(output_file)

print(f"Map has been saved as {output_file}. Open it in a browser to view.")

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import csv

# -----------------------------
# 1. Load and prepare the datasets
# -----------------------------

# Load water sampling data
df_water = pd.read_csv("stl_water.csv")
df_water[['Latitude', 'Longitude']] = df_water['Cordinate (lat/lon)'].str.split(',', expand=True).astype(float)

# Load median household income data
df_income = pd.read_csv("Median_Household_Income.csv")
df_income["INTPTLAT"] = pd.to_numeric(df_income["INTPTLAT"], errors="coerce")
df_income["INTPTLON"] = pd.to_numeric(df_income["INTPTLON"], errors="coerce")

# -----------------------------
# 2. Build a spatial index (KDTree) for income records
# -----------------------------
from scipy.spatial import cKDTree

income_coords = df_income[["INTPTLAT", "INTPTLON"]].values
tree = cKDTree(income_coords)

def get_nearest_income(lat, lon):
    distance, idx = tree.query([lat, lon])
    return df_income.iloc[idx]["Median_Household_Income"]

df_water["Nearest_Median_Income"] = df_water.apply(
    lambda row: get_nearest_income(row["Latitude"], row["Longitude"]), axis=1
)

# -----------------------------
# 3. Prepare the contaminant columns and compute R^2 & P-value
# -----------------------------

elements = [
    "Copper (Cu) 324.752", "Selenium (Se) 196.026", "Silver (Ag) 243.778", "Zinc (Zn) 213.857", 
    "Lithium (Li) 670.784", "Sodium (Na) 589.592", "Potassium (K) 766.490", "Magnesium (Mg) 285.213", 
    "Calcium (Ca) 317.933", "Strontium (Sr) 421.552", "Ceseium (Cs) 455.531", "Barium (Ba) 233.527", 
    "Iron (Fe) 238.204", "Manganese (Mn) 259.372", "Manganese (Mn) 257.610", "Thallium (Tl) 190.801", 
    "Vandium (V) 290.880", "Uranium (U) 385.958", "Selenium (Se) 196.026"
]

for el in elements:
    df_water[el] = pd.to_numeric(df_water[el], errors="coerce")

results = []

for el in elements:
    mask = df_water["Nearest_Median_Income"].notna() & df_water[el].notna()
    x = df_water.loc[mask, "Nearest_Median_Income"]
    y = df_water.loc[mask, el]
    
    if x.empty or y.empty:
        r_squared, p_value = np.nan, np.nan
    else:
        x_log = np.log10(x)
        if len(x_log) > 1:
            r_value, p_value = pearsonr(x_log, y)
            r_squared = r_value ** 2
        else:
            r_squared, p_value = np.nan, np.nan
    
    results.append([el, f"{r_squared:.6f}", f"{p_value:.6f}"])

# Convert results to a DataFrame
df_results = pd.DataFrame(results, columns=["Element", "R^2", "P-value"])

# Save results as CSV with standard quoting for Google Sheets compatibility
df_results.to_csv("contaminant_correlation_results.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

print("CSV file 'contaminant_correlation_results.csv' has been saved and formatted for better compatibility with Google Sheets.")

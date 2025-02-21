import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

# -----------------------------
# 1. Load and prepare the datasets
# -----------------------------

# Load water sampling data and split coordinates
df_water = pd.read_csv("stl_water.csv")
df_water[['Latitude', 'Longitude']] = df_water['Cordinate (lat/lon)'].str.split(',', expand=True).astype(float)

# Load median household income data and ensure coordinate fields are numeric
df_income = pd.read_csv("Median_Household_Income.csv")
df_income["INTPTLAT"] = pd.to_numeric(df_income["INTPTLAT"], errors="coerce")
df_income["INTPTLON"] = pd.to_numeric(df_income["INTPTLON"], errors="coerce")

# -----------------------------
# 2. Build a spatial index (KDTree) for income records
# -----------------------------

# Create a KDTree based on income dataset coordinates
income_coords = df_income[["INTPTLAT", "INTPTLON"]].values
tree = cKDTree(income_coords)

# Define a helper function to find the nearest median income given a water sample point
def get_nearest_income(lat, lon):
    distance, idx = tree.query([lat, lon])
    return df_income.iloc[idx]["Median_Household_Income"]

# For each water sample, attach the nearest median household income
df_water["Nearest_Median_Income"] = df_water.apply(
    lambda row: get_nearest_income(row["Latitude"], row["Longitude"]), axis=1
)

# -----------------------------
# 3. Prepare the contaminant columns
# -----------------------------

# List of contaminant element columns (as in your stl_water.csv)
elements = [
    "Arsenic (As) 193.696",
    "Cadmium (Cd) 214.440",
    "Chromium (Cr) 267.716",
    "Copper (Cu) 324.752",
    "Lead (Pb) 220.353",
    "Nickel (Ni) 231.604",
    "Selenium (Se) 196.026",
    "Silver (Ag) 243.778",
    "Zinc (Zn) 213.857",
    "Lithium (Li) 670.784",
    "Sodium (Na) 589.592",
    "Potassium (K) 766.490",
    "Magnesium (Mg) 285.213",
    "Calcium (Ca) 317.933",
    "Strontium (Sr) 421.552",
    "Beryllium (Be) 313.107",
    "Ceseium (Cs) 455.531",
    "Aluminum (Al) 396.153",
    "Barium (Ba) 233.527",
    "Cobalt (Co) 228.616",
    "Iron (Fe) 238.204",
    "Manganese (Mn) 259.372",
    "Manganese (Mn) 257.610",
    "Gallium (Ga) 417.206",
    "Thallium (Tl) 190.801",
    "Vandium (V) 290.880",
    "Indium (In) 230.606",
    "Uranium (U) 385.958",
    "Uranium (U) 367.007",
    "Uranium (U) 409.014",
    "Uranium (U) 424.167"
]

# Convert each element column to numeric (non-numeric entries like "ND" become NaN)
for el in elements:
    df_water[el] = pd.to_numeric(df_water[el], errors="coerce")

# -----------------------------
# 4. Plotting: Create scatter plots with trendlines for each element,
#    and calculate & display the Pearson correlation coefficient (R)
# -----------------------------

for el in elements:
    plt.figure(figsize=(8, 6))
    
    # Create a mask to filter out missing values in both income and contaminant data
    mask = df_water["Nearest_Median_Income"].notna() & df_water[el].notna()
    x = df_water.loc[mask, "Nearest_Median_Income"]
    y = df_water.loc[mask, el]
    
    # Skip plotting if there are no valid points
    if x.empty or y.empty:
        print(f"Skipping plot for {el} due to insufficient data.")
        continue
    
    # Scatter plot: x = Median Income, y = contaminant level for the element
    plt.scatter(x, y, alpha=0.7, edgecolor='k', label="Sampled + Nearest Income Found")
    
    # Calculate Pearson correlation coefficient (R)
    if len(x) > 1:
        r_value = np.corrcoef(x, y)[0, 1]
    else:
        r_value = np.nan

    # Fit a linear trendline if there are at least two points
    if len(x) > 1:
        coeffs = np.polyfit(x, y, deg=1)
        poly_eqn = np.poly1d(coeffs)
        x_trend = np.linspace(x.min(), x.max(), 100)
        y_trend = poly_eqn(x_trend)
        plt.plot(x_trend, y_trend, color='red', linewidth=2, label="Linear Trend")
    
    # Label the correlation coefficient on the plot
    plt.text(0.05, 0.95, f"R = {r_value:.2f}", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
    
    plt.xlabel("Median Household Income ($ in USD)")
    plt.ylabel(f"{el} (μg/L)")
    plt.title(f"Median Household Income vs {el}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # If you prefer to save each plot instead of displaying, you can uncomment:
    # plt.savefig(f"Income_vs_{el.replace(' ', '_').replace('(', '').replace(')', '')}.png")

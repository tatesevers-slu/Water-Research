import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import seaborn as sns

# -----------------------------
# 1. Load and process the water sampling data
# -----------------------------
df = pd.read_csv("stl_water.csv")
# Split the coordinate string into separate Latitude and Longitude columns
df[['Latitude', 'Longitude']] = df['Cordinate (lat/lon)'].str.split(',', expand=True).astype(float)

elements = [
    "Copper (Cu) 324.752", "Selenium (Se) 196.026", "Silver (Ag) 243.778", "Zinc (Zn) 213.857", 
    "Lithium (Li) 670.784", "Sodium (Na) 589.592", "Potassium (K) 766.490", "Magnesium (Mg) 285.213", 
    "Calcium (Ca) 317.933", "Strontium (Sr) 421.552", "Ceseium (Cs) 455.531", "Barium (Ba) 233.527", 
    "Iron (Fe) 238.204", "Manganese (Mn) 257.610", "Thallium (Tl) 190.801", 
    "Vandium (V) 290.880", "Uranium (U) 385.958"
]

# Convert contaminant columns to numeric (non-numeric values like "ND" become NaN)
for el in elements:
    df[el] = pd.to_numeric(df[el], errors="coerce")

# --- Log-transform the contaminant data using log(x + 1) ---
# This transformation ensures that if x is 0, log(0+1) becomes 0.
for el in elements:
    df[el + " (log)"] = np.log10(df[el] + 1)

# Use these log-transformed columns for PCA
log_elements = [el + " (log)" for el in elements]

# Impute missing values using the column mean (mean of each log-transformed variable)
df_elements = df[log_elements]
df_elements_imputed = df_elements.fillna(df_elements.mean())

# Standardize the log-transformed data before applying PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_elements_imputed)

# -----------------------------
# 2. Perform PCA on the log-transformed water sampling data
# -----------------------------
# Use all components (by not specifying n_components, PCA computes all possible)
pca = PCA()
pca_scores = pca.fit_transform(X_scaled)
pca_columns = [f'PC{i+1}' for i in range(pca_scores.shape[1])]
pca_df = pd.DataFrame(pca_scores, columns=pca_columns, index=df.index)
df = pd.concat([df, pca_df], axis=1)

# Plot the explained variance ratio of each principal component
plt.figure(figsize=(9, 4))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        tick_label=[f'PC{i}' for i in range(1, len(pca.explained_variance_ratio_) + 1)])
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
#plt.title('PCA Explained Variance')
plt.tight_layout()
plt.show()

# Output the PCA loadings (the eigenvectors) for each component
loadings = pd.DataFrame(pca.components_.T,
                        index=df_elements_imputed.columns,
                        columns=[f'PC{i+1}' for i in range(pca.n_components_)])
print("Principal Component Loadings:")
print(loadings)

import re

def simplify_label(label):
    """
    Example function that:
      1. Removes any trailing numbers (e.g., '324.752') and the space before them.
      2. Splits out the symbol (e.g., 'Cu') from something like '(Cu)'.
      3. Retains '(log)' or whatever is left in the last parentheses if present.
      4. Produces something like 'Cu (log)'.
    Adjust the logic to exactly match your naming patterns.
    """
    # e.g., "Copper (Cu) 324.752 (log)" -> remove numeric "324.752":
    label_no_num = re.sub(r"\s\d+(\.\d+)?", "", label)  # remove the numeric portion
    # Example result: "Copper (Cu) (log)"

    # Now split by '(' => e.g. ["Copper ", "Cu) ", "log)"]
    parted = label_no_num.split("(")
    if len(parted) == 3:
        # parted[1] ~ "Cu) ", parted[2] ~ "log)"
        # Extract the symbol & the suffix
        symbol = parted[1].strip(" )")  # e.g. "Cu"
        suffix = parted[2].strip(" )")  # e.g. "log"
        return f"{symbol} ({suffix})"   # e.g. "Cu (log)"
    else:
        # Fallback if the structure doesn't match (optional)
        return label  # or handle differently if you prefer

# After you've created `loadings`, rename its row index:
new_index = [simplify_label(idx) for idx in loadings.index]
loadings.index = new_index

# Now recreate the heatmap with the updated labels:
plt.figure(figsize=(9, 4))
sns.heatmap(loadings, annot=True, cmap="coolwarm", fmt=".2f")
#plt.title("Heatmap of PCA Loadings")
plt.xlabel("Principal Components")
plt.ylabel("Contaminant (log-transformed)")  # row labels are now simplified
plt.tight_layout()
plt.show()

# Additionally, create a scatter plot of loadings for PC1 vs. PC2
plt.figure(figsize=(9, 4))
plt.scatter(loadings['PC1'], loadings['PC2'])
for i, txt in enumerate(loadings.index):
    plt.annotate(txt, (loadings['PC1'][i], loadings['PC2'][i]), fontsize=9)
plt.xlabel('Loading on PC1')
plt.ylabel('Loading on PC2')
#plt.title('Contaminant Loadings: PC1 vs. PC2')
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 3. Load and clean the median income data; assign the nearest income via KDTree
# -----------------------------
income_df = pd.read_csv("Median_Household_Income.csv")
income_df["INTPTLAT"] = pd.to_numeric(income_df["INTPTLAT"], errors="coerce")
income_df["INTPTLON"] = pd.to_numeric(income_df["INTPTLON"], errors="coerce")
# Clean and convert the median income column (remove commas or dollar signs)
income_df["Median_Household_Income"] = income_df["Median_Household_Income"].replace('[\$,]', '', regex=True)
income_df["Median_Household_Income"] = pd.to_numeric(income_df["Median_Household_Income"], errors="coerce")

# Build a KDTree using the income data coordinates
income_coords = income_df[["INTPTLAT", "INTPTLON"]].values
tree = cKDTree(income_coords)

def get_nearest_income(lat, lon):
    distance, idx = tree.query([lat, lon])
    return income_df.iloc[idx]["Median_Household_Income"]

# Attach the nearest median household income to each water sample
df["Nearest_Median_Income"] = df.apply(lambda row: get_nearest_income(row["Latitude"], row["Longitude"]), axis=1)
print("Nearest Median Income (first 5):")
print(df["Nearest_Median_Income"].head())

# --- Log-transform the socioeconomic indicator using log(x+1) ---
df["log_Median_Income"] = np.log10(df["Nearest_Median_Income"] + 1)

# -----------------------------
# 4. Regress log-transformed median household income on a subset of PCA components
# -----------------------------
# For example, if you want to use the first 5 principal components
df_reg = df.dropna(subset=["log_Median_Income", "PC1", "PC2", "PC3", "PC4", "PC5"])
print("Number of observations for regression:", df_reg.shape[0])
X_reg = df_reg[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]
X_reg = sm.add_constant(X_reg)
y = df_reg['log_Median_Income']

model = sm.OLS(y, X_reg).fit()
print(model.summary())

# Plot the cumulative explained variance by all PCA components
plt.figure(figsize=(9, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
#plt.title('Cumulative Explained Variance by PCA Components')
plt.grid(True)
plt.show()

plt.figure(plt.figure(figsize=(9, 4)))
plt.scatter(pca_df['PC1'], pca_df['PC2'], cmap='plasma', edgecolor='k', s=60)
plt.xlabel('PC1')
plt.ylabel('PC2')
#plt.title('Scatter Plot of Scores: PC1 vs PC2')
plt.show()

pcs = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
print("\nCorrelation (and R²) between each PC and log(Median_Income):")
for pc in pcs:
    r = np.corrcoef(df_reg[pc], df_reg['log_Median_Income'])[0, 1]
    R2 = r**2
    print(f"{pc}: r = {r:.3f}, R² = {R2:.3f}")

from scipy.stats import pearsonr

# …

pcs = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
print("\nCorrelation (r), R² and p‑value between each PC and log(Median_Income):")
for pc in pcs:
    # dropna so that length matches
    valid = df_reg[[pc, 'log_Median_Income']].dropna()
    r, p = pearsonr(valid[pc], valid['log_Median_Income'])
    R2 = r**2
    print(f"{pc}: r = {r:.3f}, R² = {R2:.3f}, p = {p:.3f}")

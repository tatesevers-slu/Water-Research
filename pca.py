import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import seaborn as sns

# load and process the water sampling data
df = pd.read_csv("stl_water.csv")
df[['Latitude', 'Longitude']] = df['Cordinate (lat/lon)'].str.split(',', expand=True).astype(float)

elements = [
    "Copper (Cu) 324.752", "Selenium (Se) 196.026", "Silver (Ag) 243.778", "Zinc (Zn) 213.857", 
    "Lithium (Li) 670.784", "Sodium (Na) 589.592", "Potassium (K) 766.490", "Magnesium (Mg) 285.213", 
    "Calcium (Ca) 317.933", "Strontium (Sr) 421.552", "Ceseium (Cs) 455.531", "Barium (Ba) 233.527", 
    "Iron (Fe) 238.204", "Manganese (Mn) 259.372", "Manganese (Mn) 257.610", "Thallium (Tl) 190.801", 
    "Vandium (V) 290.880", "Uranium (U) 385.958"
]

# convert contaminant columns to numeric (non-numeric "ND" become NaN)
for el in elements:
    df[el] = pd.to_numeric(df[el], errors="coerce")

# log transform of contaminant
# Add a small constant to avoid taking log(0)
for el in elements:
    df[el + "_log"] = np.log10(df[el] + 1e-6)

log_elements = [el + "_log" for el in elements]

# impute missing values using the column mean (mean of each log-transformed variable)
df_elements = df[log_elements]
df_elements_imputed = df_elements.fillna(df_elements.mean())

# standardize the log-transformed data before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_elements_imputed)

# PCA on the log-transformed water sampling data
pca = PCA(n_components=5)
pca_scores = pca.fit_transform(X_scaled)
pca_columns = [f'PC{i+1}' for i in range(pca_scores.shape[1])]
pca_df = pd.DataFrame(pca_scores, columns=pca_columns, index=df.index)
df = pd.concat([df, pca_df], axis=1)

# plot the explained variance ratio of each principal component
plt.figure(figsize=(6, 4))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        tick_label=[f'PC{i}' for i in range(1, len(pca.explained_variance_ratio_) + 1)])
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.tight_layout()
plt.show()

# output the PCA loadings (how each original log-transformed element contributes to each PC)
loadings = pd.DataFrame(pca.components_.T, 
                        index=df_elements_imputed.columns, 
                        columns=[f'PC{i+1}' for i in range(pca.n_components_)])
print("Principal Component Loadings:")
print(loadings)

# create a heatmap of the loadings
plt.figure(figsize=(10, 8))
sns.heatmap(loadings, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap of PCA Loadings")
plt.xlabel("Principal Components")
plt.ylabel("Contaminant (log-transformed)")
plt.show()

# load and clean the median income data; assign the nearest income via KDTree
income_df = pd.read_csv("Median_Household_Income.csv")
income_df["INTPTLAT"] = pd.to_numeric(income_df["INTPTLAT"], errors="coerce")
income_df["INTPTLON"] = pd.to_numeric(income_df["INTPTLON"], errors="coerce")
# Clean and convert the median income column (remove commas or dollar signs if present)
income_df["Median_Household_Income"] = income_df["Median_Household_Income"].replace('[\$,]', '', regex=True)
income_df["Median_Household_Income"] = pd.to_numeric(income_df["Median_Household_Income"], errors="coerce")

# Build a KDTree using the income data coordinates
income_coords = income_df[["INTPTLAT", "INTPTLON"]].values
tree = cKDTree(income_coords)

def get_nearest_income(lat, lon):
    distance, idx = tree.query([lat, lon])
    return income_df.iloc[idx]["Median_Household_Income"]

# for each water sample, attach the nearest median household income
df["Nearest_Median_Income"] = df.apply(lambda row: get_nearest_income(row["Latitude"], row["Longitude"]), axis=1)
print("Nearest Median Income (first 5):")
print(df["Nearest_Median_Income"].head())

# log-transform the socioeconomic indicator
df["log_Median_Income"] = np.log10(df["Nearest_Median_Income"] + 1e-6)

# regress log-transformed median household income on the first six PCA components
# drop rows with missing values in the required columns
df_reg = df.dropna(subset=["log_Median_Income", "PC1", "PC2", "PC3", "PC4", "PC5"])
print("Number of observations for regression:", df_reg.shape[0])

X_reg = df_reg[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]
X_reg = sm.add_constant(X_reg)
y = df_reg['log_Median_Income']

model = sm.OLS(y, X_reg).fit()
print(model.summary())

# cumulative explained variance by PCA components
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.grid(True)
plt.show()

# significant Principal Components and Their Key Loadings
# Remove the constant and extract p-values for just the PCs
pvals = model.pvalues.drop('const')
significant_threshold = 0.05
significant_pcs = pvals[pvals < significant_threshold].index

print("Significant Principal Components (p < 0.05):")
print(pvals[pvals < significant_threshold])

# for each significant PC, print the loadings sorted by absolute value,
# so you can see which original contaminants drive the relationship.
for pc in significant_pcs:
    print(f"\nLoadings for {pc} (sorted by absolute value):")
    loadings_sorted = loadings[pc].abs().sort_values(ascending=False)
    # Print the original loadings with their signs, sorted in order of absolute magnitude.
    loadings_with_sign = loadings.loc[loadings_sorted.index, pc]
    print(loadings_with_sign)

# Task 1: Loading and exploring dataset
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Load dataset with error handling
try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display first few rows
print(df.head())

# Check data types and missing values
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())

# Clean dataset (no missing values, but included for robustness)
df_cleaned = df.dropna()

# Task 2: Basic statistics
print("\nDescriptive Statistics:\n", df_cleaned.describe())

# Grouping by species and computing mean of numerical columns
grouped_means = df_cleaned.groupby('species').mean()
print("\nMean Values by Species:\n", grouped_means)

# Observations
for feature in df_cleaned.columns[:-1]:
    max_species = grouped_means[feature].idxmax()
    print(f"{feature.title()} is highest in {max_species}.")

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Line chart: Simulated time-series of sepal length
df_cleaned['date'] = pd.date_range(start='2023-01-01', periods=len(df_cleaned), freq='D')
plt.figure(figsize=(10, 5))
for species in df_cleaned['species'].unique():
    subset = df_cleaned[df_cleaned['species'] == species]
    plt.plot(subset['date'], subset['sepal length (cm)'], label=species)
plt.title('Sepal Length Over Time by Species')
plt.xlabel('Date')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart: Average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x=grouped_means.index, y=grouped_means['petal length (cm)'])
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()

# Histogram: Sepal width distribution
plt.figure(figsize=(6, 4))
sns.histplot(df_cleaned['sepal width (cm)'], kde=True, bins=20, color='skyblue')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df_cleaned, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()


# My finding and observations
# Setosa has the smallest petal dimensions and widest sepal width.
# Virginica shows the largest petal length and width.
# Versicolor lies between the two in most features.
# Strong correlation between sepal and petal length, especially in Virginica and Versicolor.
# Sepal width has a wider distribution, especially for Setosa.

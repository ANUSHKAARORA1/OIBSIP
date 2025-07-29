import pandas as pd

# Load the dataset
df = pd.read_csv("AB_NYC_2019.csv")

# 1. Show basic info
print("Original Data:")
print(df.info())

# 2. Drop duplicate rows 
df = df.drop_duplicates()

# 3. Check for missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# 4. Handle missing values

df['name'] = df['name'].fillna("No name")  # fill missing listing names
df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
df = df.dropna(subset=['host_name', 'last_review'])  # drop rows missing critical info

# 5. Standardize column names 
df.columns = df.columns.str.lower().str.replace(' ', '_')

# 6. Detect and handle outliers in price

upper_limit = df['price'].quantile(0.99)
df = df[df['price'] <= upper_limit]

# 7. Save cleaned dataset
df.to_csv("cleaned_AB_NYC_2019.csv", index=False)

print("\nCleaned Data:")
print(df.info())
print("\nFirst 5 rows of cleaned data:")
print(df.head())

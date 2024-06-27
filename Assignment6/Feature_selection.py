import pandas as pd
import numpy as np

# Create a sample dataset
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': ['a', 'b', 'a', 'b', 'c'],
    'C': [10, 15, np.nan, 25, 30],
    'D': [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Data cleaning: Drop any completely empty columns (not applicable here, but included for completeness)
df.dropna(axis=1, how='all', inplace=True)

# Handling missing values
# Fill missing values in numerical columns with the median
df['A'].fillna(df['A'].median(), inplace=True)
df['C'].fillna(df['C'].median(), inplace=True)

# Encoding categorical variables
df['B'] = df['B'].astype('category').cat.codes

# Feature engineering: Create new features
# Create a new feature based on column A and C (example: sum of A and C)
df['A_plus_C'] = df['A'] + df['C']
# Create a new binary feature based on column D
df['D_binary'] = df['D'].apply(lambda x: 1 if x > 0 else 0)

# Normalization: Min-Max scaling for columns A and C
df['A'] = (df['A'] - df['A'].min()) / (df['A'].max() - df['A'].min())
df['C'] = (df['C'] - df['C'].min()) / (df['C'].max() - df['C'].min())

# Display the first few rows of the preprocessed dataset
print(df.head())
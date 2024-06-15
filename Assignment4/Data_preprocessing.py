import pandas as pd
import numpy as np

# Sample data creation
data = {
    'age': [25, np.nan, 35, 45, np.nan, 50],
    'salary': [50000, 60000, np.nan, 80000, 90000, np.nan],
    'city': ['New York', 'Los Angeles', 'Chicago', np.nan, 'Houston', 'Phoenix']
}
df = pd.DataFrame(data)

# Handling missing values
# Fill missing values in 'age' with the mean of the column
df['age'].fillna(df['age'].mean(), inplace=True)

# Fill missing values in 'salary' with the mean of the column
df['salary'].fillna(df['salary'].mean(), inplace=True)

# Fill missing values in 'city' with the mode of the column
df['city'].fillna(df['city'].mode()[0], inplace=True)

# Encoding categorical variables
# Create a dictionary for city encoding
city_dict = {city: i for i, city in enumerate(df['city'].unique())}
df['city'] = df['city'].map(city_dict)

# Normalizing numerical features
# Min-Max normalization
df['age'] = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
df['salary'] = (df['salary'] - df['salary'].min()) / (df['salary'].max() - df['salary'].min())

print(df)

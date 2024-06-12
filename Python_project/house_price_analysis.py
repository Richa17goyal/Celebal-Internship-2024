import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate random data for demonstration
np.random.seed(0)
num_samples = 100
area = np.random.randint(500, 3000, num_samples)  # House area in square feet
bedrooms = np.random.randint(1, 6, num_samples)   # Number of bedrooms
bathrooms = np.random.randint(1, 4, num_samples)  # Number of bathrooms
age = np.random.randint(1, 50, num_samples)       # Age of the house in years
price = 100 * area + 5000 * bedrooms + 7000 * bathrooms - 300 * age + np.random.normal(0, 10000, num_samples)

# Create DataFrame
data = pd.DataFrame({
    'Area': area,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Age': age,
    'Price': price
})

# Basic data analysis
print("Data Summary:")
print(data.describe())

# Correlation analysis
correlation = data.corr()
print("\nCorrelation Matrix:")
print(correlation)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(data['Area'], data['Price'], alpha=0.7)
plt.title('House Price vs Area')
plt.xlabel('Area (sqft)')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(data['Age'], data['Price'], alpha=0.7)
plt.title('House Price vs Age')
plt.xlabel('Age (years)')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()

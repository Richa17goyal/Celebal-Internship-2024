import pandas as pd
import numpy as np

# Load dataset
url = 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
df = pd.read_csv(url)

# Data preprocessing
# Handling missing values (if any)
df.fillna(df.median(), inplace=True)

# Normalization: Min-Max scaling for features
def min_max_scaling(column):
    return (column - column.min()) / (column.max() - column.min())

for column in df.columns:
    df[column] = min_max_scaling(df[column])

# Splitting data into features (X) and target (y)
X = df.drop(['medv'], axis=1).values  # Features
y = df['medv'].values  # Target variable

# Adding intercept term for linear regression
X = np.c_[np.ones(X.shape[0]), X]

# Splitting the data into training and test sets
split_ratio = 0.8
split_index = int(split_ratio * len(y))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Linear regression using Normal Equation
def normal_equation(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Train the model
theta = normal_equation(X_train, y_train)

# Predictions
def predict(X, theta):
    return X @ theta

y_pred_train = predict(X_train, theta)
y_pred_test = predict(X_test, theta)

# Calculate Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f'Training Mean Squared Error: {mse_train}')
print(f'Test Mean Squared Error: {mse_test}')
import pandas as pd
import numpy as np

# Sample dataset
data = {
    'Gender': ['Male', 'Female', 'Male', 'Male', 'Female'],
    'Married': ['No', 'Yes', 'Yes', 'Yes', 'No'],
    'ApplicantIncome': [5000, 3000, 4000, 6000, 3500],
    'LoanAmount': [200, 150, 180, 210, 130],
    'Loan_Status': ['Y', 'N', 'Y', 'Y', 'N']
}
df = pd.DataFrame(data)

# Preprocessing
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Features and target
X = df.drop('Loan_Status', axis=1).values
y = df['Loan_Status'].values

# Add intercept term
X = np.c_[np.ones(X.shape[0]), X]

# Logistic Regression using Gradient Descent
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    return -1/len(y) * (y @ np.log(h) + (1 - y) @ np.log(1 - h))

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        theta -= alpha/m * X.T @ (sigmoid(X @ theta) - y)
    return theta

# Initialize parameters
theta = np.zeros(X.shape[1])
alpha = 0.01
iterations = 1000

# Train model
theta = gradient_descent(X, y, theta, alpha, iterations)

# Prediction
def predict(X, theta):
    return sigmoid(X @ theta) >= 0.5

# Predict on training data
predictions = predict(X, theta)
accuracy = np.mean(predictions == y)

print(f'Accuracy: {accuracy * 100}%')

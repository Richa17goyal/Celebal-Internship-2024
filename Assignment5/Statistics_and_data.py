import pandas as pd
import numpy as np

# Load dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Data cleaning: Dropping unnecessary columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Handling missing values
# Fill missing Age values with median
df['Age'].fillna(df['Age'].median(), inplace=True)
# Fill missing Embarked values with the mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
# Fill missing Fare values with median
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Encoding categorical variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)

# Feature engineering: Create new features
# Create FamilySize feature
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# Create IsAlone feature
df['IsAlone'] = 1  # Initialize to 1 (True)
df['IsAlone'].loc[df['FamilySize'] > 1] = 0  # If FamilySize > 1, set to 0 (False)

# Normalization: Min-Max scaling for Age and Fare
df['Age'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())
df['Fare'] = (df['Fare'] - df['Fare'].min()) / (df['Fare'].max() - df['Fare'].min())

# Drop original SibSp and Parch as they are not needed anymore
df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

# Display the first few rows of the preprocessed dataset
print(df.head())

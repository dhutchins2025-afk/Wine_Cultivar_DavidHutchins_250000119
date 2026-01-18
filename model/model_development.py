

 # Load required packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from scipy.stats import ttest_ind

# Data preprocessing
from sklearn.preprocessing import StandardScaler

# Model Training --- Splitting the dataset
from sklearn.model_selection import train_test_split

 # Load the wine dataset from sklearn
data = load_wine()
wine_df = pd.DataFrame(data.data, columns=data.feature_names)
wine_df['target'] = data.target # Add target column for visualization

 # Data Description
print(wine_df.info())  # Structure
print(wine_df.shape[0])  # Number of rows
print(wine_df.shape[1])  # Number of columns
print(wine_df.columns)  # Names of the columns

# Target (class) distribution
print(wine_df['target'].value_counts()
      )
# Summary statistics of 'alcohol' content
print(wine_df['alcohol'].describe())

# Quick bar plot of alcohol content
plt.figure(figsize=(20, 6))
sns.countplot(x='alcohol', data=wine_df)
plt.title('Distribution of Alcohol Content')
plt.show()

 # Histogram of 'alcohol' content
plt.figure(figsize=(10, 6))
sns.histplot(wine_df['alcohol'], bins=30, kde=True) 
plt.title('Histogram of Alcohol Content')
plt.xlabel('Alcohol Content (%)')
plt.show()

# Boxplot of 'alcohol' content
plt.figure(figsize=(8, 6))
sns.boxplot(x=wine_df['alcohol'])
plt.title('Boxplot of Alcohol Content')
plt.xlabel('Alcohol Content (%)')
plt.show()

# Bar plot of 'target' (class)
plt.figure(figsize=(10, 6))
sns.countplot(x='target', data=wine_df)
plt.title('Distribution of Wine Classes')
plt.xlabel('Wine Class')
plt.show()

# Create a two-way table
pd.crosstab(wine_df['alcohol'], wine_df['target'])

 # Scatter plot of 'alcohol' content vs 'flavanoids' with target classes
plt.figure(figsize=(10, 6))
sns.scatterplot(x='alcohol', y='flavanoids', hue='target', data=wine_df)
plt.title('Alcohol Content vs flavanoids with Target Classes')
plt.xlabel('Alcohol Content (%)')
plt.ylabel('flavanoids')
plt.show()

# T-test to compare 'alcohol' content between different target classes
class_0_alcohol = wine_df.loc[wine_df['target'] == 0, 'alcohol']
class_1_alcohol = wine_df.loc[wine_df['target'] == 1, 'alcohol']

t_stat, p_value = ttest_ind(class_0_alcohol, class_1_alcohol, 
alternative='two-sided', equal_var=False)
print("T-test (two-sided) p-value:", p_value)

 # Take a preview
print(wine_df.head())

# Data Exploration
print(wine_df.describe())

 # Split data into features and label 
features = wine_df[data.feature_names].copy()
labels = wine_df["target"].copy()

# Instantiate scaler and fit on features
scaler = StandardScaler()
scaler.fit(features)

# Transform features
X_scaled = scaler.transform(features.values)

# Split data into train and test
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, labels,train_size=.7,random_state=0)

# Building the model
from sklearn.linear_model import LogisticRegression

# Initializing the model 
logistic_regression = LogisticRegression()

# Training the models
logistic_regression.fit(X_train_scaled, y_train)

# Making predictions with the model
log_reg_preds = logistic_regression.predict(X_test_scaled)

# Evaluate the model
from sklearn.metrics import classification_report
print(classification_report(y_test, log_reg_preds))

# Initializing the model 
from sklearn.svm import SVC
svm= SVC()

# Training the models
svm.fit(X_train_scaled, y_train)

# Making predictions with the model
svm_preds = svm.predict(X_test_scaled)

# Evaluate the model
print(classification_report(y_test, svm_preds))

# Initializing the model 
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)

# Training the models
tree.fit(X_train_scaled, y_train)

# Making predictions with the model
tree_preds = tree.predict(X_test_scaled)

# Evaluate the model
print(classification_report(y_test, tree_preds))

from sklearn.metrics import accuracy_score

models = {
    "Logistic Regression": log_reg_preds,
    "SVM": svm_preds,
    "Decision Tree": tree_preds
}

for name, preds in models.items():
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")


import joblib

# Save the Logistic Regression model
joblib.dump(logistic_regression, "best_wine_model.pkl")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# --- To load them later ---
# loaded_scaler = joblib.load("model/scaler.pkl")
# loaded_model = joblib.load("model/best_wine_model.pkl")

# Example: Making predictions on new data
# new_data_scaled = loaded_scaler.transform(new_data)
# predictions = loaded_model.predict(new_data_scaled




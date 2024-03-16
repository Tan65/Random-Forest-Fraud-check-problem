# Random-Forest-Fraud-check-problem
Use Random Forest to prepare a model on fraud data  treating those who have taxable_income &lt;= 30000 as "Risky" and others are "Good"
# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb

# Step 1: Read the data
df = pd.read_csv('Fraud_check.csv')

# Step 2: Define the threshold and create the target variable 'Risk'
threshold = 30000
df['Risk'] = df['Taxable.Income'].apply(lambda x: 'Risky' if x <= threshold else 'Good')

# Step 3: Drop unnecessary columns and convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Undergrad', 'Marital.Status', 'Urban'])
df.drop('Taxable.Income', axis=1, inplace=True)  # Drop the Taxable.Income column

# Step 4: Exploratory Data Analysis (EDA)
# Visualizations for entropy and Gini impurity can be added here

# Step 5: Split the data into training and testing sets
X = df.drop('Risk', axis=1)
y = df['Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the models
# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# AdaBoost
ada_model = AdaBoostClassifier(random_state=42)
ada_model.fit(X_train, y_train)

# LightGBM
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)

from sklearn.preprocessing import LabelEncoder

# Encode target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Initialize and fit the XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train_encoded)

# Step 7: Make predictions and evaluate the models
# Random Forest
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)

# AdaBoost
ada_pred = ada_model.predict(X_test)
ada_accuracy = accuracy_score(y_test, ada_pred)
print("AdaBoost Accuracy:", ada_accuracy)

# LightGBM
lgb_pred = lgb_model.predict(X_test)
lgb_accuracy = accuracy_score(y_test, lgb_pred)
print("LightGBM Accuracy:", lgb_accuracy)

# XGBoost
xgb_pred = xgb_model.predict(X_test)
from sklearn.preprocessing import LabelEncoder

# Encode predicted labels
label_encoder = LabelEncoder()
xgb_pred_encoded = label_encoder.fit_transform(xgb_pred)

# Compute accuracy score
xgb_accuracy = accuracy_score(y_test_encoded, xgb_pred_encoded)
print("XGBoost Accuracy:", xgb_accuracy)

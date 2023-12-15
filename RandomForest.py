import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from module import clean, enrich, PrincipalComponentAnalysis, unbiased_RT, lab_bias_df

# Load data
TrainingData = pd.read_csv('train.csv')
TrainingData = clean(TrainingData)
TrainingData = enrich(TrainingData)
TrainingData = PrincipalComponentAnalysis(TrainingData, 31, 'RT')

# Split data into training and testing sets
train_data = TrainingData.iloc[:1000]
test_data = TrainingData.iloc[1001:]

# Train Random Forest regression model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
X_train = train_data.drop(['Compound', 'SMILES', 'Lab', 'RT'], axis=1)
y_train = train_data['RT']
rf.fit(X_train, y_train)

# Predict on the test set
X_test = test_data.drop(['Compound', 'SMILES', 'Lab', 'RT'], axis=1)
y_true = test_data['RT']
y_pred = rf.predict(X_test)

# Evaluate model performance using mean squared error
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {mse}")

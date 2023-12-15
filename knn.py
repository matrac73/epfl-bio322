import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from module import (
    clean,
    enrich,
    PrincipalComponentAnalysis
)


# Load data
TrainingData = pd.read_csv('train.csv')
TrainingData = clean(TrainingData)
TrainingData = enrich(TrainingData)
TrainingData = PrincipalComponentAnalysis(TrainingData, 31, 'RT')

# Split data into training and testing sets
train_data = TrainingData.iloc[:1000]
test_data = TrainingData.iloc[1001:]

# Train k-nearest neighbors regression model
knn = KNeighborsRegressor(n_neighbors=4)
X_train = train_data.drop(['Compound', 'SMILES', 'Lab', 'RT'], axis=1)
y_train = train_data['RT']
knn.fit(X_train, y_train)

# Predict on the test set
X_test = test_data.drop(['Compound', 'SMILES', 'Lab', 'RT'], axis=1)
y_true = test_data['RT']
y_pred = knn.predict(X_test)

# Evaluate model performance using mean squared error
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse}")



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Simulate a dataset
np.random.seed(0)
data = {
    'timestamp': pd.date_range(start='2023-01-01', periods=1000, freq='H'),
    'temperature': np.random.normal(loc=70, scale=10, size=1000),
    'vibration': np.random.normal(loc=0.5, scale=0.2, size=1000),
    'failure': np.random.binomial(1, 0.1, 1000)
}
df = pd.DataFrame(data)

# Feature engineering
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['rolling_temp_mean'] = df['temperature'].rolling(window=24).mean().fillna(df['temperature'].mean())
df['rolling_vibration_mean'] = df['vibration'].rolling(window=24).mean().fillna(df['vibration'].mean())

# Prepare features and labels
X = df[['temperature', 'vibration', 'hour', 'day_of_week', 'rolling_temp_mean', 'rolling_vibration_mean']]
y = df['failure']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Simulate new data
new_data = {
    'temperature': [75],
    'vibration': [0.6],
    'hour': [14],
    'day_of_week': [2],
    'rolling_temp_mean': [73],
    'rolling_vibration_mean': [0.55]
}
new_df = pd.DataFrame(new_data)
prediction = model.predict(new_df)
print(f'Predicted failure: {"Yes" if prediction[0] == 1 else "No"}')

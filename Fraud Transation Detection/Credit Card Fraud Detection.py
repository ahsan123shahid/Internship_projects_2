import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler  # Faster than SMOTE

# Load the data (use only 30% of the dataset for speed)
data = pd.read_csv('creditcard.csv').sample(frac=0.3, random_state=42)

# Scale 'Amount' and 'Time' features
scaler = StandardScaler()
data[['Time', 'Amount']] = scaler.fit_transform(data[['Time', 'Amount']])

# Split the data into features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Undersampling to balance classes (much faster than SMOTE)
rus = RandomUnderSampler(sampling_strategy=0.2, random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Train the RandomForestClassifier (fewer trees, limited depth)
model = RandomForestClassifier(n_estimators=50, max_depth=8, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
print("Precision Score:", precision_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Recall Score:", recall_score(y_test, y_pred))

# Example: New transaction data
new_transaction = {
    'Time': [100000],
    'V1': [1.23], 'V2': [-0.45], 'V3': [0.67], 'V4': [-1.89], 'V5': [0.12],
    'V6': [-0.34], 'V7': [0.56], 'V8': [-0.78], 'V9': [0.91], 'V10': [-0.12],
    'V11': [0.34], 'V12': [-0.56], 'V13': [0.78], 'V14': [-0.91], 'V15': [0.12],
    'V16': [-0.34], 'V17': [0.56], 'V18': [-0.78], 'V19': [0.91], 'V20': [-0.12],
    'V21': [0.34], 'V22': [-0.56], 'V23': [0.78], 'V24': [-0.91], 'V25': [0.12],
    'V26': [-0.34], 'V27': [0.56], 'V28': [-0.78], 'Amount': [100.0]
}

# Convert new transaction into DataFrame and scale features
new_data = pd.DataFrame(new_transaction)
new_data[['Time', 'Amount']] = scaler.transform(new_data[['Time', 'Amount']])

# Make the prediction
prediction = model.predict(new_data)

# Interpret the result
print("The transaction is predicted to be:", "FRAUD" if prediction[0] == 1 else "NON-FRAUD")

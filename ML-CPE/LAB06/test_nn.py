import joblib
from data_loader import load_data
from split_data import split_dataset
from preprocessing import standardize_data

X, y, y_min, _ = load_data('winequality-red.csv.csv')
X_train, X_test, y_train, y_test = split_dataset(X, y)
_, X_test_scaled, _ = standardize_data(X_train, X_test)

model = joblib.load("nn_model.joblib")

print("\n--- Predictions on the Selected Dataset ---")
sample_indices = [0, 10, 20, 30]
X_sample = X_test_scaled[sample_indices]
y_true_sample = y_test[sample_indices] + y_min

y_pred = model.predict(X_sample) + y_min

for i, idx in enumerate(sample_indices):
    print(f"Sample {i+1}: True Quality = {y_true_sample[i]}, Predicted Quality = {y_pred[i]}")
    if y_true_sample[i] == y_pred[i]:
        print("  -> Result: CORRECT ✅")
    else:
        print("  -> Result: INCORRECT ❌")
import joblib
from sklearn.metrics import accuracy_score
from data_loader import load_data
from split_data import split_dataset
from preprocessing import standardize_data
from nn_model import build_model
from evaluate import show_results

X, y, _, _ = load_data('winequality-red.csv.csv')
X_train, X_test, y_train, y_test = split_dataset(X, y)
X_train_scaled, X_test_scaled, _ = standardize_data(X_train, X_test)

configs = ["Config_1", "Config_2"]
epochs_list = [20, 50]
results = []
best_acc = 0
best_model = None

for config in configs:
    for epochs in epochs_list:
        model = build_model(config, epochs)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results.append({"Configuration": config, "Epochs": epochs, "Accuracy": acc})
        
        if acc > best_acc:
            best_acc = acc
            best_model = model

show_results(results)
joblib.dump(best_model, "nn_model.joblib")
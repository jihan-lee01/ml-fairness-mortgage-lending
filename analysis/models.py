import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

def load_data():
    X_train = pd.read_csv('.../preprocessed_X_train.csv')
    X_test = pd.read_csv('.../preprocessed_X_test.csv')
    y_train = pd.read_csv('.../preprocessed_y_train.csv')
    y_test = pd.read_csv('.../preprocessed_y_test.csv')

    # clean column names to be compatible with XGBoost
    X_train.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_train.columns]
    X_test.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_test.columns]

    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()

# Train model
def train_model(X_train, y_train, model_type='rf'):
    if model_type == 'rf':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'xgb':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None, labels=[1, 0])  # 1: approved, 0: denied
    recall = recall_score(y_test, y_pred, average=None, labels=[1, 0])
    print(f'Accuracy: {accuracy}')
    print(f'Precision (Approved): {precision[0]}, Precision (Denied): {precision[1]}')
    print(f'Recall (Approved): {recall[0]}, Recall (Denied): {recall[1]}')
    return y_pred

# Plot feature importance and print top 20 features
def plot_feature_importance(model, X_train, model_type='rf'):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # top 20 features
    plt.figure(figsize=(15, 10))
    plt.title(f'Top 20 Features ({model_type.upper()})')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

    print("Top 20 Feature Importances:")
    for i in indices:
        print(f"{X_train.columns[i]}: {importances[i]}")

def main():
    X_train, X_test, y_train, y_test = load_data()

    # Random Forest
    print("Random Forest Results:")
    rf_model = train_model(X_train, y_train, model_type='rf')
    evaluate_model(rf_model, X_test, y_test)
    plot_feature_importance(rf_model, X_train, model_type='rf')

    # XGBoost
    print("XGBoost Results:")
    xgb_model = train_model(X_train, y_train, model_type='xgb')
    evaluate_model(xgb_model, X_test, y_test)
    plot_feature_importance(xgb_model, X_train, model_type='xgb')

if __name__ == "__main__":
    main()

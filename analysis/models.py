import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV, ParameterGrid
import seaborn as sns


def load_data():
    X_train = pd.read_csv('data/preprocessed_X_train.csv')
    X_test = pd.read_csv('data/preprocessed_X_test.csv')
    y_train = pd.read_csv('data/preprocessed_y_train.csv')
    y_test = pd.read_csv('data/preprocessed_y_test.csv')
    print(y_train['loan_approved'].value_counts())

    # clean column names to be compatible with XGBoost
    columns_to_drop = ['interest_rate', 'rate_spread', 'origination_charges', 'total_loan_costs']
    X_train = X_train.drop(columns=columns_to_drop, axis=1)
    X_test = X_test.drop(columns=columns_to_drop, axis=1)

    X_train.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_train.columns]
    X_test.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_test.columns]

    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()

def findOptimalParams(model, x_train, y_train, modelType):
    modelGrid = {
        "rf": 
            {
                'n_estimators': [10, 50, 100],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': [None, 'sqrt', 'log2']
            },
        "xgb":
            {
                'booster': ['gbtree', 'dart'],
                'learning_rate': [0.1, 0.3, 0.5],
                'min_split_loss': [0, 10, 100, 1000],
                'max_depth': [4, 6, 8, 10],
                'scale_pos_weight': [0.25, 0.5, 0.8, 1]
            }
    }
    
    if modelType not in modelGrid.keys():
        raise Exception("not a valid model")
    
    modelGrid = modelGrid[modelType]
    
    optimalParam = RandomizedSearchCV(model, modelGrid, cv=5, n_jobs=-1, n_iter=int(0.33 * len(list(ParameterGrid(modelGrid)))))
    optimalParam.fit(x_train, y_train)
    optimalParam = optimalParam.best_params_
    print(optimalParam)
    
    model.set_params(**optimalParam)
    
    return model
    
# Apply SMOTE to balance the dataset
def balance_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    return X_train_balanced, y_train_balanced

# Train model
def train_model(X_train, y_train, model_type='rf'):
    if model_type == 'rf':
        rf_optimal = {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': None}
        model = RandomForestClassifier(random_state=42)
        model.set_params(**rf_optimal)
    elif model_type == 'xgb':
        xgb_optimal = {'scale_pos_weight': 1, 'min_split_loss': 0, 'max_depth': 8, 'learning_rate': 0.5, 'booster': 'dart'}
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.set_params(**xgb_optimal)
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
    
    yProb = [x[1] for x in model.predict_proba(X_test)]
    fpr, tpr, thresholds = roc_curve(y_test, yProb)
    data = {'tpr': tpr, 'fpr': fpr}
    return y_pred, data

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
    #
    # print("Top 20 Feature Importances:")
    # for i in indices:
    #     print(f"{X_train.columns[i]}: {importances[i]}")
    
def plot_ROC(data):
    totaldf = None
    for model in data:
        tpr = data[model]['tpr']
        fpr = data[model]['fpr']
    
        df = pd.DataFrame({'tpr': tpr, 'fpr': fpr, 'model': model})
        
        if totaldf is None:
            totaldf = df
        else:
            totaldf = pd.concat([totaldf, df])
        

    lines = sns.lineplot(data = totaldf, x='fpr', y='tpr', hue='model')
    lines.set_title('ROC Curve')
    lines.set_xlabel('FPR')
    lines.set_ylabel('TPR')
    lines.set_xlim([0, 1])
    lines.set_ylim([0, 1])
    plt.show()


def main():
    X_train, X_test, y_train, y_test = load_data()
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train)  # Balance the dataset with SMOTE

    bigData = {}
    # Random Forest
    print("Random Forest Results:")
    rf_model = train_model(X_train_balanced, y_train_balanced, model_type='rf')
    y_pred, data = evaluate_model(rf_model, X_test, y_test)
    # plot_feature_importance(rf_model, X_train, model_type='rf')
    bigData["Random Forest"] = data
    # XGBoost
    print("XGBoost Results:")
    xgb_model = train_model(X_train_balanced, y_train_balanced, model_type='xgb')
    y_pred, data = evaluate_model(xgb_model, X_test, y_test)
    # plot_feature_importance(xgb_model, X_train_balanced, model_type='xgb')
    bigData["XGBoost"] = data
    
    plot_ROC(bigData)
if __name__ == "__main__":
    main()
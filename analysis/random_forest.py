from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, ParameterGrid
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, f1_score

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # Load the data
    data = pd.read_csv('data/sampled_preprocessed_data.csv')
    y = data['loan_approved'].to_numpy()
    x = data.drop(columns=['loan_approved']).to_numpy()

    # Split the data into training and testing sets
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

    # Define the hyperparameters to search
    paramGrid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [None, 'sqrt', 'log2']
    }
    
    optimal = {'n_estimators': 10, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': None, 'max_depth': 5}
    
    model = RandomForestClassifier()
    
    optimalParam = RandomizedSearchCV(model, paramGrid, cv=5, n_jobs=-1, n_iter=int(0.33 * len(list(ParameterGrid(paramGrid)))))
    optimalParam.fit(xTrain, yTrain)
    
    model.set_params(optimal)
    model.fit(xTrain, yTrain)
    
    yHat = model.predict(xTest)
    yProb = [x[1] for x in model.predict_proba(xTest)]
    
    fpr, tpr, thresholds = roc_curve(yTest, yProb)
    rocAuc = roc_auc_score(yTest, yHat)
    precision, recall, _ = precision_recall_curve(yTest, yProb)
    aucPrc = auc(recall, precision)
    f1 = f1_score(yTest, yHat)
    
    return (
        {'AUC': rocAuc, 'AUPRC': aucPrc, 'F1': f1},
        {'tpr': tpr, 'fpr': fpr},
        optimalParam.best_params_
    )

def doTree(x, y):
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)
    
    model = RandomForestClassifier(n_estimators=10, min_samples_split=10, min_samples_leaf=1, max_features=None, max_depth=5)
    model.fit(xTrain, yTrain)
    
    yHat = model.predict(xTest)
    yProb = [x[1] for x in model.predict_proba(xTest)]
    
    fpr, tpr, thresholds = roc_curve(yTest, yProb)
    rocAuc = roc_auc_score(yTest, yHat)
    precision, recall, _ = precision_recall_curve(yTest, yProb)
    aucPrc = auc(recall, precision)
    f1 = f1_score(yTest, yHat)
    
    return (
        {'AUC': rocAuc, 'AUPRC': aucPrc, 'F1': f1},
        {'Precision': precision, 'Recall': recall}
        # {'tpr': tpr, 'fpr': fpr}
    )

def main2():
    data = pd.read_csv('data/sampled_preprocessed_data.csv')
    y = data['loan_approved'].to_numpy()
    x = data.drop(columns=['loan_approved']).to_numpy()
    
    one = doTree(x, y)
    
    ytmp = []
    for val in y:
        if val == 1:
            ytmp.append(0)
        else:
            ytmp.append(1)
    
    y = np.array(ytmp)
    
    two = doTree(x, y)
    
    print(one, two)

    
def plotRoc():
    tpr=[0, 0.99175382, 0.99185438, 0.99597747, 0.99607804,
       0.9961786 , 0.99658085, 0.99678198, 0.99678198, 0.99718423,
       0.99728479, 0.99738536, 0.99738536, 0.99748592, 0.99748592,
       0.99748592, 0.99758648, 0.99758648, 0.99778761, 0.99778761,
       0.99778761, 0.99788817, 0.99798874, 0.9980893 , 0.9980893 ,
       0.99829043, 0.99829043, 0.99849155, 0.99859212, 0.99859212,
       0.99869268, 0.99899437, 0.99909493, 0.99909493, 0.99919549,
       0.99939662, 0.99939662, 0.99939662, 0.99959775, 0.99959775,
       0.99959775, 1, 1]

    fpr = [0, 0, 0, 0.05357143, 0.05357143,
       0.07142857, 0.07142857, 0.10714286, 0.125, 0.125,
       0.14285714, 0.14285714, 0.16071429, 0.16071429, 0.19642857,
       0.23214286, 0.23214286, 0.26785714, 0.30357143, 0.375,
       0.42857143, 0.46428571, 0.46428571, 0.51785714, 0.53571429,
       0.57142857, 0.58928571, 0.64285714, 0.66071429, 0.67857143,
       0.67857143, 0.69642857, 0.69642857, 0.71428571, 0.73214286,
       0.73214286, 0.76785714, 0.78571429, 0.78571429, 0.83928571,
       0.875, 0.91071429, 1]

    data = pd.DataFrame({'tpr': tpr, 'fpr': fpr})

    lines = sns.lineplot(x=fpr, y=tpr)
    lines.set_title('ROC Curve')
    lines.set_xlabel('FPR')
    lines.set_ylabel('TPR')
    lines.set_xlim([0, 1])
    lines.set_ylim([0, 1])
    plt.show()


if __name__ == "__main__":
    # main2()
    plotRoc()
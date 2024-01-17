import pandas as pd
import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def preprocess_data(df):
    # Fill missing values with KNNImputer
    knn_imputer = KNNImputer(n_neighbors=5)
    df[['Age']] = knn_imputer.fit_transform(df[['Age']])
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Encode categorical variables
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    return df

# Read the dataset
url = 'https://raw.githubusercontent.com/kelly-olsson/titanic/main/train.csv'
df = pd.read_csv(url)

# Preprocess the data
df = preprocess_data(df)

# Separate into x and y values
predictorVariables = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[predictorVariables]
y = df['Survived']

# ####### Uncomment this code to find best features ########

# # Scale the data when searching for best features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X = pd.DataFrame(X_scaled, columns=predictorVariables)

# # Use mutual_info_classif for feature selection
# test = SelectKBest(score_func=mutual_info_classif, k=4)
# mi_scores = test.fit(X, y)
# np.set_printoptions(precision=3)

# # Best predictor variables were consistently ['Pclass', 'Sex', 'Age', 'Fare']
# print("\nPredictor variables: " + str(predictorVariables))
# print("Predictor Mutual Information Scores: " + str(mi_scores.scores_))

# # Select significant variables using the get_support() function
# cols = mi_scores.get_support(indices=True)
# print(cols)
# features = X.columns[cols]
# print(features.values)

# ####### End Feature Selection Section ########

# Use consistently best features
features = ['Pclass', 'Sex', 'Age', 'Fare']

# Re-assign X with significant columns only after chi-square test
X = df[features]

scaler = StandardScaler()
X = scaler.fit_transform(X)
pickle.dump(scaler, open('sc_x.pkl', 'wb'))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Build logistic regression model and perform cross-validation
logisticModel = LogisticRegression(fit_intercept=True, solver='liblinear')
grid = GridSearchCV(logisticModel, param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100]}, cv=5)
grid.fit(X_train, y_train)

print("Best parameters found by grid search: ", grid.best_params_)

best_model = grid.best_estimator_

scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
for metric in scoring_metrics:
    scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring=metric)
    print(f"{metric.capitalize()} scores across folds: {scores}")
    print(f"Mean {metric.capitalize()}: {scores.mean()}")
    print(f"Standard deviation {metric.capitalize()}: {scores.std()}\n")

with open('model_pkl', 'wb') as files:
    pickle.dump(grid.best_estimator_, files)

# Load saved model
with open('model_pkl', 'rb') as f:
    loadedModel = pickle.load(f)

y_pred = loadedModel.predict(X_test)
print("***Predictions")
print(y_pred)

# Show confusion matrix and accuracy scores
cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('\nAccuracy:', accuracy)
print("\nConfusion Matrix")
print(cm)
print("Recall: " + str(recall))
print("Precision: " + str(precision))
print("F1-Score: " + str(f1))

# Create a single prediction.
singleSampleDf = pd.DataFrame(columns=features)

pClass =  3
sex = 0
age = 22
fare = 7.25

passengerData = {'Pclass': pClass, 'Sex': sex, 'Age': age, 'Fare': fare}
singleSampleDf = pd.concat([singleSampleDf,
                            pd.DataFrame.from_records([passengerData])])

# Scale the singleSampleDf using the same scaler object
loaded_scalerX = pickle.load(open('sc_x.pkl', 'rb'))
singleSampleDf_scaled = loaded_scalerX.transform(singleSampleDf)

singlePrediction = loadedModel.predict(singleSampleDf_scaled)
print("Single prediction: " + str(singlePrediction))
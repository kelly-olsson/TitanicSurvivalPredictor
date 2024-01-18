# Titanic Survival Predictor

## Introduction
This project applies machine learning techniques to predict survival on the Titanic. It leverages logistic regression, a robust method for binary classification, to analyze historical passenger data and predict survival outcomes.

## Dataset
The Titanic dataset used in this project is sourced from Kaggle and includes various passenger details from the Titanic disaster. The dataset consists of the following columns:

- `survival`: Indicates survival status (0 = No, 1 = Yes)
- `pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `sex`: Sex of the passenger
- `Age`: Age in years
- `sibsp`: Number of siblings/spouses aboard
- `parch`: Number of parents/children aboard
- `ticket`: Ticket number
- `fare`: Passenger fare
- `cabin`: Cabin number
- `embarked`: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

This dataset provides a historical context for machine learning, particularly in understanding the factors that may have influenced passenger survival.

Dataset Source: [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data)

## Methodology
- **Data Preprocessing**: Includes filling missing values using KNNImputer and encoding categorical variables.
- **Feature Selection**: Employed SelectKBest and mutual_info_classif for identifying significant features.
- **Modeling**: Logistic regression, optimized with GridSearchCV for hyperparameter tuning and cross-validation, is employed for prediction.

## Visualizations
The project includes several visualizations:
- Correlation Heatmaps
- Survival Rates by Class (Bar Plot)
- Age-wise Distribution of Survivors (Histogram)
- Gender-based Survival Comparison (Bar Plot)
- Fare vs. Survival (Box Plot)

[Exploratory Data Analysis Presentation + Demo Here](https://youtu.be/z1ShwfPWH-A?si=BqfHZ5f8fNpUCVZ7)

## Results
Key findings from the model include the significant impact of passenger class, gender, and fare on survival probabilities. The model demonstrates a balanced accuracy with precision and recall metrics.

## Usage
The final model is hosted, allowing users to interactively input passenger details and receive a prediction on their survival outcome. 

Experience the model live at [Titanic Survival Predictor](http://kolssonbcit.pythonanywhere.com/)

## Repository Contents
- Code for data preprocessing and modeling
- Serialized model for prediction

## Conclusion
This project not only provides a practical application of logistic regression to a historical dataset but also showcases the importance of feature selection and model tuning in predictive analytics.

---

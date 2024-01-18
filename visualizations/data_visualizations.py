import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# Read the dataset
df = pd.read_csv('train.csv')

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Basic dataset statistics
# print(df.describe())

# Age has null values
print(df.describe().T)

# Visualization 1: Survival rate by gender
sns.set(style='darkgrid')
sns.countplot(x='Survived', data=df, hue='Sex')
plt.title('Survival rate by gender')
plt.show()

# Visualization 2: Survival rate by passenger class
sns.countplot(x='Survived', data=df, hue='Pclass')
plt.title('Survival rate by passenger class')
# plt.show()

# Visualization 3: Age distribution of passengers
sns.histplot(df['Age'].dropna(), kde=True)
plt.title('Age distribution of passengers')
# plt.show()

# Visualization 4: Fare distribution
sns.histplot(df['Fare'], kde=True)
plt.title('Fare distribution')
# plt.show()

# Visualization 5: Survival rate by embarkation point
sns.countplot(x='Survived', data=df, hue='Embarked')
plt.title('Survival rate by embarkation point')
# plt.show()

# Visualization 6: Age distribution by passenger class
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Age distribution by passenger class')
# plt.show()

# Visualization 7: Correlation heatmap of features
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation heatmap of features')
# plt.show()

# plt.show(block=True)
# plt.interactive(False)
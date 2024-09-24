import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data_path = '../data/COPD_Data_Nepal.csv'
df = pd.read_csv(data_path)

# Display basic statistics
print(df.describe())

# EDA: Distribution of COPD Diagnosis
sns.countplot(x='COPD_Diagnosis', data=df)
plt.title("COPD Diagnosis Distribution")
plt.show()

# EDA: Age distribution
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# EDA: Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

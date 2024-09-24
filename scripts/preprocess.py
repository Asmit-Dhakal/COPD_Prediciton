import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load data
data_path = '../data/COPD_Data_Nepal.csv'
df = pd.read_csv(data_path)

# One-hot encoding for categorical variables using separate encoders
gender_encoder = OneHotEncoder(drop='first', sparse_output=False)
smoking_encoder = OneHotEncoder(drop='first', sparse_output=False)
location_encoder = OneHotEncoder(drop='first', sparse_output=False)

# Encode 'Gender', 'Smoking_Status', and 'Location' separately
encoded_gender = gender_encoder.fit_transform(df[['Gender']])
encoded_smoking = smoking_encoder.fit_transform(df[['Smoking_Status']])
encoded_location = location_encoder.fit_transform(df[['Location']])

# Verify the shape of the encoded arrays
print(f"Encoded Gender Shape: {encoded_gender.shape}")
print(f"Encoded Smoking Shape: {encoded_smoking.shape}")
print(f"Encoded Location Shape: {encoded_location.shape}")

# Assign appropriate column names based on the categories after dropping the first category
gender_columns = [f"Gender_{cat}" for cat in gender_encoder.categories_[0][1:]]  # Skip the first category
smoking_columns = [f"Smoking_Status_{cat}" for cat in smoking_encoder.categories_[0][1:]]  # Skip the first category
location_columns = [f"Location_{cat}" for cat in location_encoder.categories_[0][1:]]  # Skip the first category

# Drop original categorical columns and concatenate the one-hot encoded columns
df_encoded = pd.concat([df.drop(['Gender', 'Smoking_Status', 'Location'], axis=1),
                        pd.DataFrame(encoded_gender, columns=gender_columns),
                        pd.DataFrame(encoded_smoking, columns=smoking_columns),
                        pd.DataFrame(encoded_location, columns=location_columns)], axis=1)

# Standardize numerical features
scaler = StandardScaler()
scaled_columns = ['Age', 'BMI', 'Air_Pollution_Level']
df_encoded[scaled_columns] = scaler.fit_transform(df_encoded[scaled_columns])

# Save preprocessed data (optional)
df_encoded.to_csv('../data/copd_preprocessed.csv', index=False)

# Train-test split
X = df_encoded.drop('COPD_Diagnosis', axis=1)
y = df_encoded['COPD_Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print final shape of training and test sets
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")


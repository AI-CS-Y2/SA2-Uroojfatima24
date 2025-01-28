import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Load the dataset
file_path = 'Depression Student Dataset.csv'  
data = pd.read_csv(file_path)

# Preprocessing
# Encode categorical variables
categorical_cols = ['Gender', 'Sleep Duration', 'Dietary Habits', 
                    'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

# One-hot encode categorical columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_categorical = pd.DataFrame(encoder.fit_transform(data[categorical_cols]),
                                   columns=encoder.get_feature_names_out(categorical_cols))

# Label encode the target variable
label_encoder = LabelEncoder()
data['Depression'] = label_encoder.fit_transform(data['Depression'])

# Combine encoded categorical and numerical data
numerical_cols = ['Age', 'Academic Pressure', 'Study Satisfaction', 'Study Hours', 'Financial Stress']
processed_data = pd.concat([encoded_categorical, data[numerical_cols], data['Depression']], axis=1)

# Split data into features and target
X = processed_data.drop('Depression', axis=1)
y = processed_data['Depression']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Random Forest metrics
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_classification_report = classification_report(y_test, rf_predictions, target_names=label_encoder.classes_)

# Train and evaluate Support Vector Machine
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_probabilities = svm_model.predict_proba(X_test)[:, 1]  # For ROC-AUC

# SVM metrics
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_classification_report = classification_report(y_test, svm_predictions, target_names=label_encoder.classes_)
roc_auc = roc_auc_score(y_test, svm_probabilities)

# Output results
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Classification Report:\n", rf_classification_report)
print("SVM Accuracy:", svm_accuracy)
print("SVM Classification Report:\n", svm_classification_report)
print("SVM ROC-AUC Score:", roc_auc)

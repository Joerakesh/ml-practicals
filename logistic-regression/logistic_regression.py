# ============================================================
# LOGISTIC REGRESSION - HEART DISEASE PREDICTION
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("heart_disease_dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

# X = input features
# y = target/output

X = df.drop("Heart Disease", axis=1)

y = df["Heart Disease"]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(y.name)


# ============================================================
# 4. IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns

print("\nCategorical Columns:")
print(list(categorical_columns))

print("\nNumerical Columns:")
print(list(numerical_columns))


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

# Replace missing categorical values
# with the most frequently occurring value

imputer = SimpleImputer(
    strategy="most_frequent"
)

X[categorical_columns] = imputer.fit_transform(
    X[categorical_columns]
)

print("\nMissing Values After Imputation:")
print(X.isnull().sum())


# ============================================================
# 6. LABEL ENCODING
# ============================================================

# Convert categorical values into numerical values

label_encoders = {}

for column in categorical_columns:

    encoder = LabelEncoder()

    X[column] = encoder.fit_transform(
        X[column]
    )

    label_encoders[column] = encoder


print("\nData After Label Encoding:")
print(X.head())


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ============================================================
# 8. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

# Fit scaler using training data
# and transform training data

X_train_scaled = scaler.fit_transform(
    X_train
)

# Only transform test data

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 9. CREATE LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ============================================================
# 10. TRAIN THE MODEL
# ============================================================

model.fit(
    X_train_scaled,
    y_train
)

print("\nLogistic Regression Model Trained Successfully!")


# ============================================================
# 11. MAKE PREDICTIONS
# ============================================================

# Predicted class
y_pred = model.predict(
    X_test_scaled
)

# Probability of class 1
y_probability = model.predict_proba(
    X_test_scaled
)[:, 1]


print("\nFirst 20 Predictions:")
print(y_pred[:20])


# ============================================================
# 12. CALCULATE CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ============================================================
# 13. DISPLAY CLASSIFICATION METRICS
# ============================================================

print("\n========================================")
print("       CLASSIFICATION METRICS")
print("========================================")

print(
    "Accuracy  :",
    round(accuracy, 4)
)

print(
    "Precision :",
    round(precision, 4)
)

print(
    "Recall    :",
    round(recall, 4)
)

print(
    "F1 Score  :",
    round(f1, 4)
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 16. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)


# ============================================================
# 17. CALCULATE AUC
# ============================================================

roc_auc = auc(
    fpr,
    tpr
)

print(
    "\nROC-AUC Score:",
    round(roc_auc, 4)
)


# ============================================================
# 18. PLOT ROC CURVE
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label="Logistic Regression (AUC = {:.3f})".format(
        roc_auc
    )
)

# Random classifier line

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Heart Disease Prediction"
)

plt.legend()

plt.grid()

plt.show()

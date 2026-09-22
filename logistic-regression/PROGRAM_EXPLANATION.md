# 0. Step-by-Step Explanation

## Step 1 — Import Libraries

The program imports pandas:

```python
import pandas as pd
```

Pandas is used to load and manipulate the CSV dataset.

Matplotlib is imported for visualization:

```python
import matplotlib.pyplot as plt
```

Scikit-learn provides the machine learning functionality.

---

## Step 2 — Load the Dataset

```python
df = pd.read_csv("heart_disease_dataset.csv")
```

This loads the CSV file into a pandas DataFrame called `df`.

The program then displays:

```python
print(df.head())
print(df.shape)
```

`head()` displays the first five records.

`shape` displays:

```text
(rows, columns)
```

For this dataset:

```text
(1000, 16)
```

---

# 1. Separating Features and Target

The target column is:

```text
Heart Disease
```

Therefore:

```python
X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]
```

### X

`X` contains the input features.

```text
15 features
```

### y

`y` contains the value the model is trying to predict.

```text
Heart Disease
```

The basic idea is:

```text
X → Model → y
```

---

# 2. Identifying Categorical and Numerical Features

The program identifies categorical columns:

```python
categorical_columns = X.select_dtypes(
    include=["object"]
).columns
```

and numerical columns:

```python
numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns
```

Categorical columns in this dataset include:

```text
Gender
Smoking
Alcohol Intake
Family History
Diabetes
Obesity
Exercise Induced Angina
Chest Pain Type
```

Numerical columns include:

```text
Age
Cholesterol
Blood Pressure
Heart Rate
Exercise Hours
Stress Level
Blood Sugar
```

---

# 3. Handling Missing Values

The program uses:

```python
SimpleImputer(strategy="most_frequent")
```

This replaces missing categorical values with the most frequently occurring value in that column.

Example:

```text
Male
Female
Male
?
Male
```

The missing value can be replaced with:

```text
Male
```

After imputation, the program checks:

```python
print(X.isnull().sum())
```

The expected result is zero missing values for the processed features.

---

# 4. Encoding Categorical Variables

Machine learning algorithms generally require numerical input.

The program uses `LabelEncoder`:

```python
encoder = LabelEncoder()

X[column] = encoder.fit_transform(X[column])
```

For example, a categorical column could become:

```text
Female → 0
Male   → 1
```

The exact numeric assignment is determined by the encoder for each column.

After encoding, all 15 input features are numerical.

---

# 5. Train/Test Split

The dataset is divided into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

This produces:

```text
Training:
800 records

Testing:
200 records
```

because the dataset contains 1000 records.

### test_size

```python
test_size=0.20
```

means 20% of the data is reserved for testing.

Therefore:

```text
80% → Training
20% → Testing
```

### random_state

```python
random_state=42
```

makes the split reproducible.

### stratify

```python
stratify=y
```

helps preserve the target class distribution between training and testing sets.

---

# 6. Feature Scaling

The program uses:

```python
scaler = StandardScaler()
```

Then:

```python
X_train_scaled = scaler.fit_transform(X_train)
```

and:

```python
X_test_scaled = scaler.transform(X_test)
```

Standardization puts numerical features onto a comparable scale.

The important difference is:

```text
Training data:
fit_transform()

Testing data:
transform()
```

The scaler learns its parameters from the training data only.

---

# 7. Creating the Logistic Regression Model

The model is created using:

```python
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

### max_iter

```text
max_iter=1000
```

sets the maximum number of optimization iterations.

A larger value can help the optimization process converge when the default number of iterations is insufficient.

---

# 8. Training the Model

The model is trained using:

```python
model.fit(
    X_train_scaled,
    y_train
)
```

This is the learning stage.

Conceptually:

```text
Training Features
       +
Training Target
       ↓
Logistic Regression
       ↓
Learned Model
```

---

# 9. Making Predictions

The program uses:

```python
y_pred = model.predict(X_test_scaled)
```

This produces the predicted class:

```text
0 or 1
```

Example:

```text
[1 0 0 0 1 0 1 ...]
```

---

# 10. Predicting Probabilities

The program also uses:

```python
y_probability = model.predict_proba(
    X_test_scaled
)[:, 1]
```

Unlike `predict()`, `predict_proba()` returns probabilities.

For example:

```text
Class 0    Class 1
0.20       0.80
0.70       0.30
0.10       0.90
```

The expression:

```python
[:, 1]
```

selects the probability for class `1`.

These probabilities are used for ROC/AUC calculation.

---

# 11. Model Evaluation

The program calculates four major classification metrics.

## Accuracy

```python
accuracy_score(y_test, y_pred)
```

Accuracy measures the proportion of predictions that are correct.

Your result:

```text
Accuracy = 0.865
```

or:

```text
86.5%
```

---

## Precision

```python
precision_score(y_test, y_pred)
```

Your result:

```text
Precision = 0.8493
```

Precision answers:

> Of the cases predicted as positive, how many were actually positive?

---

## Recall

```python
recall_score(y_test, y_pred)
```

Your result:

```text
Recall = 0.7949
```

Recall answers:

> Of all actual positive cases, how many did the model correctly identify?

---

## F1 Score

```python
f1_score(y_test, y_pred)
```

Your result:

```text
F1 Score = 0.8212
```

F1 combines precision and recall into a single metric.

---

# 12. Confusion Matrix

Your result was:

```text
[[111  11]
 [ 16  62]]
```

For binary classification, the matrix can be interpreted as:

```text
                 Predicted
                 0       1

Actual 0        111     11
Actual 1         16     62
```

Therefore:

```text
TN = 111
FP = 11
FN = 16
TP = 62
```

### True Negative — TN

Actual class = 0

Predicted class = 0

```text
111
```

### False Positive — FP

Actual class = 0

Predicted class = 1

```text
11
```

### False Negative — FN

Actual class = 1

Predicted class = 0

```text
16
```

### True Positive — TP

Actual class = 1

Predicted class = 1

```text
62
```

---

# 13. Classification Report

The classification report gives:

```text
precision
recall
f1-score
support
```

Your result:

```text
              precision    recall  f1-score   support

           0       0.87      0.91      0.89       122
           1       0.85      0.79      0.82        78

    accuracy                           0.86       200
```

### Support

Support tells us how many actual samples belong to each class in the test set.

```text
Class 0 → 122
Class 1 → 78
```

Total:

```text
122 + 78 = 200
```

---

# 14. ROC Curve

The program calculates:

```python
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)
```

ROC stands for:

```text
Receiver Operating Characteristic
```

The ROC curve compares:

```text
False Positive Rate
```

against:

```text
True Positive Rate
```

at different classification thresholds.

---

# 15. AUC

The program calculates:

```python
roc_auc = auc(fpr, tpr)
```

Your result:

```text
ROC-AUC = 0.9505
```

AUC means:

```text
Area Under the ROC Curve
```

The ROC-AUC summarizes how well the model separates the two classes across thresholds.

Your reported AUC is approximately:

```text
95.05%
```

---

# 16. Your Current Results

The program produced:

| Metric       | Result |
| ------------ | -----: |
| Test samples |    200 |
| Accuracy     | 0.8650 |
| Precision    | 0.8493 |
| Recall       | 0.7949 |
| F1 Score     | 0.8212 |
| ROC-AUC      | 0.9505 |

Confusion Matrix:

```text
[[111  11]
 [ 16  62]]
```

These are the results from the run using the current train/test split and preprocessing configuration.

---

# 17. Understanding Your Output

Your model correctly classified:

```text
111 + 62 = 173
```

out of 200 test records.

Therefore:

```text
173 / 200 = 0.865
```

which matches:

```text
Accuracy = 0.865
```

The model made:

```text
11 + 16 = 27
```

incorrect predictions.

---

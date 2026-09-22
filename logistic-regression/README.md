# Logistic Regression — Heart Disease Prediction

A Machine Learning practical implementing **Logistic Regression** for binary classification using the `heart_disease_dataset.csv` dataset.

The project demonstrates the complete workflow:

> Dataset → Data Exploration → Missing-Value Handling → Encoding → Train/Test Split → Feature Scaling → Logistic Regression → Prediction → Evaluation → ROC-AUC

---

## 1. Project Overview

This project predicts whether a patient has **heart disease** based on patient-related features such as:

- Age
- Gender
- Cholesterol
- Blood Pressure
- Heart Rate
- Smoking
- Alcohol Intake
- Exercise Hours
- Family History
- Diabetes
- Obesity
- Stress Level
- Blood Sugar
- Exercise Induced Angina
- Chest Pain Type

The target column is:

```text
Heart Disease
```

It is a binary classification problem:

```text
0 → No Heart Disease
1 → Heart Disease
```

---

## 2. Dataset

### Dataset file

```text
heart_disease_dataset.csv
```

### Dataset size

The dataset contains:

```text
1000 rows
16 columns
```

There are:

```text
15 input features
1 target variable
```

### Features

```text
Age
Gender
Cholesterol
Blood Pressure
Heart Rate
Smoking
Alcohol Intake
Exercise Hours
Family History
Diabetes
Obesity
Stress Level
Blood Sugar
Exercise Induced Angina
Chest Pain Type
```

### Target

```text
Heart Disease
```

---

# 3. What is Logistic Regression?

Logistic Regression is a **supervised machine learning algorithm** mainly used for **classification**.

In this project, it predicts one of two classes:

```text
0 → No Heart Disease
1 → Heart Disease
```

Although its name contains "Regression", Logistic Regression is commonly used for binary classification.

The model produces a probability for the positive class.

For example:

```text
Probability = 0.82
```

Using a threshold of 0.5:

```text
0.82 >= 0.5
       ↓
Class 1
```

Another example:

```text
Probability = 0.23

0.23 < 0.5
       ↓
Class 0
```

The probabilities are also used to calculate the ROC curve and AUC.

---

# 4. Project Structure

The folder is organized as follows:

```text
logistic-regression/
│
├── .venv/
│   └── Python virtual environment
│
├── heart_disease_dataset.csv
│   └── Dataset used for training/testing
│
├── logistic_regression.py
│   └── Main Logistic Regression program
│
└── README.md
    └── Project documentation
```

The `.venv` directory should normally not be committed to Git.

---

# 5. Requirements

This project requires:

- Python 3
- pandas
- matplotlib
- scikit-learn

---

# 6. Creating the Virtual Environment

It is recommended to use a Python virtual environment.

From the project directory:

```bash
python -m venv .venv
```

---

# 7. Installing Dependencies

With the virtual environment activated:

```bash
pip install pandas matplotlib scikit-learn
```

Verify the installation:

```bash
python -c "import pandas, matplotlib, sklearn; print('All libraries installed successfully')"
```

---

# 8. Running the Program

Make sure you are inside the project directory:

```bash
cd ~/ml-practicals/logistic-regression
```

Activate the environment:

```bash
source .venv/bin/activate
```

Run:

```bash
python logistic_regression.py
```

The program will print:

- Dataset preview
- Dataset shape
- Feature names
- Target name
- Categorical columns
- Numerical columns
- Missing-value information
- Encoded data
- Training/test sizes
- Predictions
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report
- ROC-AUC

---

# 9. Machine Learning Workflow

The program follows these steps.

```text
                    Dataset
                       │
                       ▼
              Load CSV with pandas
                       │
                       ▼
             Separate X and y
                       │
                       ▼
        Identify categorical/numerical
                  columns
                       │
                       ▼
              Handle missing data
                       │
                       ▼
            Label Encode categories
                       │
                       ▼
             Train/Test Split
                 80% / 20%
                       │
                       ▼
              Standard Scaling
                       │
                       ▼
          Logistic Regression Model
                       │
                       ▼
                   Training
                       │
                       ▼
                  Prediction
                       │
                       ▼
             Model Evaluation
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
       Accuracy    Confusion    ROC-AUC
       Precision   Matrix
       Recall
       F1 Score
```

---

# 10. Important Practical/Viva Questions

You should be able to answer these before moving to the next algorithm.

### Basic

**Q1. What is Logistic Regression?**

A supervised learning algorithm primarily used for classification.

**Q2. What is the target in this project?**

```text
Heart Disease
```

**Q3. What are X and y?**

```text
X → input features
y → target/output
```

**Q4. How many records are in the dataset?**

```text
1000
```

**Q5. How many features are used?**

```text
15
```

---

### Preprocessing

**Q6. Why do we use SimpleImputer?**

To handle missing values.

**Q7. Why do we use LabelEncoder?**

To convert categorical values into numerical values.

**Q8. Why do we use StandardScaler?**

To standardize feature values onto a comparable scale.

**Q9. Why do we split the dataset?**

To train the model on one portion and evaluate it on unseen data.

---

### Model

**Q10. What does `fit()` do?**

It trains the model using the training data.

**Q11. What does `predict()` do?**

It predicts the class labels.

**Q12. What does `predict_proba()` do?**

It returns class probabilities.

---

### Evaluation

**Q13. What is accuracy?**

The proportion of total predictions that are correct.

**Q14. What is precision?**

Among predicted positive cases, the proportion that are actually positive.

**Q15. What is recall?**

Among actual positive cases, the proportion correctly identified.

**Q16. What is F1-score?**

A combined measure based on precision and recall.

**Q17. What is a confusion matrix?**

A table showing true positives, true negatives, false positives, and false negatives.

**Q18. What is ROC-AUC?**

A measure based on the ROC curve that summarizes class-separation performance across classification thresholds.

---

# 11. Quick Revision

If you need to remember the whole practical quickly:

```text
1. Read CSV
       ↓
2. Separate X and y
       ↓
3. Find categorical/numerical columns
       ↓
4. Handle missing values
       ↓
5. Encode categorical columns
       ↓
6. Split 80/20
       ↓
7. StandardScaler
       ↓
8. Create LogisticRegression
       ↓
9. fit()
       ↓
10. predict()
       ↓
11. Accuracy
12. Precision
13. Recall
14. F1
       ↓
15. Confusion Matrix
       ↓
16. Classification Report
       ↓
17. ROC Curve
       ↓
18. AUC
```

---

# 12. Useful Commands

Activate the environment:

```bash
source .venv/bin/activate
```

Run the program:

```bash
python logistic_regression.py
```

Check installed packages:

```bash
pip list
```

Update packages:

```bash
pip install --upgrade pandas matplotlib scikit-learn
```

Deactivate the environment:

```bash
deactivate
```

---

# 13. Important Note for Learning

This project is primarily a **learning/practical implementation**.

The preprocessing approach here follows the structure of the classroom code you provided. In a production-quality ML workflow, preprocessing such as imputation, encoding, and scaling should generally be fitted using the training data only, ideally through a scikit-learn `Pipeline`, to avoid data leakage.

For your practical preparation, first focus on understanding:

```text
X / y
↓
Preprocessing
↓
Train/Test Split
↓
Scaling
↓
Model
↓
Prediction
↓
Evaluation
```

That same structure will make the next algorithms—**KNN, Decision Tree, Random Forest, and SVM**—much easier to understand.

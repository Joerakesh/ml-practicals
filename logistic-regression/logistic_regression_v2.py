# ============================================================
# ADVANCED LOGISTIC REGRESSION - HEART DISEASE PREDICTION
# ============================================================
#
# Industry-style concepts:
#
# 1. Leakage-safe preprocessing
# 2. Pipeline + ColumnTransformer
# 3. Stratified Cross Validation
# 4. Hyperparameter tuning
# 5. ROC-AUC + PR-AUC
# 6. Probability-based prediction
# 7. Decision threshold optimization
# 8. Cost-sensitive classification
# 9. Probability calibration
# 10. Model coefficient interpretation
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    log_loss
)

from sklearn.calibration import calibration_curve


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_PATH = "heart_disease_dataset.csv"

TARGET = "Heart Disease"

RANDOM_STATE = 42

TEST_SIZE = 0.20

# ------------------------------------------------------------
# Decision costs
# ------------------------------------------------------------
#
# False Negative:
# Actual disease but model predicts no disease.
#
# False Positive:
# No disease but model predicts disease.
#
# These values are examples.
# In a real project they should come from domain/business
# requirements.
#
# Here we intentionally make FN more expensive.

FN_COST = 10
FP_COST = 2


# ------------------------------------------------------------
# Threshold search
# ------------------------------------------------------------

THRESHOLDS = np.arange(
    0.05,
    0.96,
    0.01
)


# ------------------------------------------------------------
# Business/domain constraint
# ------------------------------------------------------------

MIN_RECALL = 0.90


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# Make sure target exists

if TARGET not in df.columns:

    raise ValueError(
        f"Target column '{TARGET}' was not found."
    )


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    TARGET,
    axis=1
)

y = df[TARGET]


print("\nTarget Distribution:")
print(
    y.value_counts()
)


print("\nTarget Proportion:")
print(
    y.value_counts(normalize=True)
)


# ============================================================
# 5. IDENTIFY FEATURE TYPES
# ============================================================

categorical_columns = X.select_dtypes(
    include=[
        "object",
        "category",
        "bool"
    ]
).columns.tolist()


numerical_columns = X.select_dtypes(
    exclude=[
        "object",
        "category",
        "bool"
    ]
).columns.tolist()


print("\nNumerical Columns:")
print(
    numerical_columns
)


print("\nCategorical Columns:")
print(
    categorical_columns
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================
#
# IMPORTANT:
#
# Test data should remain untouched during:
#
# - preprocessing decisions
# - hyperparameter tuning
# - threshold selection
#
# We use the test set only for final evaluation.
#
# stratify=y keeps the class distribution similar.
#
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y
)


print("\nTraining Shape:")
print(
    X_train.shape
)


print("\nTest Shape:")
print(
    X_test.shape
)


# ============================================================
# 7. NUMERICAL PREPROCESSING
# ============================================================
#
# Missing numerical values:
#     median
#
# Scaling:
#     StandardScaler
#
# Scaling is useful for Logistic Regression because its
# coefficients are optimized numerically.
#
# ============================================================

numeric_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "scaler",
        StandardScaler()
    )

])


# ============================================================
# 8. CATEGORICAL PREPROCESSING
# ============================================================
#
# Missing categorical values:
#     most frequent
#
# Encoding:
#     OneHotEncoder
#
# We use OneHotEncoder rather than manually assigning numbers
# to categories because nominal categories don't necessarily
# have an order.
#
# handle_unknown="ignore" means a new category at prediction
# time won't crash the pipeline.
#
# ============================================================

categorical_pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )

])


# ============================================================
# 9. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer([

    (
        "numeric",
        numeric_pipeline,
        numerical_columns
    ),

    (
        "categorical",
        categorical_pipeline,
        categorical_columns
    )

])


# ============================================================
# 10. CREATE COMPLETE ML PIPELINE
# ============================================================
#
# Instead of:
#
# preprocess
#     ↓
# transform
#     ↓
# model
#
# everything is packaged together.
#
# This prevents preprocessing inconsistencies and helps avoid
# data leakage during cross-validation.
#
# ============================================================

model_pipeline = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        )
    )

])


# ============================================================
# 11. HYPERPARAMETER SEARCH
# ============================================================
#
# C controls regularization strength.
#
# Smaller C:
#     stronger regularization
#
# Larger C:
#     weaker regularization
#
# class_weight="balanced":
#     useful when classes are imbalanced.
#
# We don't manually choose these.
# GridSearchCV evaluates combinations using cross-validation.
#
# ============================================================

param_grid = {

    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],

    "model__penalty": [
        "l2"
    ],

    "model__class_weight": [
        None,
        "balanced"
    ]

}


# ============================================================
# 12. STRATIFIED CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=RANDOM_STATE

)


# ============================================================
# 13. GRID SEARCH
# ============================================================
#
# ROC-AUC is used here because we want to evaluate the model's
# ability to rank positive examples above negative examples
# independently of a single threshold.
#
# ============================================================

grid_search = GridSearchCV(

    estimator=model_pipeline,

    param_grid=param_grid,

    scoring="roc_auc",

    cv=cv,

    n_jobs=-1,

    refit=True,

    return_train_score=True

)


print("\n========================================")
print("RUNNING 5-FOLD CROSS VALIDATION")
print("========================================")


grid_search.fit(
    X_train,
    y_train
)


# Best trained pipeline

best_model = grid_search.best_estimator_


print("\nBest Parameters:")

print(
    grid_search.best_params_
)


print("\nBest Cross-Validated ROC-AUC:")

print(
    round(
        grid_search.best_score_,
        4
    )
)


# ============================================================
# 14. GET PROBABILITY PREDICTIONS
# ============================================================
#
# This is important.
#
# model.predict()
#     → directly gives 0 or 1
#
# model.predict_proba()
#     → gives probability
#
# We need probability because we want to control the decision
# threshold ourselves.
#
# ============================================================

y_probability = best_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 15. DEFAULT 0.50 THRESHOLD
# ============================================================

y_pred_default = (

    y_probability >= 0.50

).astype(int)


print("\n========================================")
print("DEFAULT THRESHOLD = 0.50")
print("========================================")


print(
    "Accuracy :",
    round(
        accuracy_score(
            y_test,
            y_pred_default
        ),
        4
    )
)


print(
    "Precision:",
    round(
        precision_score(
            y_test,
            y_pred_default,
            zero_division=0
        ),
        4
    )
)


print(
    "Recall   :",
    round(
        recall_score(
            y_test,
            y_pred_default,
            zero_division=0
        ),
        4
    )
)


print(
    "F1       :",
    round(
        f1_score(
            y_test,
            y_pred_default,
            zero_division=0
        ),
        4
    )
)


# ============================================================
# 16. ROC-AUC
# ============================================================
#
# ROC-AUC evaluates how well the model separates positive and
# negative examples across different thresholds.
#
# ============================================================

roc_auc = roc_auc_score(

    y_test,

    y_probability

)


# ============================================================
# 17. PR-AUC
# ============================================================
#
# Average Precision is commonly used as a summary of the
# Precision-Recall curve.
#
# It becomes particularly useful when positive cases are
# relatively uncommon.
#
# ============================================================

pr_auc = average_precision_score(

    y_test,

    y_probability

)


print("\n========================================")
print("PROBABILITY / RANKING METRICS")
print("========================================")


print(
    "ROC-AUC:",
    round(
        roc_auc,
        4
    )
)


print(
    "PR-AUC :",
    round(
        pr_auc,
        4
    )
)


# ============================================================
# 18. THRESHOLD EXPERIMENT
# ============================================================
#
# THIS IS THE ADVANCED VERSION OF YOUR ORIGINAL IDEA.
#
# For every threshold:
#
# probability
#      ↓
# convert to 0/1
#      ↓
# confusion matrix
#      ↓
# calculate metrics
#      ↓
# store in DataFrame
#
# ============================================================

threshold_results = []


for threshold in THRESHOLDS:

    # Convert probability into class

    y_pred = (

        y_probability >= threshold

    ).astype(int)


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(

        y_test,

        y_pred,

        labels=[0, 1]

    ).ravel()


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Specificity
    # --------------------------------------------------------
    #
    # Specificity = TN / (TN + FP)
    #
    # It tells us how well the model identifies negatives.
    #
    # --------------------------------------------------------

    specificity = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else 0

    )


    # --------------------------------------------------------
    # False Positive Rate
    # --------------------------------------------------------

    false_positive_rate = (

        fp / (fp + tn)

        if (fp + tn) > 0

        else 0

    )


    # --------------------------------------------------------
    # Decision cost
    # --------------------------------------------------------
    #
    # We intentionally make false negatives more expensive.
    #
    # Cost = FN × FN_COST + FP × FP_COST
    #
    # --------------------------------------------------------

    decision_cost = (

        FN_COST * fn

        +

        FP_COST * fp

    )


    # --------------------------------------------------------
    # Store everything
    # --------------------------------------------------------

    threshold_results.append({

        "Threshold":
            round(
                float(threshold),
                2
            ),

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "Specificity":
            specificity,

        "FPR":
            false_positive_rate,

        "TP":
            tp,

        "FP":
            fp,

        "TN":
            tn,

        "FN":
            fn,

        "Decision_Cost":
            decision_cost

    })


# ============================================================
# 19. CREATE THRESHOLD DATAFRAME
# ============================================================

threshold_df = pd.DataFrame(

    threshold_results

)


print("\n========================================")
print("THRESHOLD ANALYSIS")
print("========================================")


print(

    threshold_df[
        [
            "Threshold",
            "Precision",
            "Recall",
            "F1",
            "Specificity",
            "Decision_Cost"
        ]
    ].round(4).to_string(
        index=False
    )

)


# ============================================================
# 20. SELECT THRESHOLD
# ============================================================
#
# DON'T simply say:
#
# "Highest F1 = best"
#
# Instead define a requirement.
#
# Here:
#
# Recall >= 90%
#
# Then among those thresholds:
#
# minimize decision cost.
#
# ============================================================

eligible_thresholds = threshold_df[

    threshold_df["Recall"] >= MIN_RECALL

].copy()


if not eligible_thresholds.empty:

    selected_row = eligible_thresholds.loc[

        eligible_thresholds[
            "Decision_Cost"
        ].idxmin()

    ]


    selected_threshold = float(

        selected_row[
            "Threshold"
        ]

    )


else:

    # If 90% recall is impossible,
    # choose the threshold with maximum recall.

    selected_row = threshold_df.loc[

        threshold_df[
            "Recall"
        ].idxmax()

    ]


    selected_threshold = float(

        selected_row[
            "Threshold"
        ]

    )


print("\n========================================")
print("SELECTED DECISION THRESHOLD")
print("========================================")


print(
    "Selected Threshold:",
    selected_threshold
)


print(
    "Minimum Recall:",
    f"{MIN_RECALL:.0%}"
)


print(
    "False Negative Cost:",
    FN_COST
)


print(
    "False Positive Cost:",
    FP_COST
)


# ============================================================
# 21. FINAL PREDICTIONS
# ============================================================

y_pred_selected = (

    y_probability >= selected_threshold

).astype(int)


# ============================================================
# 22. FINAL METRICS
# ============================================================

final_accuracy = accuracy_score(

    y_test,

    y_pred_selected

)


final_precision = precision_score(

    y_test,

    y_pred_selected,

    zero_division=0

)


final_recall = recall_score(

    y_test,

    y_pred_selected,

    zero_division=0

)


final_f1 = f1_score(

    y_test,

    y_pred_selected,

    zero_division=0

)


final_cm = confusion_matrix(

    y_test,

    y_pred_selected

)


print("\n========================================")
print("FINAL MODEL EVALUATION")
print("========================================")


print(
    "Threshold:",
    selected_threshold
)


print(
    "Accuracy :",
    round(
        final_accuracy,
        4
    )
)


print(
    "Precision:",
    round(
        final_precision,
        4
    )
)


print(
    "Recall   :",
    round(
        final_recall,
        4
    )
)


print(
    "F1       :",
    round(
        final_f1,
        4
    )
)


print(
    "ROC-AUC  :",
    round(
        roc_auc,
        4
    )
)


print(
    "PR-AUC   :",
    round(
        pr_auc,
        4
    )
)


print("\nConfusion Matrix:")

print(
    final_cm
)


print("\nClassification Report:")

print(

    classification_report(

        y_test,

        y_pred_selected,

        zero_division=0

    )

)


# ============================================================
# 23. PROBABILITY CALIBRATION
# ============================================================
#
# Suppose the model predicts:
#
#     Patient A -> 0.80
#
# Does that really mean approximately 80% probability?
#
# Calibration evaluates that.
#
# Brier Score:
# lower = better
#
# Log Loss:
# lower = better
#
# ============================================================

brier = brier_score_loss(

    y_test,

    y_probability

)


probability_log_loss = log_loss(

    y_test,

    y_probability

)


print("\n========================================")
print("PROBABILITY CALIBRATION")
print("========================================")


print(
    "Brier Score:",
    round(
        brier,
        4
    )
)


print(
    "Log Loss   :",
    round(
        probability_log_loss,
        4
    )
)


# ============================================================
# 24. LOGISTIC REGRESSION COEFFICIENTS
# ============================================================
#
# Logistic Regression is useful because it is interpretable.
#
# Positive coefficient:
#     pushes prediction toward class 1
#
# Negative coefficient:
#     pushes prediction toward class 0
#
# ============================================================

coefficient_df = pd.DataFrame()


try:

    fitted_preprocessor = (

        best_model.named_steps[
            "preprocessor"
        ]

    )


    fitted_model = (

        best_model.named_steps[
            "model"
        ]

    )


    feature_names = (

        fitted_preprocessor
        .get_feature_names_out()

    )


    coefficients = (

        fitted_model.coef_[0]

    )


    coefficient_df = pd.DataFrame({

        "Feature":
            feature_names,

        "Coefficient":
            coefficients,

        "Absolute_Coefficient":
            np.abs(coefficients)

    })


    coefficient_df = (

        coefficient_df
        .sort_values(
            "Absolute_Coefficient",
            ascending=False
        )

    )


    print("\n========================================")
    print("TOP MODEL COEFFICIENTS")
    print("========================================")


    print(

        coefficient_df
        .head(15)
        .round(4)
        .to_string(
            index=False
        )

    )


except Exception as error:

    print(
        "\nCould not extract coefficients:",
        error
    )


# ============================================================
# 25. ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(

    y_test,

    y_probability

)


plt.figure(

    figsize=(8, 6)

)


plt.plot(

    fpr,

    tpr,

    label=f"ROC-AUC = {roc_auc:.3f}"

)


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
    "ROC Curve - Logistic Regression"
)


plt.legend()


plt.grid()


plt.show()


# ============================================================
# 26. PRECISION-RECALL CURVE
# ============================================================

precision_values, recall_values, _ = (

    precision_recall_curve(

        y_test,

        y_probability

    )

)


plt.figure(

    figsize=(8, 6)

)


plt.plot(

    recall_values,

    precision_values,

    label=f"PR-AUC = {pr_auc:.3f}"

)


plt.xlabel(
    "Recall"
)


plt.ylabel(
    "Precision"
)


plt.title(
    "Precision-Recall Curve"
)


plt.legend()


plt.grid()


plt.show()


# ============================================================
# 27. THRESHOLD vs METRICS
# ============================================================
#
# This visualization makes your original threshold idea much
# easier to understand.
#
# ============================================================

plt.figure(

    figsize=(10, 6)

)


plt.plot(

    threshold_df["Threshold"],

    threshold_df["Precision"],

    label="Precision"

)


plt.plot(

    threshold_df["Threshold"],

    threshold_df["Recall"],

    label="Recall"

)


plt.plot(

    threshold_df["Threshold"],

    threshold_df["F1"],

    label="F1"

)


plt.axvline(

    selected_threshold,

    linestyle="--",

    label=(
        f"Selected = "
        f"{selected_threshold:.2f}"
    )

)


plt.xlabel(
    "Decision Threshold"
)


plt.ylabel(
    "Metric"
)


plt.title(
    "Decision Threshold Analysis"
)


plt.legend()


plt.grid()


plt.show()


# ============================================================
# 28. CALIBRATION CURVE
# ============================================================

prob_true, prob_pred = calibration_curve(

    y_test,

    y_probability,

    n_bins=10,

    strategy="uniform"

)


plt.figure(

    figsize=(8, 6)

)


plt.plot(

    prob_pred,

    prob_true,

    marker="o",

    label="Logistic Regression"

)


plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--",

    label="Perfect Calibration"

)


plt.xlabel(
    "Mean Predicted Probability"
)


plt.ylabel(
    "Fraction of Positives"
)


plt.title(
    "Probability Calibration Curve"
)


plt.legend()


plt.grid()


plt.show()


# ============================================================
# 29. SAVE EXPERIMENT RESULTS
# ============================================================
#
# Instead of losing all threshold experiments after the
# program finishes, save them.
#
# ============================================================

threshold_df.to_csv(

    "threshold_analysis.csv",

    index=False

)


if not coefficient_df.empty:

    coefficient_df.to_csv(

        "logistic_coefficients.csv",

        index=False

    )


print("\n========================================")
print("FILES SAVED")
print("========================================")

print(
    "threshold_analysis.csv"
)


if not coefficient_df.empty:

    print(
        "logistic_coefficients.csv"
    )


# ============================================================
# 30. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("EXPERIMENT COMPLETED")
print("========================================")


print("\nBest Hyperparameters:")

print(
    grid_search.best_params_
)


print("\nSelected Threshold:")

print(
    selected_threshold
)


print("\nFinal Metrics:")

print(
    "Accuracy :",
    round(
        final_accuracy,
        4
    )
)


print(
    "Precision:",
    round(
        final_precision,
        4
    )
)


print(
    "Recall   :",
    round(
        final_recall,
        4
    )
)


print(
    "F1       :",
    round(
        final_f1,
        4
    )
)


print(
    "ROC-AUC  :",
    round(
        roc_auc,
        4
    )
)


print(
    "PR-AUC   :",
    round(
        pr_auc,
        4
    )
)
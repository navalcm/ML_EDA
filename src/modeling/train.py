"""
Model Training Module

Loads processed train/test data,
trains Logistic Regression, Random Forest and XGBoost,
evaluates all models,
saves individual models,
and saves the best model.
"""

# ============================================================
# IMPORTS
# ============================================================

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (
    LOGISTIC_MODEL_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RF_MODEL_PATH,
    TARGET_COL,
    TEST_PROCESSED_PATH,
    TRAIN_PROCESSED_PATH,
    XGB_MODEL_PATH,
)
from src.logger import get_logger

# ============================================================
# LOGGER
# ============================================================

logger = get_logger(__name__)


# ============================================================
# LOAD PROCESSED DATA
# ============================================================


def load_processed_data():
    """
    Load processed training and testing datasets.
    """

    logger.info("=" * 70)
    logger.info("LOADING PROCESSED DATA")
    logger.info("=" * 70)

    if not TRAIN_PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Training file not found: {TRAIN_PROCESSED_PATH}")

    if not TEST_PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Testing file not found: {TEST_PROCESSED_PATH}")

    train_df = pd.read_csv(TRAIN_PROCESSED_PATH)
    test_df = pd.read_csv(TEST_PROCESSED_PATH)

    logger.info(
        "Train shape: %s",
        train_df.shape,
    )

    logger.info(
        "Test shape: %s",
        test_df.shape,
    )

    return train_df, test_df


# ============================================================
# PREPARE FEATURES
# ============================================================


def prepare_features(train_df, test_df):
    """
    Separate features X and target y.
    """

    logger.info("=" * 70)
    logger.info("PREPARING FEATURES")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # Check target
    # --------------------------------------------------------

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in training data.")

    if TARGET_COL not in test_df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in testing data.")

    # --------------------------------------------------------
    # Separate X and y
    # --------------------------------------------------------

    X_train = train_df.drop(columns=[TARGET_COL])

    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])

    y_test = test_df[TARGET_COL]

    # --------------------------------------------------------
    # Make sure features are numeric
    # --------------------------------------------------------

    non_numeric_columns = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    if non_numeric_columns:
        raise ValueError(
            "Non-numeric columns found in processed data: "
            f"{non_numeric_columns}. "
            "Feature engineering must encode categorical "
            "columns before model training."
        )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logger.info(
        "X_train shape: %s",
        X_train.shape,
    )

    logger.info(
        "X_test shape: %s",
        X_test.shape,
    )

    logger.info(
        "y_train shape: %s",
        y_train.shape,
    )

    logger.info(
        "y_test shape: %s",
        y_test.shape,
    )

    logger.info(
        "Training target distribution:\n%s",
        y_train.value_counts(),
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# BUILD MODELS
# ============================================================


def build_models():
    """
    Create all baseline ML models.
    """

    logger.info("=" * 70)
    logger.info("CREATING ML MODELS")
    logger.info("=" * 70)

    models = {
        # ----------------------------------------------------
        # Logistic Regression
        # ----------------------------------------------------
        "Logistic Regression": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        # ----------------------------------------------------
        # XGBoost
        # ----------------------------------------------------
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    logger.info(
        "Models created: %s",
        list(models.keys()),
    )

    return models


# ============================================================
# SAVE INDIVIDUAL MODEL
# ============================================================


def save_model(model, model_name):
    """
    Save individual trained model.
    """

    if model_name == "Logistic Regression":
        path = LOGISTIC_MODEL_PATH

    elif model_name == "Random Forest":
        path = RF_MODEL_PATH

    elif model_name == "XGBoost":
        path = XGB_MODEL_PATH

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        path,
    )

    logger.info(
        "%s saved: %s",
        model_name,
        path,
    )

    return path


# ============================================================
# TRAIN MODELS
# ============================================================


def train_baseline_models():
    """
    Train all baseline models,
    evaluate them,
    save individual models,
    and save the best model.
    """

    logger.info("")
    logger.info("=" * 70)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_df, test_df = load_processed_data()

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = prepare_features(
        train_df,
        test_df,
    )

    # --------------------------------------------------------
    # Create models
    # --------------------------------------------------------

    models = build_models()

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results = {}

    trained_models = {}

    # ========================================================
    # TRAIN EACH MODEL
    # ========================================================

    for model_name, model in models.items():
        logger.info("")
        logger.info("-" * 70)
        logger.info(
            "TRAINING: %s",
            model_name,
        )
        logger.info("-" * 70)

        try:
            # ------------------------------------------------
            # Train
            # ------------------------------------------------

            model.fit(
                X_train,
                y_train,
            )

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            predictions = model.predict(X_test)

            # ------------------------------------------------
            # Accuracy
            # ------------------------------------------------

            accuracy = accuracy_score(
                y_test,
                predictions,
            )

            results[model_name] = accuracy

            trained_models[model_name] = model

            # ------------------------------------------------
            # Log accuracy
            # ------------------------------------------------

            logger.info(
                "%s Accuracy: %.4f",
                model_name,
                accuracy,
            )

            # ------------------------------------------------
            # Classification report
            # ------------------------------------------------

            report = classification_report(
                y_test,
                predictions,
                zero_division=0,
            )

            logger.info(
                "\n%s",
                report,
            )

            # ------------------------------------------------
            # Save individual model
            # ------------------------------------------------

            save_model(
                model,
                model_name,
            )

        except Exception:
            logger.exception(
                "ERROR while training %s",
                model_name,
            )

    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if not results:
        raise RuntimeError("No model was successfully trained.")

    # ========================================================
    # FIND BEST MODEL
    # ========================================================

    best_model_name = max(
        results,
        key=results.get,
    )

    best_accuracy = results[best_model_name]

    best_model = trained_models[best_model_name]

    # ========================================================
    # LOG RESULTS
    # ========================================================

    logger.info("")
    logger.info("=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)

    for model_name, accuracy in results.items():
        logger.info(
            "%-25s : %.4f",
            model_name,
            accuracy,
        )

    logger.info("=" * 70)

    logger.info(
        "BEST MODEL: %s",
        best_model_name,
    )

    logger.info(
        "BEST ACCURACY: %.4f",
        best_accuracy,
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    logger.info(
        "BEST MODEL SAVED: %s",
        MODEL_PATH,
    )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    logger.info("=" * 70)
    logger.info("MODEL TRAINING COMPLETED")
    logger.info("=" * 70)

    return {
        "best_model_name": best_model_name,
        "best_accuracy": best_accuracy,
        "results": results,
        "best_model": best_model,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train_baseline_models()

from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


# ============================================================
# RAW DATA
# ============================================================

RAW_DATA_FILE = RAW_DATA_DIR / "Loan_Default.csv"


# ============================================================
# PROCESSED DATA
# ============================================================

TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train_processed.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test_processed.csv"

# Aliases used by training code
TRAIN_PROCESSED_PATH = TRAIN_DATA_FILE
TEST_PROCESSED_PATH = TEST_DATA_FILE


# ============================================================
# MODEL DIRECTORIES
# ============================================================

# Main model directory
MODEL_DIR = PROJECT_ROOT / "models"

# Alias used by some training modules
MODELS_DIR = MODEL_DIR


# Individual model directories
LOGISTIC_MODEL_DIR = MODEL_DIR / "logistic_regression"
RF_MODEL_DIR = MODEL_DIR / "random_forest"
XGB_MODEL_DIR = MODEL_DIR / "xgboost"


# ============================================================
# MODEL FILES
# ============================================================

LOGISTIC_MODEL_PATH = LOGISTIC_MODEL_DIR / "logistic_regression.joblib"

RF_MODEL_PATH = RF_MODEL_DIR / "random_forest.joblib"

XGB_MODEL_PATH = XGB_MODEL_DIR / "xgboost.joblib"


# ============================================================
# GENERAL / BEST MODEL
# ============================================================

MODEL_PATH = MODEL_DIR / "best_model.joblib"


# ============================================================
# MODEL VERSIONS
# ============================================================

LOGISTIC_MODEL_VERSION = "v1.0.0"

RF_MODEL_VERSION = "v1.0.0"

XGB_MODEL_VERSION = "v1.0.0"


# ============================================================
# TARGET COLUMN
# ============================================================

TARGET_COLUMN = "Status"

# Alias used by training code
TARGET_COL = TARGET_COLUMN


# ============================================================
# TARGET LEAKAGE FEATURES
# ============================================================

LEAKAGE_COLS = [
    "rate_of_interest",
    "Interest_rate_spread",
    "Upfront_charges",
]


# ============================================================
# ZERO / NEAR-ZERO VARIANCE FEATURES
# ============================================================

ZERO_VAR_COLS = [
    "ID",
    "year",
    "construction_type",
    "Secured_by",
    "Security_Type",
    "open_credit",
]


# ============================================================
# UNPREDICTIVE FEATURES
# ============================================================

UNPREDICTIVE_COLS = [
    "Credit_Score",
    "term",
    "Gender",
    "Region",
    "approv_in_adv",
    "loan_limit",
    "business_or_commercial",
    "credit_type",
    "Credit_Worthiness",
    "interest_only",
]


# ============================================================
# MASTER FEATURE DROP LIST
# ============================================================

DROP_FEATURES = list(dict.fromkeys(LEAKAGE_COLS + ZERO_VAR_COLS + UNPREDICTIVE_COLS))


# ============================================================
# NUMERICAL FEATURES
# ============================================================

NUMERICAL_FEATURES = [
    "property_value",
    "dtir1",
    "LTV",
    "income",
    "loan_amount",
    "LTV_calculated",
    "Payment_to_Income",
    "Loan_to_Income",
    "DTI_x_LTV",
]


# ============================================================
# BINARY / ORDINAL FEATURES
# ============================================================

BINARY_FEATURES = [
    "property_value_isna",
    "dtir1_isna",
    "income_isna",
    "LTV_isna",
    "age_ordinal",
    "total_units_ordinal",
]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_FEATURES = [
    "loan_type",
    "loan_purpose",
    "occupancy_type",
    "submission_of_application",
    "Neg_ammortization",
    "lump_sum_payment",
    "co-applicant_credit_type",
]


# ============================================================
# LOGGING
# ============================================================

LOG_DIR = PROJECT_ROOT / "logs"

LOG_FILE = LOG_DIR / "pipeline.log"


# ============================================================
# REPORT DIRECTORIES
# ============================================================

REPORT_DIR = PROJECT_ROOT / "reports"

METRICS_DIR = REPORT_DIR / "metrics"

FIGURES_DIR = REPORT_DIR / "figures"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

DEFAULT_THRESHOLD = 0.50

DEFAULT_MODEL = "random_forest"

DEFAULT_MODEL_PATH = RF_MODEL_PATH

DEFAULT_MODEL_NAME = "random_forest"

DEFAULT_MODEL_VERSION = RF_MODEL_VERSION


# ============================================================
# REQUIRED DIRECTORIES
# ============================================================

DIRECTORIES = [
    # Data
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    # Models
    MODEL_DIR,
    LOGISTIC_MODEL_DIR,
    RF_MODEL_DIR,
    XGB_MODEL_DIR,
    # Logs
    LOG_DIR,
    # Reports
    REPORT_DIR,
    METRICS_DIR,
    FIGURES_DIR,
]


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for directory in DIRECTORIES:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

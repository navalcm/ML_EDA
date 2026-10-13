"""
Loan Default Prediction Flask API
----------------------------------
Provides:
    GET  /health
    POST /predict
"""

import json
import logging
from typing import Any, Dict

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from src.config import MODEL_PATH
from src.features import execute_feature_engineering

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# MODEL PREDICTOR
# ============================================================


class LoanDefaultPredictor:
    """Production predictor for loan default scoring."""

    def __init__(self, model_path=MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}. Train model first.")

        self.model_path = model_path
        self.model = joblib.load(model_path)

        logger.info("Model loaded successfully: %s", model_path)

    # --------------------------------------------------------
    # Predict single application
    # --------------------------------------------------------

    def predict_single(self, application_dict: Dict[str, Any]) -> Dict[str, Any]:

        logger.info("Prediction request received")

        # Convert input dictionary into DataFrame
        df_single = pd.DataFrame([application_dict])

        # Feature engineering
        df_fe = execute_feature_engineering(df_single)

        # ----------------------------------------------------
        # Align features with model expected features
        # ----------------------------------------------------

        if hasattr(self.model, "feature_names_in_"):
            expected_cols = list(self.model.feature_names_in_)

            # Add missing columns
            for col in expected_cols:
                if col not in df_fe.columns:
                    df_fe[col] = 0

            # Keep only expected columns
            df_input = df_fe[expected_cols]

        else:
            df_input = df_fe

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prob = float(self.model.predict_proba(df_input)[0, 1])

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        if prob >= 0.50:
            decision = "REJECT / HIGH RISK"
        else:
            decision = "APPROVE / LOW RISK"

        # ----------------------------------------------------
        # Risk score
        # ----------------------------------------------------

        risk_score = int((1 - prob) * 850)

        logger.info(
            "Prediction completed | probability=%.4f | decision=%s",
            prob,
            decision,
        )

        return {
            "default_probability": round(prob, 4),
            "underwriting_recommendation": decision,
            "risk_score": risk_score,
        }


# ============================================================
# LOAD MODEL ONCE
# ============================================================

try:
    predictor = LoanDefaultPredictor()

    logger.info("Loan Default Predictor initialized successfully")

except Exception as exc:
    predictor = None

    logger.exception("Failed to initialize predictor: %s", exc)


# ============================================================
# HEALTH CHECK API
# ============================================================


@app.route("/health", methods=["GET"])
def health():

    logger.info("Health check requested")

    if predictor is None:
        return jsonify({"status": "unhealthy", "model_loaded": False}), 503

    return jsonify({"status": "healthy", "model_loaded": True}), 200


# ============================================================
# PREDICT API
# ============================================================


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get JSON request
        application_data = request.get_json()

        if not application_data:
            return jsonify({"error": "Request body must contain JSON data"}), 400

        logger.info("Received /predict request with %d fields", len(application_data))

        # Make prediction
        result = predictor.predict_single(application_data)

        return jsonify(result), 200

    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)

        return jsonify({"error": "Prediction failed", "message": str(exc)}), 500


# ============================================================
# SAMPLE APPLICATION
# ============================================================

sample_application = {
    "loan_amount": 300000,
    "property_value": 400000,
    "income": 7500,
    "term": 360,
    "dtir1": 42.0,
    "LTV": 75.0,
    "loan_type": "type1",
    "loan_purpose": "p1",
    "occupancy_type": "pr",
    "submission_of_application": "to_inst",
    "Neg_ammortization": "not_neg",
    "lump_sum_payment": "not_lpsm",
    "co-applicant_credit_type": "CIB",
    "age": "35-44",
    "total_units": "1U",
}


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # Show sample prediction
    # --------------------------------------------------------

    if predictor is not None:
        result = predictor.predict_single(sample_application)

        print("=" * 60)
        print("SAMPLE API INFERENCE RESPONSE")
        print("=" * 60)
        print(json.dumps(result, indent=2))
        print("=" * 60)

    # --------------------------------------------------------
    # START FLASK SERVER
    # --------------------------------------------------------

    logger.info("Starting Flask API...")
    logger.info("Health endpoint: http://127.0.0.1:5000/health")
    logger.info("Predict endpoint: http://127.0.0.1:5000/predict")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )

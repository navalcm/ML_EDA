"""
Main entry point for model training.
"""

from src.logger import get_logger
from src.modeling.train import train_baseline_models

# ============================================================
# LOGGER
# ============================================================

logger = get_logger(__name__)


# ============================================================
# MAIN
# ============================================================


def main():
    """Run the baseline model training pipeline."""

    logger.info("=" * 60)
    logger.info("ML MODEL TRAINING PIPELINE STARTED")
    logger.info("=" * 60)

    try:
        # ----------------------------------------------------
        # Start model training
        # ----------------------------------------------------

        logger.info("Starting baseline model training...")

        train_baseline_models()

        # ----------------------------------------------------
        # Training completed
        # ----------------------------------------------------

        logger.info("Baseline model training completed successfully.")

    except Exception:
        # ----------------------------------------------------
        # Log complete error traceback
        # ----------------------------------------------------

        logger.exception("Model training pipeline failed.")

        # Re-raise error so the program exits with failure
        raise

    finally:
        logger.info("=" * 60)
        logger.info("ML MODEL TRAINING PIPELINE FINISHED")
        logger.info("=" * 60)


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()

"""
Dataset loading module.

Loads and summarizes the raw Loan Default dataset
from the configured raw data location.
"""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_FILE, TARGET_COL
from src.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# LOAD RAW DATA
# ============================================================


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw Loan Default dataset.

    Returns
    -------
    pd.DataFrame
        Raw Loan Default dataset.
    """

    logger.info("=" * 70)
    logger.info("LOADING RAW DATA")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # Validate raw dataset path
    # --------------------------------------------------------

    raw_data_path = Path(RAW_DATA_FILE)

    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_data_path}")

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    logger.info(
        "Loading dataset from: %s",
        raw_data_path,
    )

    df = pd.read_csv(raw_data_path)

    logger.info(
        "Raw dataset shape: %s",
        df.shape,
    )

    # --------------------------------------------------------
    # Validate target column
    # --------------------------------------------------------

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    logger.info(
        "Target column: %s",
        TARGET_COL,
    )

    logger.info("Dataset loaded successfully.")

    return df


# ============================================================
# DATASET SUMMARY
# ============================================================


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a dataset summary as a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Dataset summary table.
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # --------------------------------------------------------
    # Create summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "missing_values",
                "duplicate_rows",
                "target_column",
            ],
            "value": [
                int(df.shape[0]),
                int(df.shape[1]),
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
                TARGET_COL,
            ],
        }
    )

    # --------------------------------------------------------
    # Add target distribution
    # --------------------------------------------------------

    if TARGET_COL in df.columns:
        target_distribution = df[TARGET_COL].value_counts(dropna=False).to_dict()

        summary = pd.concat(
            [
                summary,
                pd.DataFrame(
                    {
                        "metric": ["target_distribution"],
                        "value": [target_distribution],
                    }
                ),
            ],
            ignore_index=True,
        )

    logger.info(
        "Dataset summary generated:\n%s",
        summary,
    )

    return summary


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("DATASET LOADING TEST")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    data = load_raw_data()

    # --------------------------------------------------------
    # Generate summary
    # --------------------------------------------------------

    summary = dataset_summary(data)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nDataset shape:")
    print(data.shape)

    print("\nColumns:")
    print(data.columns.tolist())

    print("\nFirst 5 rows:")
    print(data.head())

    print("\nDataset Summary:")
    print(summary)

# Pandas-based cleaning for the CSV catalog data. Applied to each DataFrame
# right after pd.read_csv in db.py, before rows are converted into Pydantic
# model instances -- so agents.py never sees duplicate, incomplete, or
# out-of-range rows.

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace on every text column, and treat a field that's
    whitespace-only (e.g. "   ") as missing rather than as real content."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip().replace("", pd.NA)
    return df


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the product catalog: dedupe rows and ids, drop rows missing an
    identifying field, and clip numeric fields to their valid ranges."""
    before = len(df)
    df = _strip_string_columns(df)

    df = df.drop_duplicates()
    # Also dedupe on id alone: two rows sharing an id but differing in any
    # other column survive the full-row dedupe above, and would otherwise
    # give one product two conflicting catalog entries.
    df = df.drop_duplicates(subset=["id"], keep="first")

    # Fields an agent can't work without -- a product with no id, name,
    # category, or price can't be matched, filtered, or quoted, so drop it
    # rather than carry a half-usable row.
    df = df.dropna(subset=["id", "name", "category", "price"])

    # Everything else is optional: default it instead, so a missing brand or
    # rating doesn't cost an otherwise-valid product.
    df["brand"] = df["brand"].fillna("Unknown")
    df["description"] = df["description"].fillna("")
    df["stock"] = df["stock"].fillna(0).clip(lower=0).astype(int)
    df["rating"] = df["rating"].fillna(0.0).clip(lower=0.0, upper=5.0)
    df = df[df["price"] > 0]  # a free/negative-priced product is bad data, not a deal

    dropped = before - len(df)
    if dropped:
        logger.warning("dropped %d invalid/duplicate rows while cleaning products.csv", dropped)

    return df.reset_index(drop=True)


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Clean reviews: drop duplicate/incomplete rows, clip ratings to
    [0, 5], and drop rows whose date doesn't parse."""
    before = len(df)
    df = _strip_string_columns(df)

    df = df.drop_duplicates()
    df = df.dropna(subset=["product_id", "rating", "text", "date"])

    df["rating"] = df["rating"].clip(lower=0.0, upper=5.0)

    # errors="coerce" turns unparseable dates into NaT rather than raising,
    # so one malformed date drops its own row instead of the whole load.
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    df = df[parsed_dates.notna()]

    dropped = before - len(df)
    if dropped:
        logger.warning("dropped %d invalid/duplicate rows while cleaning reviews.csv", dropped)

    return df.reset_index(drop=True)


def clean_store_policies(df: pd.DataFrame) -> pd.DataFrame:
    """Clean store policies: drop duplicate/incomplete rows."""
    before = len(df)
    df = _strip_string_columns(df)

    df = df.drop_duplicates()
    df = df.dropna(subset=["policy_type", "description", "conditions"])
    # Kept as a string ("0", not 0) to match StorePolicy.timeframe, which is
    # typed str because the column mixes day counts with non-numeric values.
    df["timeframe"] = df["timeframe"].fillna("0")

    dropped = before - len(df)
    if dropped:
        logger.warning("dropped %d invalid/duplicate rows while cleaning store_policies.csv", dropped)

    return df.reset_index(drop=True)

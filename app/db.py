# Import necessary modules for handling CSV files
import functools  # lru_cache: load + clean each CSV once per process, not once per request
import pandas as pd  # For handling CSV data
from .data_cleaning import clean_products, clean_reviews, clean_store_policies
from .models import Product, Review, StorePolicy  # Import models

# Implement Functions to Load Data
#
# Every specialized agent in agents.py calls these on its own __init__, and
# agents are re-instantiated per request -- so without caching, the CSVs
# would be re-read, re-parsed, and re-cleaned on every single query.
# lru_cache(maxsize=1) makes each function do that work exactly once per
# process and hand back the same in-memory tuple thereafter. Returning a
# tuple (not a list) means callers can't accidentally mutate the shared
# cached data in place -- see DECISIONS.md for the full rationale.

# Load Products from CSV
@functools.lru_cache(maxsize=1)
def load_products():
    """Loads, cleans, and caches product data from CSV; returns a tuple of Product objects."""
    df = pd.read_csv('data/products.csv')  # Read 'products.csv' into a pandas DataFrame
    df = clean_products(df)  # Dedupe, drop incomplete rows, clip numeric ranges

    # Convert DataFrame rows into a tuple of Product objects
    return tuple(Product(**row) for row in df.to_dict(orient='records'))


# Load Reviews from CSV
@functools.lru_cache(maxsize=1)
def load_reviews():
    """Loads, cleans, and caches review data from CSV; returns a tuple of Review objects."""
    df = pd.read_csv('data/reviews.csv')
    df = clean_reviews(df)  # Dedupe, drop incomplete rows, clip ratings, drop bad dates

    # Convert DataFrame rows into a tuple of Review objects
    return tuple(Review(**row) for row in df.to_dict(orient='records'))


# Load Store Policies from CSV
@functools.lru_cache(maxsize=1)
def load_store_policies():
    """Loads, cleans, and caches store policy data from CSV; returns a tuple of StorePolicy objects."""
    df = pd.read_csv('data/store_policies.csv', dtype={'timeframe': str})
    df = clean_store_policies(df)  # Dedupe, drop incomplete rows

    # Convert DataFrame rows into a tuple of StorePolicy objects
    return tuple(StorePolicy(**row) for row in df.to_dict(orient='records'))

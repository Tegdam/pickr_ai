# Import necessary modules for handling CSV files
import pandas as pd  # For handling CSV data
from .data_cleaning import clean_products, clean_reviews, clean_store_policies
from .models import Product, Review, StorePolicy  # Import models

# Implement Functions to Load Data

# Load Products from CSV
def load_products():
    """Loads product data from CSV and returns a list of Product objects."""
    df = pd.read_csv('data/products.csv')  # Read 'products.csv' into a pandas DataFrame
    df = clean_products(df)  # Dedupe, drop incomplete rows, clip numeric ranges
    products = []  # Create an empty list to store products

    # Convert DataFrame rows into a list of Product objects
    for row in df.to_dict(orient='records'):
        products.append(Product(**row))  # Create Product instances and append to the products list

    return products  # Return the list of products


# Load Reviews from CSV
def load_reviews():
    """Loads review data from CSV and returns a list of Review objects."""
    df = pd.read_csv('data/reviews.csv')
    df = clean_reviews(df)  # Dedupe, drop incomplete rows, clip ratings, drop bad dates
    reviews = []

    # Convert DataFrame rows into a list of Review objects
    for row in df.to_dict(orient='records'):
        reviews.append(Review(**row)) # Create Review instances and append to the reviews list
        
    return reviews # Return the list of reviews


# Load Store Policies from CSV
def load_store_policies():
    """Loads store policy data from CSV and returns a list of StorePolicy objects."""
    df = pd.read_csv('data/store_policies.csv', dtype={'timeframe': str})
    df = clean_store_policies(df)  # Dedupe, drop incomplete rows
    policies = []

    # Convert DataFrame rows into a list of StorePolicy objects
    for row in df.to_dict(orient='records'):
        policies.append(StorePolicy(**row)) # Create StorePolicy instances and append to the policies list

    return policies # Return the list of store policies

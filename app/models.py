# Import necessary modules for data validation
from pydantic import BaseModel  # For data validation
from typing import Optional  # For optional model attributes

# Define Pydantic Models

# Define the UserQuery Model
class UserQuery(BaseModel):
    query: str  # Define a string field for storing user queries

# Define the Product Model
class Product(BaseModel):
    id: Optional[str]  # Unique product identifier
    name: Optional[str]  # Product name
    brand: Optional[str]  # Brand name
    category: Optional[str]  # Product category, e.g., Electronics, Clothing
    price: Optional[float]  # Product price
    description: Optional[str]  # Brief product description
    stock: Optional[int]  # Available stock quantity
    rating: Optional[float]  # Customer rating out of 5

# Define the Review Model
class Review(BaseModel):
    product_id: Optional[str] # ID of the product being reviewed
    rating: Optional[float]  # Customer rating out of 5
    text: Optional[str]  # Review text
    date: Optional[str]  # Date of the review in MM-DD-YYYY format   

# Define the Store Policy Model
class StorePolicy(BaseModel):
    policy_type: Optional[str]  # Type of policy, e.g., Return, Warranty
    description: Optional[str]  # Detailed description of the policy
    conditions: Optional[str]  # Conditions under which the policy applies
    timeframe: Optional[str]  # Timeframe for the policy, e.g., 30 days for returns

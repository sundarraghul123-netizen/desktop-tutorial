"""Central configuration. Secrets come from .env only (never hard-coded)."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv optional at import time
    pass

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.getenv("DATABASE_PATH", str(BASE_DIR / "upi_finance.db"))
MODEL_DIR = BASE_DIR / "models"
SAMPLE_PATH = BASE_DIR / "data" / "sample" / "sample_upi_statement.csv"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
LOW_CONF = 0.70                      # below this a transaction goes to the review queue
EXCLUDED = {"Transfer", "Refund"}    # never counted as income or expense

# "Utilities" merged into Bills; streaming lives in Subscriptions; ride-hailing is a subcategory, not a merchant.
CATEGORIES = {
    "Food": ["Restaurants", "Food Delivery", "Groceries", "Cafes"],
    "Shopping": ["Online Shopping", "Clothing", "Electronics", "General"],
    "Transportation": ["Cab", "Metro", "Fuel", "Public Transport"],
    "Bills": ["Electricity", "Water", "Gas", "Internet", "Mobile", "EMI / Loan"],
    "Rent": ["Rent"],
    "Healthcare": ["Pharmacy", "Doctor", "Fitness"],
    "Entertainment": ["Movies", "Games", "Events"],
    "Education": ["Courses", "Fees"],
    "Travel": ["Flights", "Train", "Hotels", "General"],
    "Investment": ["Mutual Funds", "Stocks"],
    "Insurance": ["Life", "Health", "Vehicle", "General"],
    "Subscriptions": ["Streaming", "Apps", "Software"],
    "Bank Charges": ["Bank Charges"],
    "Other": ["Cash Withdrawal", "Other"],
}
INCOME_CATS = {
    "Salary": ["Salary"], "Interest": ["Interest"], "Other Income": ["Other Income"],
    "Refund": ["Refund"], "Transfer": ["Self Transfer"],
}
ALL_CATS = {**CATEGORIES, **INCOME_CATS}

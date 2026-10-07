"""Merchant extraction/normalisation and multi-level categorisation.
Order: user-confirmed map -> known-merchant DB -> keyword rules -> ML classifier -> Other (low confidence)."""
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

if __package__:
    from .config import CATEGORIES, MODEL_DIR
else:
    from config import CATEGORIES, MODEL_DIR

# (canonical name, category, subcategory, aliases) - first match wins, so specific entries come first.
KNOWN = [
    ("Amazon Prime", "Subscriptions", "Streaming", ["amazon prime", "prime video", "primevideo"]),
    ("Swiggy", "Food", "Food Delivery", ["swiggy"]), ("Zomato", "Food", "Food Delivery", ["zomato"]),
    ("Blinkit", "Food", "Groceries", ["blinkit", "grofers"]), ("Zepto", "Food", "Groceries", ["zepto"]),
    ("BigBasket", "Food", "Groceries", ["bigbasket", "bbdaily"]), ("DMart", "Food", "Groceries", ["dmart", "avenue supermarts"]),
    ("Starbucks", "Food", "Cafes", ["starbucks"]), ("Cafe Coffee Day", "Food", "Cafes", ["cafe coffee day", "ccd"]),
    ("Dominos", "Food", "Restaurants", ["dominos", "domino s"]), ("McDonalds", "Food", "Restaurants", ["mcdonalds", "mcdonald s", "mcd"]),
    ("KFC", "Food", "Restaurants", ["kfc"]), ("Pizza Hut", "Food", "Restaurants", ["pizza hut"]),
    ("Amazon", "Shopping", "Online Shopping", ["amazon", "amzn"]), ("Flipkart", "Shopping", "Online Shopping", ["flipkart"]),
    ("Meesho", "Shopping", "Online Shopping", ["meesho"]), ("Myntra", "Shopping", "Clothing", ["myntra"]),
    ("Ajio", "Shopping", "Clothing", ["ajio"]), ("Nykaa", "Shopping", "General", ["nykaa"]),
    ("Decathlon", "Shopping", "General", ["decathlon"]), ("Croma", "Shopping", "Electronics", ["croma", "reliance digital"]),
    ("Uber", "Transportation", "Cab", ["uber"]), ("Ola", "Transportation", "Cab", ["ola", "olacabs", "ola cabs"]),
    ("Rapido", "Transportation", "Cab", ["rapido"]), ("Metro", "Transportation", "Metro", ["dmrc", "bmrcl", "metro"]),
    ("Indian Oil", "Transportation", "Fuel", ["indian oil", "iocl"]), ("HPCL", "Transportation", "Fuel", ["hpcl", "hindustan petroleum"]),
    ("BPCL", "Transportation", "Fuel", ["bpcl", "bharat petroleum"]), ("Shell", "Transportation", "Fuel", ["shell"]),
    ("IRCTC", "Travel", "Train", ["irctc"]), ("RedBus", "Travel", "General", ["redbus"]),
    ("MakeMyTrip", "Travel", "General", ["makemytrip", "mmt"]), ("Goibibo", "Travel", "General", ["goibibo"]),
    ("Cleartrip", "Travel", "General", ["cleartrip"]), ("IndiGo", "Travel", "Flights", ["indigo"]),
    ("Netflix", "Subscriptions", "Streaming", ["netflix"]), ("Spotify", "Subscriptions", "Streaming", ["spotify"]),
    ("Hotstar", "Subscriptions", "Streaming", ["hotstar", "disney"]), ("YouTube Premium", "Subscriptions", "Streaming", ["youtube"]),
    ("Google One", "Subscriptions", "Apps", ["google one", "google play", "googleplay"]),
    ("Airtel", "Bills", "Mobile", ["airtel"]), ("Jio", "Bills", "Mobile", ["jio", "reliance jio"]),
    ("Vodafone Idea", "Bills", "Mobile", ["vodafone", "vodafone idea"]), ("BSNL", "Bills", "Mobile", ["bsnl"]),
    ("ACT Fibernet", "Bills", "Internet", ["act fibernet", "act broadband"]),
    ("BESCOM", "Bills", "Electricity", ["bescom"]), ("Tata Power", "Bills", "Electricity", ["tata power"]),
    ("Adani Electricity", "Bills", "Electricity", ["adani electricity"]),
    ("Zerodha", "Investment", "Stocks", ["zerodha"]), ("Groww", "Investment", "Mutual Funds", ["groww"]),
    ("Upstox", "Investment", "Stocks", ["upstox"]),
    ("LIC", "Insurance", "Life", ["lic", "life insurance corporation"]), ("HDFC Life", "Insurance", "Life", ["hdfc life"]),
    ("Star Health", "Insurance", "Health", ["star health"]), ("Acko", "Insurance", "Vehicle", ["acko"]),
    ("Apollo Pharmacy", "Healthcare", "Pharmacy", ["apollo"]), ("PharmEasy", "Healthcare", "Pharmacy", ["pharmeasy"]),
    ("1mg", "Healthcare", "Pharmacy", ["1mg"]), ("Netmeds", "Healthcare", "Pharmacy", ["netmeds"]),
    ("Practo", "Healthcare", "Doctor", ["practo"]), ("Cult.fit", "Healthcare", "Fitness", ["cult fit", "cultfit"]),
    ("BookMyShow", "Entertainment", "Movies", ["bookmyshow", "bms"]), ("PVR INOX", "Entertainment", "Movies", ["pvr", "inox"]),
    ("Udemy", "Education", "Courses", ["udemy"]), ("Coursera", "Education", "Courses", ["coursera"]),
    ("Unacademy", "Education", "Courses", ["unacademy"]), ("Byjus", "Education", "Courses", ["byju", "byjus"]),
]


def _clean(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


KNOWN_C = [(n, c, s, re.compile(r"\b(?:" + "|".join(re.escape(a) for a in al) + r")\b")) for n, c, s, al in KNOWN]

NOISE = {"upi", "imps", "neft", "rtgs", "pos", "atm", "ecom", "paytm", "phonepe", "gpay", "okaxis", "okhdfcbank", "okicici",
         "oksbi", "ybl", "ibl", "axl", "apl", "paid", "payment", "payments", "pay", "to", "from", "by", "txn", "ref", "online",
         "pvt", "ltd", "limited", "private", "india", "bank", "dr", "cr", "debit", "credit", "mumbai", "bangalore", "bengaluru",
         "delhi", "chennai", "hyderabad", "pune", "kolkata", "upiint", "transfer"}


def extract_merchant(desc):
    """Return (normalised merchant, recognised_in_known_db)."""
    c = _clean(desc)
    for name, _, _, rx in KNOWN_C:
        if rx.search(c):
            return name, True
    s = re.sub(r"@\S+", " ", str(desc))            # drop VPA handle (name@bank)
    toks = [t for t in _clean(s).split()
            if t not in NOISE and not t.isdigit() and len(t) > 1 and not re.fullmatch(r"[a-z]*\d{4,}[a-z\d]*", t)]
    return (" ".join(toks[:2]).title() or "Unknown"), False


# (regex on cleaned description, category, subcategory, required type or None)
RULES = [(re.compile(p), c, s, t) for p, c, s, t in [
    (r"\b(?:self|own account|own acc|to self)\b", "Transfer", "Self Transfer", None),
    (r"\b(?:salary|payroll|sal cr|stipend)\b", "Salary", "Salary", "Income"),
    (r"\b(?:interest|int pd|int cr)\b", "Interest", "Interest", "Income"),
    (r"\b(?:refund|reversal|cashback|chargeback)\b", "Refund", "Refund", "Income"),
    (r"\b(?:rent|landlord)\b", "Rent", "Rent", "Expense"),
    (r"\b(?:emi|loan)\b", "Bills", "EMI / Loan", "Expense"),
    (r"\b(?:electricity|power bill|discom)\b", "Bills", "Electricity", "Expense"),
    (r"\b(?:water bill|water board)\b", "Bills", "Water", "Expense"),
    (r"\b(?:broadband|fibernet|wifi|internet)\b", "Bills", "Internet", "Expense"),
    (r"\b(?:recharge|prepaid|postpaid|mobile bill)\b", "Bills", "Mobile", "Expense"),
    (r"\b(?:insurance|premium|policy)\b", "Insurance", "General", "Expense"),
    (r"\b(?:mutual fund|sip|demat)\b", "Investment", "Mutual Funds", "Expense"),
    (r"\b(?:atm|cash withdrawal|cash wdl)\b", "Other", "Cash Withdrawal", "Expense"),
    (r"\b(?:tuition|school|college|course|exam)\b", "Education", "Fees", "Expense"),
    (r"\b(?:hospital|clinic|pharmacy|medical|doctor|diagnostic|dental)\b", "Healthcare", "Doctor", "Expense"),
    (r"\b(?:fee|fees|charges|gst|penalty|amc)\b", "Bank Charges", "Bank Charges", "Expense"),
    (r"\b(?:petrol|diesel|fuel)\b", "Transportation", "Fuel", "Expense"),
    (r"\b(?:cafe|coffee|chai|tea)\b", "Food", "Cafes", "Expense"),
    (r"\b(?:restaurant|dhaba|biryani|kitchen|bakery|sweets|tiffins?)\b", "Food", "Restaurants", "Expense"),
    (r"\b(?:grocery|kirana|supermarket|vegetables?|milk)\b", "Food", "Groceries", "Expense"),
    (r"\b(?:flight|airlines|resort|hostel)\b", "Travel", "General", "Expense"),
    (r"\b(?:movie|cinema|gaming|concert)\b", "Entertainment", "Movies", "Expense"),
]]

EXTRA_SEEDS = {
    "Food": ["restaurant", "biryani", "bakery", "sweets", "dhaba", "kitchen", "grocery", "kirana", "vegetables", "milk",
             "juice", "chai", "tea", "coffee", "tiffins", "foods", "mess", "canteen"],
    "Shopping": ["store", "mart", "fashion", "mall", "electronics", "garments", "boutique", "traders", "enterprises"],
    "Transportation": ["taxi", "cab", "fuel", "petrol", "parking", "fastag", "toll", "auto", "travels bus"],
    "Healthcare": ["hospital", "clinic", "pharmacy", "medical", "diagnostic", "dental", "lab", "gym", "fitness"],
    "Education": ["school", "college", "tuition", "academy", "institute", "classes"],
    "Entertainment": ["cinema", "movie", "gaming", "concert", "theatre"],
    "Bills": ["recharge", "broadband", "electricity", "gas agency", "dth", "water"],
    "Rent": ["rent", "landlord", "pg owner"],
}


def seed_data():
    X, y = [], []
    for _, cat, _, al in KNOWN:
        for a in al:
            X.append(a); y.append(cat)
    for cat, words in EXTRA_SEEDS.items():
        X += words; y += [cat] * len(words)
    return X, y


def _pipe():
    return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)), LogisticRegression(max_iter=500, C=10))


MODEL_PATH = MODEL_DIR / "category_classifier.joblib"


class MLClassifier:
    """TF-IDF + logistic regression. Seeded from the merchant DB so it works before any user corrections."""

    def __init__(self, pipe=None):
        self.pipe = pipe

    @classmethod
    def load_or_seed(cls):
        try:
            if MODEL_PATH.exists():
                return cls(joblib.load(MODEL_PATH))     # trusted local file written by this app only
        except Exception:
            pass
        X, y = seed_data()
        m = cls(_pipe().fit(X, y))
        m.save()
        return m

    def save(self):
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(self.pipe, MODEL_PATH)

    def predict(self, texts):
        p = self.pipe.predict_proba(texts)
        i = p.argmax(1)
        return self.pipe.classes_[i], p.max(1)

    def retrain(self, user_pairs, min_labels=20):
        """Retrain with confirmed corrections; deploy only if it is not worse on held-out user labels."""
        if len(user_pairs) < min_labels or len({c for _, c in user_pairs}) < 2:
            return f"Need at least {min_labels} confirmed merchants (have {len(user_pairs)}); model unchanged."
        X, y = zip(*user_pairs)
        Xtr, Xte, ytr, yte = train_test_split(list(X), list(y), test_size=0.25, random_state=42)
        sx, sy = seed_data()
        new = _pipe().fit(sx + Xtr * 3, sy + ytr * 3)
        a_new, a_old = new.score(Xte, yte), (self.pipe.score(Xte, yte) if self.pipe is not None else 0)
        if a_new >= a_old:
            self.pipe = _pipe().fit(sx + list(X) * 3, sy + list(y) * 3)   # final fit on all confirmed labels
            self.save()
            return f"New model deployed (held-out accuracy {a_new:.0%} vs {a_old:.0%})."
        return f"New model rejected (held-out accuracy {a_new:.0%} < current {a_old:.0%}); kept current model."


def categorize_one(desc, ttype, user_map, ml):
    merchant, known = extract_merchant(desc)
    c = _clean(desc)
    key = merchant.lower()
    if key in user_map:
        cat, sub = user_map[key]
        return merchant, cat, sub, 1.0, "user", f"Classified as {cat} because you confirmed this merchant ({merchant})."
    if known:
        for n, cat, sub, _ in KNOWN_C:
            if n == merchant:
                if ttype == "Income":
                    return merchant, "Refund", "Refund", 0.8, "merchant_db", f"Credit from {merchant}; treated as a refund."
                return merchant, cat, sub, 0.97, "merchant_db", f"Classified as {cat} because the merchant is recognised as {merchant}."
    for rx, cat, sub, t in RULES:
        if (t is None or t == ttype) and rx.search(c):
            return merchant, cat, sub, 0.85, "rule", f"Classified as {cat} because the description contains a {cat.lower()} keyword."
    if ttype == "Income":
        return merchant, "Other Income", "Other Income", 0.7, "default", "Credit with no recognised source; treated as other income."
    if ml is not None and ml.pipe is not None:
        lab, p = ml.predict([f"{merchant} {c}"])
        if p[0] >= 0.6:
            return merchant, str(lab[0]), CATEGORIES[str(lab[0])][0], float(p[0]), "ml", f"Predicted {lab[0]} by the text classifier ({p[0]:.0%})."
        return merchant, "Other", "Other", float(min(p[0], 0.5)), "ml", "Not recognised; please confirm the category."
    return merchant, "Other", "Other", 0.4, "default", "Not recognised; please confirm the category."


def categorize_df(df, user_map=None, ml=None):
    user_map = user_map or {}
    cache, rows = {}, []
    for desc, t in zip(df.description, df.transaction_type):
        k = (desc, t)
        if k not in cache:
            cache[k] = categorize_one(desc, t, user_map, ml)
        rows.append(cache[k])
    out = df.copy()
    out[["merchant", "category", "subcategory", "category_confidence", "category_source", "category_reason"]] = pd.DataFrame(rows, index=df.index)
    return out

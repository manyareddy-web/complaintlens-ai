"""
complaint_analyzer.py
=====================
The "brain" of ComplaintLens AI.

Given a complaint text, this module works out:
    * Category          (machine learning + keyword rules)
    * Sentiment         (lexicon based, optional Transformers model)
    * Priority          (explainable scoring system)
    * Urgency           (explainable scoring system)
    * Keywords          (phrase matching + domain vocabulary)
    * Summary           (extractive summary)
    * Recommended action

Everything works OFFLINE on a normal laptop. Heavy libraries (Transformers /
PyTorch) are only used if you switch them on in the .env file.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# NLTK is used for tokenizing, stemming and stop-words. If it is not installed
# we fall back to simple built-in versions so the app never crashes.
try:
    from nltk.stem import PorterStemmer
    from nltk.tokenize import RegexpTokenizer

    NLTK_AVAILABLE = True
except ImportError:  # pragma: no cover
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "complaints.csv"
MODEL_PATH = BASE_DIR / "models" / "category_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

CATEGORIES = [
    "Payment Issue",
    "Delivery Issue",
    "Product Issue",
    "Account Issue",
    "Technical Issue",
    "Service Issue",
    "Refund Issue",
    "Other",
]
PRIORITY_LEVELS = ["Low", "Medium", "High", "Critical"]
SENTIMENTS = ["Positive", "Neutral", "Negative"]

MIN_LENGTH = 10      # shortest complaint we accept (characters)
MAX_LENGTH = 2000    # longest complaint we accept (characters)

# Optional Transformers model (only used when USE_TRANSFORMERS=true in .env)
TRANSFORMER_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


# --------------------------------------------------------------------------
# Custom errors (the app shows these as friendly messages)
# --------------------------------------------------------------------------
class InvalidComplaintError(ValueError):
    """Raised when the complaint text is empty or not usable."""


class DatasetMissingError(FileNotFoundError):
    """Raised when data/complaints.csv cannot be found or is unreadable."""


# --------------------------------------------------------------------------
# 1. TEXT PREPROCESSING  (cleaning, tokenization, stop-words, stemming)
# --------------------------------------------------------------------------
FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with",
    "to", "from", "in", "on", "up", "out", "over", "under", "so", "than", "too",
    "very", "s", "t", "just", "is", "am", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing", "i",
    "me", "my", "myself", "we", "our", "you", "your", "he", "him", "his", "she",
    "her", "it", "its", "they", "them", "their", "this", "that", "these",
    "those", "what", "which", "who", "whom", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "own", "same", "will", "would", "can", "could", "should", "about",
    "into", "through", "during", "before", "after", "above", "below", "again",
    "further", "then", "once", "here", "there", "because", "as", "until",
    "while", "also", "even", "though", "although", "yet", "still", "ago",
}

# Negation words carry meaning ("not working"), so we never delete them.
KEEP_WORDS = {"not", "no", "nor", "never", "without", "cannot", "unable", "down", "off"}

# Words that are poor keywords even though they are not stop-words.
KEYWORD_EXCLUDE = {
    "not", "no", "nor", "never", "without", "cannot", "would", "like", "want",
    "know", "please", "thank", "thanks", "got", "get", "gets", "one", "two",
    "three", "since", "today", "yesterday", "made", "make", "using", "use",
    "try", "tried", "trying", "supposed", "promised", "nobody", "nothing",
    "something", "anything", "much", "many", "last", "next", "every", "give",
    "gave", "given", "need", "said", "says", "say", "tell", "told", "shows",
    "show", "going", "went", "come", "came", "take", "took", "see", "seen",
    "check", "checked", "keeps", "keep", "kept", "able", "different", "getting",
    "five", "four", "six", "seven", "eight", "nine", "ten", "days", "ago",
    "day", "time", "times", "thing", "things", "back", "new", "small", "big",
}


def load_stopwords() -> set[str]:
    """Combine NLTK's stop-word list (if downloaded) with our built-in list."""
    words = set(FALLBACK_STOPWORDS)
    try:
        from nltk.corpus import stopwords

        words |= set(stopwords.words("english"))
    except Exception:  # corpus not downloaded -> built-in list is enough
        pass
    return {w for w in words if w not in KEEP_WORDS and not w.endswith("n't")}


STOPWORDS = load_stopwords()

if NLTK_AVAILABLE:
    _TOKENIZER = RegexpTokenizer(r"[a-z]+(?:'[a-z]+)?|\d+")
    _STEMMER = PorterStemmer()

    def tokenize(text: str) -> list[str]:
        """Split text into lowercase word tokens (NLTK regexp tokenizer)."""
        return _TOKENIZER.tokenize(text.lower())

    @lru_cache(maxsize=20000)
    def stem(word: str) -> str:
        """Reduce a word to its root, e.g. 'delivered' -> 'deliv'."""
        return _STEMMER.stem(word)

else:  # simple fallbacks when NLTK is missing

    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", text.lower())

    @lru_cache(maxsize=20000)
    def stem(word: str) -> str:
        for suffix in ("ing", "edly", "ed", "es", "ly", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word


def clean_text(text: str) -> str:
    """Lowercase the text and remove URLs, e-mails and odd symbols."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s'₹$]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str) -> str:
    """Full pipeline used before machine learning:
    clean -> tokenize -> remove stop-words -> stem -> join back to a string."""
    tokens = tokenize(clean_text(text))
    tokens = [stem(t) for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


# --------------------------------------------------------------------------
# 2. INPUT VALIDATION
# --------------------------------------------------------------------------
def validate_complaint(text) -> str:
    """Return the cleaned-up complaint text or raise InvalidComplaintError."""
    if not isinstance(text, str):
        raise InvalidComplaintError("The complaint must be text.")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise InvalidComplaintError("Please enter a complaint before clicking Analyze.")
    if len(text) < MIN_LENGTH:
        raise InvalidComplaintError(
            f"The complaint is too short. Please write at least {MIN_LENGTH} characters."
        )
    if len(text) > MAX_LENGTH:
        raise InvalidComplaintError(
            f"The complaint is too long. Please keep it under {MAX_LENGTH} characters."
        )
    if not re.search(r"[A-Za-z]{3,}", text):
        raise InvalidComplaintError(
            "The complaint does not look like a sentence. Please describe the problem in words."
        )
    return text


# --------------------------------------------------------------------------
# 3. CATEGORY DETECTION  (keyword rules + machine-learning model)
# --------------------------------------------------------------------------
# A keyword matches the START of a word ("deliver" matches deliver, delivered,
# delivery). A trailing "$" means "exact word only" (so "app$" will not match
# "apple").
CATEGORY_KEYWORDS = {
    "Payment Issue": [
        "payment", "pay$", "paid", "paying", "transaction", "deduct", "debit",
        "credit card", "card", "upi", "net banking", "netbanking", "bank",
        "charged$", "charges$", "invoice", "bill", "gateway", "wallet", "receipt", "coupon",
        "discount", "fee$", "fees$", "overcharg", "money", "amount",
    ],
    "Delivery Issue": [
        "deliver", "courier", "parcel", "package", "shipment", "ship",
        "transit", "tracking", "dispatch", "arrive", "doorstep", "rider",
        "logistics", "delay",
    ],
    "Product Issue": [
        "product", "item$", "damage", "broken", "defect", "faulty", "quality",
        "size", "colour", "color", "crack", "screen", "battery", "expired",
        "headphone", "laptop", "phone", "shirt", "shoe", "packet", "warranty",
        "fake", "counterfeit", "overheat", "swell", "spark", "burning",
        "grinder", "used product", "charger", "fire$", "smoke", "shock", "explo",
        "device", "appliance", "television", "tv$", "watch$", "bag$", "bottle",
    ],
    "Account Issue": [
        "account", "login", "log in", "logged", "password", "otp$", "locked",
        "hack", "profile", "username", "email address", "verif", "suspended",
        "two-factor", "authentication", "sign in", "signin", "register",
        "credentials", "unauthori",
    ],
    "Technical Issue": [
        "app$", "apps$", "website", "crash", "error", "bug$", "server",
        "loading", "slow", "hang", "freez", "glitch", "update", "timeout",
        "time out", "outage", "browser", "notification", "page$", "internet",
        "network", "disconnect", "checkout", "upload", "download", "install",
        "video call", "search feature", "blank", "not working",
    ],
    "Service Issue": [
        "support", "agent", "staff", "executive", "helpline", "customer care",
        "customer service", "technician", "engineer", "appointment", "rude",
        "behaviour", "behavior", "service center", "service centre",
        "representative", "reply", "replied", "ignored", "unhelpful", "chat",
        "escalat", "manager", "waiting time", "call$", "called",
    ],
    "Refund Issue": [
        "refund", "return", "money back", "reimburs", "cancel", "chargeback",
        "credited", "pickup", "replacement",
    ],
}

# These words are very strong clues, so they count double.
STRONG_TERMS = {
    "refund", "payment", "deduct", "deliver", "parcel", "courier", "damage",
    "password", "otp$", "hack", "crash", "customer care", "support", "locked",
    "upi", "credit card", "transaction", "login",
}


def _compile_keyword(keyword: str) -> re.Pattern:
    """Turn a keyword like 'deliver' or 'app$' into a regular expression."""
    if keyword.endswith("$"):
        return re.compile(rf"\b{re.escape(keyword[:-1])}\b")
    return re.compile(rf"\b{re.escape(keyword)}")


_CATEGORY_PATTERNS = {
    category: [(_compile_keyword(k), 2 if k in STRONG_TERMS else 1) for k in keywords]
    for category, keywords in CATEGORY_KEYWORDS.items()
}

# Every single-word keyword (without "$") doubles as a "domain word" that we
# prefer when extracting keywords.
DOMAIN_PREFIXES = sorted(
    {
        k.rstrip("$")
        for keywords in CATEGORY_KEYWORDS.values()
        for k in keywords
        if " " not in k
    }
    | {"fail", "stolen", "scam", "fraud", "late", "lost", "twice", "urgent", "wrong",
       "order", "receiv", "caught"}
)


def rule_based_scores(text: str) -> dict[str, int]:
    """Count keyword hits for every category."""
    # "bank account" is about payments, not about the user's login account.
    lowered = clean_text(text).replace("bank account", "bankaccount")
    scores = {}
    for category, patterns in _CATEGORY_PATTERNS.items():
        scores[category] = sum(w for pattern, w in patterns if pattern.search(lowered))
    return scores


def load_training_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the labelled sample dataset used to train the category model."""
    if not Path(path).exists():
        raise DatasetMissingError(f"Dataset not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise DatasetMissingError(f"Dataset could not be read: {exc}") from exc
    required = {"Complaint", "Category"}
    if not required.issubset(df.columns):
        raise DatasetMissingError("Dataset must contain 'Complaint' and 'Category' columns.")
    df = df.dropna(subset=["Complaint", "Category"])
    if df.empty:
        raise DatasetMissingError("Dataset is empty.")
    return df


def build_pipeline() -> Pipeline:
    """TF-IDF turns text into numbers, Logistic Regression learns the categories."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_text,
                    tokenizer=str.split,
                    token_pattern=None,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(C=10, max_iter=2000, class_weight="balanced"),
            ),
        ]
    )


def train_category_model(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> dict:
    """Train the category classifier, save it to models/, return its metrics."""
    df = load_training_data(data_path)
    texts, labels = df["Complaint"].astype(str), df["Category"].astype(str)

    pipeline = build_pipeline()

    # Cross-validation gives an honest estimate of accuracy on unseen text.
    cv_accuracy = None
    smallest_class = labels.value_counts().min()
    if smallest_class >= 3:
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_accuracy = float(cross_val_score(pipeline, texts, labels, cv=folds).mean())

    pipeline.fit(texts, labels)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline}, model_path)

    metrics = {
        "algorithm": "TF-IDF + Logistic Regression",
        "training_samples": int(len(df)),
        "cv_accuracy": cv_accuracy,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


# --------------------------------------------------------------------------
# 4. SENTIMENT ANALYSIS  (lexicon based)
# --------------------------------------------------------------------------
POSITIVE_WORDS = {
    "thank", "thanks", "thankful", "good", "great", "excellent", "happy",
    "satisfied", "helpful", "polite", "quick", "quickly", "fast", "resolved",
    "love", "loved", "perfect", "smooth", "smoothly", "easy", "awesome",
    "amazing", "appreciate", "appreciated", "nice", "wonderful", "pleased",
    "best", "friendly", "successful", "success", "impressed", "delighted",
    "fantastic", "superb", "brilliant", "glad", "convenient",
}

NEGATIVE_WORDS = {
    "failed": 1, "fail": 1, "fails": 1, "failure": 1, "deducted": 1,
    "damaged": 1, "damage": 1, "broken": 1, "defective": 1, "faulty": 1,
    "cracked": 1, "crash": 1, "crashing": 1, "crashes": 1, "locked": 1,
    "hacked": 1, "stolen": 1, "rude": 1, "delay": 1, "delayed": 1,
    "delays": 1, "stuck": 1, "wrong": 1, "missing": 1, "refused": 1,
    "refuse": 1, "error": 1, "errors": 1, "slow": 1, "poor": 1, "expired": 1,
    "suspicious": 1, "unauthorized": 1, "unauthorised": 1, "extra": 1,
    "twice": 1, "pending": 1, "cancelled": 1, "canceled": 1, "unhelpful": 1,
    "unresolved": 1, "disappointed": 1, "disappointing": 1, "angry": 1,
    "frustrated": 1, "frustrating": 1, "worst": 1, "terrible": 1,
    "horrible": 1, "bad": 1, "awful": 1, "useless": 1, "pathetic": 1,
    "annoying": 1, "ridiculous": 1, "unacceptable": 1, "fraud": 1,
    "scam": 1, "complaint": 1, "problem": 1, "problems": 1, "issue": 1,
    "issues": 1, "trouble": 1, "never": 1, "nobody": 1, "waiting": 1,
    "wait": 1, "hangs": 1, "hang": 1, "freezing": 1, "down": 1, "lost": 1,
    "late": 1, "unknown": 1, "unpaid": 1, "overcharged": 1,
    "overheating": 1, "swelling": 1, "burning": 1, "sparks": 1, "wet": 1,
    "incomplete": 1, "nothing": 1, "disconnecting": 1, "timeout": 1, "suspended": 1,
    "blocked": 1, "denied": 1, "rejected": 1, "ignored": 1, "fake": 1,
    "torn": 1, "leaking": 1, "missed": 1, "stopped": 1, "different": 1,
    "less": 1, "unable": 1, "cannot": 1, "without": 1, "disconnected": 1,
}

NEGATORS = {"not", "no", "never", "cannot", "without", "nobody", "nothing", "unable"}
CONTRAST_WORDS = {"but", "however"}  # the part AFTER "but" is what the customer cares about


def lexicon_sentiment(text: str) -> tuple[str, float]:
    """Score the text with word lists.

    * Positive word            -> +1
    * Negative word            -> -1
    * Positive word after "not" is flipped ("not helpful" is negative)
    * Words after "but" count double ("polite but unresolved")
    Returns (label, score between -1 and +1).
    """
    tokens = tokenize(clean_text(text))
    positive = negative = 0.0
    weight = 1.0

    for i, token in enumerate(tokens):
        if token in CONTRAST_WORDS:
            weight = 2.0
            continue

        recent_words = tokens[max(0, i - 3): i]
        negated = any(w in NEGATORS or w.endswith("n't") for w in recent_words)

        if token in POSITIVE_WORDS:
            if negated:
                negative += weight
            else:
                positive += weight
        elif token in NEGATIVE_WORDS:
            negative += weight
        elif token.endswith("n't") or token == "not" or token == "no":
            negative += 0.5 * weight  # a bare "not"/"no" is a mild negative signal

    total = positive + negative
    score = 0.0 if total == 0 else (positive - negative) / total
    if score >= 0.2:
        return "Positive", round(score, 2)
    if score <= -0.2:
        return "Negative", round(score, 2)
    return "Neutral", round(score, 2)


# --------------------------------------------------------------------------
# 5. PRIORITY & URGENCY  (explainable scoring)
# --------------------------------------------------------------------------
def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# Each "signal" is a clue hidden in the complaint text.
SIGNAL_PATTERNS = {
    "financial_loss": _rx(
        r"\b(?:deduct\w*|debited|charged|overcharg\w*|twice|stolen|fraud\w*"
        r"|rupees|cash|dollars?)\b|\brs\.?\s?\d|[₹$]\s?\d"
    ),
    "safety_risk": _rx(
        r"\b(?:fire|burn\w*|spark\w*|overheat\w*|swell\w*|swollen|shock|electrocut\w*"
        r"|injur\w*|expired|poison\w*|unsafe|danger\w*|smoke|smoking|explod\w*"
        r"|explosion|allerg\w*|hospital|health|sick|bleed\w*)\b"
    ),
    "security_threat": _rx(
        r"\b(?:hack\w*|stolen|fraud\w*|unauthori[sz]ed|breach\w*|compromised|phishing"
        r"|scam\w*|identity\s+theft)\b"
        r"|without\s+my\s+(?:permission|knowledge|consent)|unknown\s+transaction"
    ),
    "suspicious_activity": _rx(r"\bsuspicious\b|\bunusual\s+(?:login|activity)\b|another\s+country"),
    "item_lost": _rx(
        r"\bmarked\s+(?:as\s+)?delivered\b|\bwrong\s+(?:bank\s+)?(?:address|account)\b"
        r"|\blost\b|\bnever\s+(?:received|arrived|delivered)\b"
    ),
    "urgent_language": _rx(
        r"\b(?:urgent\w*|immediate\w*|asap|emergency|right\s+now|at\s+once"
        r"|as\s+soon\s+as\s+possible|critical)\b"
    ),
    "time_sensitive": _rx(
        r"\b(?:tomorrow|tonight|today|deadline|exam|interview|flight|wedding"
        r"|meeting|this\s+evening)\b"
    ),
    "repeated_problem": _rx(
        r"\b(?:again|repeated\w*|multiple\s+times|many\s+times|several\s+times"
        r"|every\s+time|each\s+time|keeps|second\s+time|third\s+time|yet\s+again"
        r"|always)\b"
        r"|\b(?:two|three|four|five|six|seven|eight|nine|ten|\d+)\s+times\b"
    ),
    "service_disruption": _rx(
        r"\b(?:down|outage|crash\w*|hangs?|freez\w*|locked|suspended|blocked)\b"
        r"|\berror\s+\d{3}\b|\bblank\s+(?:white\s+)?(?:page|screen)\b"
        r"|(?:\bnot|\bstopped|n't)\s+(?:working|loading)"
        r"|(?:cannot|can't|unable\s+to)\s+(?:log\s?in|login|access|use|pay|open|connect|sign\s+in)"
    ),
    "damage_or_defect": _rx(
        r"\b(?:damag\w*|crack\w*|broken|defect\w*|faulty|torn|dent\w*|leak\w*"
        r"|tampered|used\s+product)\b"
    ),
    "business_impact": _rx(r"\b(?:business|clients?|salary|production|revenue)\b"),
    "refund_pending": _rx(r"\brefund\w*|\bmoney\s+back\b"),
    "long_duration": _rx(
        r"\b(?:weeks?|months?)\b|\b(?:[5-9]|\d{2,})\s+days\b"
        r"|\b(?:five|six|seven|eight|nine|ten|fifteen|twenty)\s+days\b"
    ),
    "unmet_expectation": _rx(
        r"\b(?:not|never|no|didn't|hasn't|haven't|wasn't|weren't|isn't|won't)\s+(?:\w+\s+){0,3}?"
        r"(?:receiv\w*|arriv\w*|deliver\w*|reach\w*|credit\w*|refund\w*|repl\w*|respon\w*"
        r"|resolv\w*|solv\w*|reflect\w*|ship\w*|fix\w*|visit\w*|inform\w*|updat\w*)"
        r"|\b(?:missing|delay\w*|wrong|rude|ignor\w*|waiting|refus\w*|stuck|pending"
        r"|unresolved|incomplete|nobody)\b"
        r"|\bno\s+(?:reply|response|updates?|one)\b|\bfail\w*|\bwet\b|\bleft\s+outside\b"
        r"|\b(?:poor|terrible|awful|disappointed|disappointing|unacceptable)\b"
        r"|\bextra\s+(?:money|charges?|fees?)\b"
        r"|\bwithout\s+(?:any\s+)?(?:information|notice|warning|informing)\b"
    ),
}

SIGNAL_LABELS = {
    "financial_loss": "Financial loss mentioned",
    "safety_risk": "Safety-related words",
    "security_threat": "Security threat (hacking / fraud)",
    "urgent_language": "Urgent language",
    "time_sensitive": "Time-sensitive situation",
    "repeated_problem": "Problem has happened repeatedly",
    "service_disruption": "Service disruption",
    "damage_or_defect": "Damaged or defective item",
    "business_impact": "Business impact",
    "suspicious_activity": "Suspicious account activity",
    "item_lost": "Item lost or sent to the wrong place",
    "refund_pending": "Refund involved",
    "long_duration": "Problem has lasted a long time",
    "unmet_expectation": "Promise or expectation not met",
}

# How many points each signal adds. Thresholds below turn points into a level.
PRIORITY_WEIGHTS = {
    "financial_loss": 3, "safety_risk": 6, "security_threat": 6,
    "urgent_language": 2, "repeated_problem": 2, "service_disruption": 2,
    "damage_or_defect": 3, "unmet_expectation": 1, "long_duration": 1,
    "refund_pending": 1, "business_impact": 4, "suspicious_activity": 4, "item_lost": 2,
}
URGENCY_WEIGHTS = {
    "urgent_language": 4, "time_sensitive": 3, "safety_risk": 7,
    "security_threat": 7, "financial_loss": 4, "service_disruption": 3,
    "business_impact": 4, "unmet_expectation": 2, "repeated_problem": 1,
    "long_duration": 1, "suspicious_activity": 4, "item_lost": 2,
}
SENSITIVE_CATEGORIES = {"Payment Issue", "Account Issue"}  # money and security are risky


def detect_signals(text: str) -> dict[str, list[str]]:
    """Find every clue (signal) in the text and the words that triggered it."""
    found = {}
    for name, pattern in SIGNAL_PATTERNS.items():
        matches = [m.group(0).strip().lower() for m in pattern.finditer(text)]
        if matches:
            found[name] = list(dict.fromkeys(matches))  # unique, keep order
    return found


def score_to_level(score: float, medium: int, high: int, critical: int) -> str:
    """Convert a numeric score into Low / Medium / High / Critical."""
    if score >= critical:
        return "Critical"
    if score >= high:
        return "High"
    if score >= medium:
        return "Medium"
    return "Low"


def calculate_priority(sentiment: str, category: str, signals: dict) -> tuple[str, int, list[str]]:
    """Priority = sentiment points + signal points + category bonus."""
    score = 0
    reasons = []

    if sentiment == "Negative":
        score += 1
        reasons.append("Negative sentiment (+1)")
    elif sentiment == "Positive":
        score -= 2
        reasons.append("Positive sentiment (-2)")

    for name, weight in PRIORITY_WEIGHTS.items():
        if name in signals:
            score += weight
            reasons.append(f"{SIGNAL_LABELS[name]}: {', '.join(signals[name][:3])} (+{weight})")

    if category in SENSITIVE_CATEGORIES:
        score += 1
        reasons.append(f"Sensitive category: {category} (+1)")

    score = max(score, 0)
    return score_to_level(score, medium=2, high=4, critical=7), score, reasons


def calculate_urgency(signals: dict) -> tuple[str, int, list[str]]:
    """Urgency only looks at time pressure and risk, not at sentiment."""
    score = 0
    reasons = []
    for name, weight in URGENCY_WEIGHTS.items():
        if name in signals:
            score += weight
            reasons.append(f"{SIGNAL_LABELS[name]}: {', '.join(signals[name][:3])} (+{weight})")
    return score_to_level(score, medium=2, high=4, critical=7), score, reasons


# --------------------------------------------------------------------------
# 6. KEYWORD EXTRACTION
# --------------------------------------------------------------------------
KEY_PHRASES = [
    "credit card", "debit card", "bank account", "net banking", "customer care",
    "customer support", "customer service", "service center", "service centre",
    "order id", "tracking number", "tracking link", "password reset",
    "reset link", "coupon code", "promo code", "refund amount", "video call",
    "two-factor authentication", "phone number", "email address",
    "delivery date", "delivery address", "payment gateway", "transaction id",
    "wrong address", "money back", "checkout page",
]


def extract_keywords(text: str, max_keywords: int = 6) -> list[str]:
    """Pick the most important words and phrases from the complaint.

    1. Find known phrases like "credit card" or "bank account".
    2. Score the remaining words: domain words (payment, refund...) beat others.
    3. Return the best ones in the order they appear in the text.
    """
    lowered = clean_text(text)
    candidates = []  # (position, keyword, score)

    # Step 1 - phrases
    remaining = lowered
    for phrase in KEY_PHRASES:
        position = lowered.find(phrase)
        if position != -1:
            candidates.append((position, phrase, 4))
            remaining = remaining.replace(phrase, " " * len(phrase))

    # Step 2 - single words
    seen = set()
    for match in re.finditer(r"[a-z]+(?:'[a-z]+)?", remaining):
        word = match.group(0)
        if (word in STOPWORDS or word in KEYWORD_EXCLUDE or word.endswith("n't")
                or len(word) < 3 or word in seen):
            continue
        seen.add(word)
        is_domain_word = any(word.startswith(prefix) for prefix in DOMAIN_PREFIXES)
        candidates.append((match.start(), word, 3 if is_domain_word else 1))

    # Step 3 - keep the best scores, then restore reading order
    # Domain words and phrases come first; plain words only fill the gaps
    # when we found fewer than 4 strong keywords.
    strong = [c for c in candidates if c[2] >= 3]
    weak = [c for c in candidates if c[2] < 3]
    best = sorted(strong, key=lambda c: (-c[2], c[0]))[:max_keywords]
    if len(best) < 4:
        best += sorted(weak, key=lambda c: c[0])[: 4 - len(best)]
    return [keyword for _, keyword, _ in sorted(best, key=lambda c: c[0])]


# --------------------------------------------------------------------------
# 7. SUMMARY  (extractive: choose the most informative sentence(s))
# --------------------------------------------------------------------------
_VERB_AGREEMENT = {
    "am": "is", "have": "has", "haven't": "hasn't", "don't": "doesn't",
    "do not": "does not", "want": "wants", "need": "needs", "get": "gets",
    "can't": "cannot", "keep": "keeps", "love": "loves", "like": "likes",
    "feel": "feels", "think": "thinks", "was": "was",
}


def to_third_person(sentence: str) -> str:
    """'I have not received my refund' -> 'The customer has not received their refund'."""
    s = sentence
    for short, full in (("I'm", "I am"), ("I've", "I have"), ("I'll", "I will"), ("I'd", "I would")):
        s = re.sub(rf"\b{re.escape(short)}", full, s, flags=re.IGNORECASE)

    s = re.sub(r"\bdo\s+I\b", "does the customer", s, flags=re.IGNORECASE)
    s = re.sub(r"\bam\s+I\b", "is the customer", s, flags=re.IGNORECASE)
    verbs = "|".join(re.escape(v) for v in sorted(_VERB_AGREEMENT, key=len, reverse=True))
    s = re.sub(
        rf"\b[Ii]\s+((?:still|also|just|already|really)\s+)?({verbs})\b",
        lambda m: f"the customer {m.group(1) or ''}{_VERB_AGREEMENT[m.group(2).lower()]}",
        s,
    )
    s = re.sub(r"\b[Ii]\b", "the customer", s)
    s = re.sub(r"\bmy\b", "their", s, flags=re.IGNORECASE)
    s = re.sub(r"\bme\b", "them", s, flags=re.IGNORECASE)
    s = re.sub(r"\bmine\b", "theirs", s, flags=re.IGNORECASE)
    return s[:1].upper() + s[1:] if s else s


def _shorten(text: str, limit: int = 220) -> str:
    """Cut long text at a word boundary."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "..."


def summarize(text: str, max_sentences: int = 2) -> str:
    """Pick the most informative sentences and rewrite them in third person."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if s.strip()]
    if not sentences:
        return ""

    def sentence_score(sentence: str) -> int:
        words = tokenize(clean_text(sentence))
        domain_words = sum(any(w.startswith(p) for p in DOMAIN_PREFIXES) for w in words)
        has_problem = any(w in NEGATIVE_WORDS or w in NEGATORS or w.endswith("n't") for w in words)
        return domain_words + (2 if has_problem else 0)

    if len(sentences) > max_sentences:
        ranked = sorted(range(len(sentences)), key=lambda i: (-sentence_score(sentences[i]), i))
        chosen = sorted(ranked[:max_sentences])
        sentences = [sentences[i] for i in chosen]

    summary = " ".join(to_third_person(s) for s in sentences)
    summary = _shorten(summary)
    if summary and summary[-1] not in ".!?":
        summary += "."
    return summary


# --------------------------------------------------------------------------
# 8. RECOMMENDED ACTION
# --------------------------------------------------------------------------
ACTIONS = {
    "Payment Issue": {
        "Low": "Send the customer a payment-support guide and confirm the transaction details.",
        "Medium": "Review the payment gateway logs and update the customer within 24 hours.",
        "High": "Verify the transaction and initiate refund investigation.",
        "Critical": "Freeze the disputed transaction, alert the fraud team and contact the customer immediately.",
    },
    "Delivery Issue": {
        "Low": "Share the latest tracking update with the customer.",
        "Medium": "Check shipment status and contact the delivery team.",
        "High": "Escalate to the logistics manager and arrange a re-delivery or replacement.",
        "Critical": "Escalate to the logistics head immediately and arrange an urgent replacement or refund.",
    },
    "Product Issue": {
        "Low": "Acknowledge the feedback and share the return or exchange policy.",
        "Medium": "Request photos of the product and offer a replacement or exchange.",
        "High": "Arrange a free pickup and send a replacement or refund after verification.",
        "Critical": "Contact the customer immediately, advise them to stop using the product and escalate to the quality and safety team.",
    },
    "Account Issue": {
        "Low": "Guide the customer through the account settings or help-centre steps.",
        "Medium": "Help the customer recover the account after identity verification.",
        "High": "Verify the customer's identity and restore access or reset credentials securely.",
        "Critical": "Immediately secure the account and escalate to the security team.",
    },
    "Technical Issue": {
        "Low": "Log the issue and share a basic troubleshooting checklist (update, restart, clear cache).",
        "Medium": "Ask for device and app version details and assign the issue to technical support.",
        "High": "Create a high-priority bug ticket and notify the engineering team.",
        "Critical": "Start the incident-response process and alert the engineering on-call team immediately.",
    },
    "Service Issue": {
        "Low": "Acknowledge the feedback and share the expected response time.",
        "Medium": "Assign a senior support agent to follow up with the customer.",
        "High": "Escalate to the support manager and call the customer back within a few hours.",
        "Critical": "Escalate to senior management immediately and arrange a personal call-back.",
    },
    "Refund Issue": {
        "Low": "Share the refund policy and the expected processing timeline.",
        "Medium": "Check the refund status and tell the customer the expected credit date.",
        "High": "Expedite the refund with the finance team and send the customer a reference number.",
        "Critical": "Escalate to the finance head, process the refund on priority and confirm with the customer.",
    },
    "Other": {
        "Low": "Thank the customer and route the message to the relevant team.",
        "Medium": "Review the message and route it to the right department.",
        "High": "Assign the case to a support executive for manual review.",
        "Critical": "Escalate to a senior manager for immediate manual review.",
    },
}


def get_recommended_action(category: str, priority: str) -> str:
    """Look up the action for a (category, priority) pair."""
    return ACTIONS.get(category, ACTIONS["Other"]).get(priority, ACTIONS["Other"]["Medium"])


# --------------------------------------------------------------------------
# 9. THE MAIN CLASS
# --------------------------------------------------------------------------
class ComplaintAnalyzer:
    """Loads the models once and analyses complaints one by one."""

    def __init__(self, use_transformers: bool | None = None):
        if use_transformers is None:
            use_transformers = os.getenv("USE_TRANSFORMERS", "false").strip().lower() in {
                "1", "true", "yes",
            }
        self.use_transformers = use_transformers
        self.model = None
        self.model_status = "Not loaded"
        self._sentiment_pipeline = None
        self._transformer_failed = False
        self._load_category_model()

    # ----- model loading ---------------------------------------------------
    def _load_category_model(self) -> None:
        """Load the saved model. If it is missing, train one. If that fails
        too, continue with keyword rules only (the app still works)."""
        try:
            if MODEL_PATH.exists():
                self.model = joblib.load(MODEL_PATH)["pipeline"]
                self.model_status = "ML model loaded"
                return
        except Exception as exc:  # corrupted file or version mismatch
            logger.warning("Could not load saved model (%s). Re-training.", exc)

        try:
            train_category_model()
            self.model = joblib.load(MODEL_PATH)["pipeline"]
            self.model_status = "ML model trained automatically"
        except Exception as exc:
            logger.warning("Could not train model (%s). Using keyword rules only.", exc)
            self.model = None
            self.model_status = "Keyword rules only (ML model unavailable)"

    # ----- category --------------------------------------------------------
    def predict_category(self, text: str) -> tuple[str, float]:
        """Blend the ML model (60%) with keyword rules (40%)."""
        rule_scores = rule_based_scores(text)
        rule_total = sum(rule_scores.values())
        rule_share = {c: (s / rule_total if rule_total else 0.0) for c, s in rule_scores.items()}

        ml_share = None
        if self.model is not None:
            try:
                probabilities = self.model.predict_proba([text])[0]
                ml_share = dict(zip(self.model.classes_, probabilities))
            except Exception as exc:
                logger.warning("ML prediction failed: %s", exc)

        if ml_share is not None and rule_total:
            combined = {c: 0.6 * ml_share.get(c, 0.0) + 0.4 * rule_share.get(c, 0.0) for c in CATEGORIES}
        elif ml_share is not None:
            combined = {c: ml_share.get(c, 0.0) for c in CATEGORIES}
        elif rule_total:
            combined = {c: rule_share.get(c, 0.0) for c in CATEGORIES}
        else:
            return "Other", 1.0

        best = max(combined, key=combined.get)
        confidence = combined[best]
        if confidence < 0.30:
            return "Other", round(1 - confidence, 2)
        return best, round(confidence, 2)

    # ----- sentiment -------------------------------------------------------
    def _transformer_sentiment(self, text: str):
        """Optional: DistilBERT sentiment. Returns None if anything goes wrong."""
        if self._transformer_failed:
            return None
        try:
            if self._sentiment_pipeline is None:
                from transformers import pipeline  # imported late: it is heavy

                self._sentiment_pipeline = pipeline("sentiment-analysis", model=TRANSFORMER_MODEL)
            result = self._sentiment_pipeline(text[:512])[0]
        except Exception as exc:
            logger.warning("Transformers unavailable (%s). Using lexicon sentiment.", exc)
            self._transformer_failed = True
            return None

        if result["score"] < 0.80:  # the model is unsure -> call it Neutral
            return "Neutral", 0.0
        if result["label"].upper().startswith("POS"):
            return "Positive", round(result["score"], 2)
        return "Negative", round(-result["score"], 2)

    def predict_sentiment(self, text: str) -> tuple[str, float]:
        if self.use_transformers:
            result = self._transformer_sentiment(text)
            if result is not None:
                return result
        return lexicon_sentiment(text)

    # ----- everything together --------------------------------------------
    def analyze(self, text: str) -> dict:
        """Run the full pipeline and return one dictionary of results."""
        text = validate_complaint(text)

        category, confidence = self.predict_category(text)
        sentiment, sentiment_score = self.predict_sentiment(text)
        signals = detect_signals(text)

        # A real complaint is never "neutral" if it describes a clear problem.
        problem_signals = {"financial_loss", "safety_risk", "security_threat",
                           "damage_or_defect", "service_disruption", "unmet_expectation",
                           "repeated_problem", "suspicious_activity", "item_lost"}
        if sentiment == "Neutral" and problem_signals & set(signals):
            sentiment = "Negative"

        # Praise or thanks is not a problem, so it goes to "Other".
        if sentiment == "Positive" and not signals:
            category, confidence = "Other", 0.80

        priority, priority_score, priority_reasons = calculate_priority(sentiment, category, signals)
        urgency, urgency_score, urgency_reasons = calculate_urgency(signals)

        return {
            "complaint_text": text,
            "category": category,
            "category_confidence": confidence,
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "priority": priority,
            "priority_score": priority_score,
            "priority_reasons": priority_reasons,
            "urgency": urgency,
            "urgency_score": urgency_score,
            "urgency_reasons": urgency_reasons,
            "keywords": extract_keywords(text),
            "summary": summarize(text),
            "recommended_action": get_recommended_action(category, priority),
            "model_status": self.model_status,
        }

    def get_model_info(self) -> dict:
        """Small dictionary for the About page."""
        info = {"status": self.model_status, "transformers": self.use_transformers}
        try:
            info.update(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
        return info


# Run `python complaint_analyzer.py` to retrain the model and try a demo.
if __name__ == "__main__":
    import complaint_analyzer as module  # import by name so the saved model loads everywhere

    print("Training model...")
    print(json.dumps(module.train_category_model(), indent=2))
    demo = module.ComplaintAnalyzer().analyze(
        "My credit card payment failed but the amount was deducted from my bank account."
    )
    for key, value in demo.items():
        print(f"{key:22}: {value}")

"""Backend configuration and threshold parameters."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_ARTIFACT_PATH = os.path.join(BASE_DIR, "ml", "models", "classifier_v1.npz")
METADATA_PATH = os.path.join(BASE_DIR, "ml", "models", "metadata.json")

# Probability thresholds for verdicts
SAFE_THRESHOLD_UPPER = 0.30
SUSPICIOUS_THRESHOLD_UPPER = 0.70

# Rate limiting
RATE_LIMIT_PER_MINUTE = 120

# Cache configuration
CACHE_MAX_ENTRIES = 10000
CACHE_TTL_SECONDS = 3600

# Top domains default allowlist
TOP_DOMAINS_SEED = {
    "google.com", "youtube.com", "facebook.com", "baidu.com", "wikipedia.org",
    "yahoo.com", "reddit.com", "amazon.com", "twitter.com", "x.com",
    "instagram.com", "linkedin.com", "netflix.com", "microsoft.com", "apple.com",
    "github.com", "gitlab.com", "stackoverflow.com", "cloudflare.com", "whatsapp.com",
    "gmail.com", "docs.google.com", "drive.google.com", "bing.com", "office.com",
    "gov.np", "nepal.gov.np", "onlinekhabar.com", "ekantipur.com", "mof.gov.np"
}

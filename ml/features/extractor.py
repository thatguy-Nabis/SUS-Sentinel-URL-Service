"""Lexical feature extraction engine implemented using NumPy and standard library."""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Union
import numpy as np

from ml.features.normalizer import extract_domain_info, normalize_url


# Suspicious authentication and deception keywords
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "update", "security", "account", "banking",
    "secure", "webscr", "signin", "confirm", "ebayisapi", "password",
    "authenticate", "wallet", "admin", "service", "support", "billing",
    "recover", "validate"
]

# High-abuse TLDs commonly seen in phishing and spam campaigns
HIGH_RISK_TLDS = {
    "xyz", "top", "club", "work", "ru", "cn", "tk", "ml", "ga",
    "cf", "gq", "buzz", "fit", "rest", "surf", "monster", "icu", "cam"
}

# Known URL shortener domains
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "buff.ly",
    "ow.ly", "rebrand.ly", "cutt.ly", "tiny.cc", "bit.do", "shorte.st"
}

# Major target brands for lookalike detection
TARGET_BRANDS = [
    "paypal", "google", "apple", "microsoft", "amazon", "netflix",
    "chase", "facebook", "instagram", "whatsapp", "wellsfargo", "binance"
]


def calculate_entropy(text: str) -> float:
    """Compute Shannon entropy of a string."""
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy


class LexicalFeatureExtractor:
    """Extracts deterministic lexical features from URL strings using pure Python and NumPy."""

    def __init__(self):
        self.feature_names: List[str] = [
            "url_length",
            "host_length",
            "path_length",
            "has_https",
            "is_ip",
            "is_shortener",
            "count_dots",
            "count_hyphens",
            "count_underscores",
            "count_slashes",
            "count_question_marks",
            "count_equal_signs",
            "count_at_symbols",
            "count_percent_signs",
            "count_ampersands",
            "count_digits",
            "digit_ratio_url",
            "digit_ratio_host",
            "subdomain_count",
            "url_entropy",
            "host_entropy",
            "suspicious_keyword_count",
            "suspicious_keyword_in_host",
            "high_risk_tld",
            "brand_in_subdomain_or_path",
            "has_double_slash_in_path",
            "longest_token_length",
            "vowel_consonant_ratio",
            "special_char_ratio",
            "has_port_specified",
        ]

    def extract_dict(self, url: str) -> Dict[str, float]:
        """Extract features as a dictionary."""
        info = extract_domain_info(url)
        normalized = normalize_url(url, strip_query_and_fragment=False)
        
        url_lower = normalized.lower()
        host = info["host"].lower()
        path = info["path"]
        reg_domain = info["registered_domain"].lower()
        subdomain = info["subdomain"].lower()
        tld = info["tld"].lower()

        url_len = len(url_lower)
        host_len = len(host)
        path_len = len(path)

        # Digit counts
        digits_url = sum(c.isdigit() for c in url_lower)
        digits_host = sum(c.isdigit() for c in host)
        digit_ratio_url = digits_url / max(1, url_len)
        digit_ratio_host = digits_host / max(1, host_len)

        # Keyword checks
        kw_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)
        kw_in_host = 1.0 if any(kw in host for kw in SUSPICIOUS_KEYWORDS) else 0.0

        # High risk TLD
        is_risky_tld = 1.0 if tld in HIGH_RISK_TLDS else 0.0

        # Shortener check
        is_shortener = 1.0 if reg_domain in SHORTENER_DOMAINS or host in SHORTENER_DOMAINS else 0.0

        # Brand lookalike check (e.g. paypal in subdomain/path but registered domain != paypal.com)
        brand_spoof = 0.0
        for brand in TARGET_BRANDS:
            if (brand in subdomain or brand in path) and (brand not in reg_domain):
                brand_spoof = 1.0
                break

        # Double slash in path (redirect tricks)
        has_double_slash = 1.0 if "//" in path else 0.0

        # Longest token in URL
        tokens = re.split(r"[\W_]+", url_lower)
        longest_token = max((len(t) for t in tokens if t), default=0)

        # Vowels and consonants
        letters = [c for c in url_lower if c.isalpha()]
        vowels = sum(1 for c in letters if c in "aeiou")
        consonants = len(letters) - vowels
        vowel_ratio = vowels / max(1, consonants)

        # Special characters
        special_count = sum(not c.isalnum() for c in url_lower)
        special_ratio = special_count / max(1, url_len)

        # Subdomains count
        subdomain_count = float(len(subdomain.split(".")) if subdomain else 0)

        # Explicit port check
        has_port = 1.0 if ":" in info["netloc"] else 0.0

        return {
            "url_length": float(url_len),
            "host_length": float(host_len),
            "path_length": float(path_len),
            "has_https": 1.0 if info["scheme"] == "https" else 0.0,
            "is_ip": float(info["is_ip"]),
            "is_shortener": is_shortener,
            "count_dots": float(url_lower.count(".")),
            "count_hyphens": float(url_lower.count("-")),
            "count_underscores": float(url_lower.count("_")),
            "count_slashes": float(url_lower.count("/")),
            "count_question_marks": float(url_lower.count("?")),
            "count_equal_signs": float(url_lower.count("=")),
            "count_at_symbols": float(url_lower.count("@")),
            "count_percent_signs": float(url_lower.count("%")),
            "count_ampersands": float(url_lower.count("&")),
            "count_digits": float(digits_url),
            "digit_ratio_url": float(digit_ratio_url),
            "digit_ratio_host": float(digit_ratio_host),
            "subdomain_count": subdomain_count,
            "url_entropy": float(calculate_entropy(url_lower)),
            "host_entropy": float(calculate_entropy(host)),
            "suspicious_keyword_count": float(kw_count),
            "suspicious_keyword_in_host": kw_in_host,
            "high_risk_tld": is_risky_tld,
            "brand_in_subdomain_or_path": brand_spoof,
            "has_double_slash_in_path": has_double_slash,
            "longest_token_length": float(longest_token),
            "vowel_consonant_ratio": float(vowel_ratio),
            "special_char_ratio": float(special_ratio),
            "has_port_specified": has_port,
        }

    def extract(self, url: str) -> np.ndarray:
        """Extract features for a single URL as a 1D NumPy array."""
        feat_dict = self.extract_dict(url)
        return np.array([feat_dict[k] for k in self.feature_names], dtype=np.float64)

    def extract_batch(self, urls: Union[List[str], np.ndarray]) -> np.ndarray:
        """Extract features for a list of URLs as a 2D NumPy array of shape (N, D)."""
        rows = [self.extract(u) for u in urls]
        if not rows:
            return np.empty((0, len(self.feature_names)), dtype=np.float64)
        return np.vstack(rows)

    def explain(self, url: str, weights: Optional[np.ndarray] = None, top_k: int = 3) -> List[str]:
        """Provide human-interpretable reasons why a URL was flagged."""
        feat_dict = self.extract_dict(url)
        reasons = []

        if feat_dict["is_ip"] > 0.5:
            reasons.append("Hostname is a raw IP address instead of a domain name")
        if feat_dict["brand_in_subdomain_or_path"] > 0.5:
            reasons.append("Well-known brand name used in subdomain/path to impersonate legitimate site")
        if feat_dict["high_risk_tld"] > 0.5:
            reasons.append("Uses a high-risk top-level domain frequently associated with abuse")
        if feat_dict["suspicious_keyword_in_host"] > 0.5:
            reasons.append("Domain name contains sensitive authentication keywords (e.g., login, verify)")
        if feat_dict["is_shortener"] > 0.5:
            reasons.append("URL shortener used to conceal destination target")
        if feat_dict["has_https"] < 0.5:
            reasons.append("Unencrypted connection (HTTP instead of HTTPS)")
        if feat_dict["count_hyphens"] >= 3:
            reasons.append("Excessive hyphens in URL often used in phishing domains")
        if feat_dict["digit_ratio_host"] > 0.3:
            reasons.append("Unusually high ratio of numbers in domain name")
        if feat_dict["subdomain_count"] >= 3:
            reasons.append("High number of subdomains structure")
        if feat_dict["url_entropy"] > 4.5:
            reasons.append("High character randomness / entropy indicating machine-generated URL")

        if not reasons:
            reasons.append("No common lexical risk indicators detected")

        return reasons[:top_k]

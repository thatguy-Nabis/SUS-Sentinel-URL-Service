"""In-memory domain allowlist management."""

from typing import Set
from backend.app.config import TOP_DOMAINS_SEED
from ml.features.normalizer import extract_domain_info


class AllowlistManager:
    """Manages fast O(1) in-memory allowlist of reputable domains."""

    def __init__(self):
        self.domains: Set[str] = set(TOP_DOMAINS_SEED)

    def is_allowed(self, url: str) -> bool:
        """Check if URL's registered domain or host is on the allowlist."""
        info = extract_domain_info(url)
        host = info["host"].lower()
        registered = info["registered_domain"].lower()

        if host in self.domains or registered in self.domains:
            return True

        # Check subdomains of known allowed domains
        for allowed in self.domains:
            if host.endswith("." + allowed):
                return True

        return False

    def add(self, domain: str) -> None:
        """Add a domain to the allowlist."""
        self.domains.add(domain.strip().lower())

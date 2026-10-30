"""URL normalization and domain parsing utilities using Python standard library."""

import re
import urllib.parse
from typing import Dict, Optional, Tuple

# Common multi-part ccTLDs for domain splitting
COMMON_SECOND_LEVEL_TLDS = {
    "co.uk", "org.uk", "gov.uk", "ac.uk",
    "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "com.np", "org.np", "edu.np", "gov.np", "net.np",
    "co.jp", "ne.jp", "ac.jp", "go.jp",
    "com.br", "org.br", "net.br",
    "co.in", "net.in", "org.in", "gen.in",
    "com.cn", "net.cn", "org.cn", "gov.cn",
    "co.za", "org.za",
    "com.sg", "edu.sg",
}

IPV4_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
IPV6_PATTERN = re.compile(r"^\[?[0-9a-fA-F:]+\]?$")


def normalize_url(url: str, strip_query_and_fragment: bool = True) -> str:
    """Normalize a URL for canonical processing and privacy preservation.

    Args:
        url: Raw URL string.
        strip_query_and_fragment: Whether to remove query strings (?...) and fragments (#...).

    Returns:
        Canonical normalized URL string.
    """
    if not url or not isinstance(url, str):
        return ""

    url = url.strip()

    # If scheme is missing, prepend http:// for parsing
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        if url.startswith("//"):
            url = "http:" + url
        else:
            url = "http://" + url

    try:
        parsed = urllib.parse.urlsplit(url)
    except Exception:
        return url.lower()

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    # Handle credentials in netloc (e.g., user:pass@host)
    if "@" in netloc:
        auth, _, host_port = netloc.rpartition("@")
    else:
        auth = ""
        host_port = netloc

    # Handle host and port
    if ":" in host_port and not host_port.endswith("]"):
        host, _, port = host_port.partition(":")
        # Strip default ports
        if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
            host_port = host
    else:
        host = host_port

    # Handle Punycode (IDN) decoding
    try:
        host = host.encode("ascii").decode("idna")
    except Exception:
        pass

    clean_netloc = f"{auth}@{host_port}" if auth else host_port
    path = parsed.path
    if not path:
        path = "/"

    # Collapse multiple consecutive slashes in path
    path = re.sub(r"/{2,}", "/", path)

    if strip_query_and_fragment:
        return f"{scheme}://{clean_netloc}{path}"
    
    query = f"?{parsed.query}" if parsed.query else ""
    fragment = f"#{parsed.fragment}" if parsed.fragment else ""
    return f"{scheme}://{clean_netloc}{path}{query}{fragment}"


def is_ip_address(host: str) -> bool:
    """Check if the given host string is an IPv4 or IPv6 address."""
    if not host:
        return False
    host = host.strip("[]")
    if IPV4_PATTERN.match(host):
        parts = host.split(".")
        return all(0 <= int(p) <= 255 for p in parts if p.isdigit())
    if ":" in host and IPV6_PATTERN.match(host):
        return True
    return False


def extract_domain_info(url: str) -> Dict[str, str]:
    """Extract scheme, hostname, registered domain, subdomains, TLD, and path.

    Returns:
        Dictionary with keys:
            - scheme: e.g. 'https'
            - netloc: full host (and port)
            - host: hostname only
            - registered_domain: e.g. 'google.com' or 'example.co.uk'
            - subdomain: e.g. 'login.accounts'
            - tld: e.g. 'com' or 'co.uk'
            - path: path string
            - is_ip: '1' if host is IP else '0'
    """
    normalized = normalize_url(url, strip_query_and_fragment=False)
    try:
        parsed = urllib.parse.urlsplit(normalized)
    except Exception:
        return {
            "scheme": "http",
            "netloc": "",
            "host": "",
            "registered_domain": "",
            "subdomain": "",
            "tld": "",
            "path": "/",
            "is_ip": "0",
        }
    
    host_port = parsed.netloc.lower()
    host = host_port.split(":")[0] if ":" in host_port else host_port

    if is_ip_address(host):
        return {
            "scheme": parsed.scheme.lower(),
            "netloc": host_port,
            "host": host,
            "registered_domain": host,
            "subdomain": "",
            "tld": "",
            "path": parsed.path or "/",
            "is_ip": "1",
        }

    parts = host.split(".")
    tld = ""
    registered_domain = host
    subdomain = ""

    if len(parts) >= 2:
        # Check for multi-level TLD (e.g. .co.uk)
        two_level_tld = ".".join(parts[-2:])
        if two_level_tld in COMMON_SECOND_LEVEL_TLDS and len(parts) >= 3:
            tld = two_level_tld
            registered_domain = ".".join(parts[-3:])
            subdomain = ".".join(parts[:-3])
        else:
            tld = parts[-1]
            registered_domain = ".".join(parts[-2:])
            subdomain = ".".join(parts[:-2])

    return {
        "scheme": parsed.scheme.lower(),
        "netloc": host_port,
        "host": host,
        "registered_domain": registered_domain,
        "subdomain": subdomain,
        "tld": tld,
        "path": parsed.path or "/",
        "is_ip": "0",
    }

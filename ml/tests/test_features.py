"""Unit tests for URL normalization and feature extraction using standard library unittest."""

import json
import os
import unittest
import numpy as np

from ml.features.normalizer import normalize_url, extract_domain_info, is_ip_address
from ml.features.extractor import LexicalFeatureExtractor, calculate_entropy


class TestFeatures(unittest.TestCase):

    def setUp(self):
        self.extractor = LexicalFeatureExtractor()

    def test_normalize_url_basic(self):
        url = "  HTTP://Example.COM:80/path//subpath?query=1#frag  "
        norm = normalize_url(url, strip_query_and_fragment=True)
        self.assertEqual(norm, "http://example.com/path/subpath")

    def test_is_ip_address(self):
        self.assertTrue(is_ip_address("192.168.1.1"))
        self.assertTrue(is_ip_address("10.0.0.1"))
        self.assertFalse(is_ip_address("google.com"))
        self.assertFalse(is_ip_address("999.999.999.999"))

    def test_extract_domain_info(self):
        info = extract_domain_info("https://sub.domain.co.uk/test")
        self.assertEqual(info["scheme"], "https")
        self.assertEqual(info["registered_domain"], "domain.co.uk")
        self.assertEqual(info["tld"], "co.uk")
        self.assertEqual(info["subdomain"], "sub")

    def test_entropy(self):
        ent_low = calculate_entropy("aaaaaaa")
        ent_high = calculate_entropy("a1b2c3d4e5f6g7")
        self.assertEqual(ent_low, 0.0)
        self.assertGreater(ent_high, ent_low)

    def test_feature_extractor_shapes(self):
        sample = "https://www.google.com/search"
        vec = self.extractor.extract(sample)
        self.assertEqual(len(vec), len(self.extractor.feature_names))
        self.assertIsInstance(vec, np.ndarray)

        batch = self.extractor.extract_batch([sample, "http://192.168.1.1/login"])
        self.assertEqual(batch.shape, (2, len(self.extractor.feature_names)))

    def test_golden_cases_parity(self):
        golden_path = os.path.join(os.path.dirname(__file__), "..", "..", "shared", "golden_cases.json")
        if os.path.exists(golden_path):
            with open(golden_path, "r", encoding="utf-8") as f:
                cases = json.load(f)
            for c in cases:
                feats = self.extractor.extract_dict(c["url"])
                if "expected_is_ip" in c:
                    self.assertEqual(feats["is_ip"], c["expected_is_ip"])
                if "expected_has_https" in c:
                    self.assertEqual(feats["has_https"], c["expected_has_https"])
                if "expected_is_shortener" in c:
                    self.assertEqual(feats["is_shortener"], c["expected_is_shortener"])


if __name__ == "__main__":
    unittest.main()

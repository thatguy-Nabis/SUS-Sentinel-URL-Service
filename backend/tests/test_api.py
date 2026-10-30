"""Unit tests for FastAPI backend routes using FastAPI TestClient."""

import unittest
from fastapi.testclient import TestClient
from backend.app.main import app


class TestBackendAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_check(self):
        res = self.client.get("/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("model_version", data)

    def test_classify_allowlist_domain(self):
        res = self.client.post("/v1/classify", json={"url": "https://www.google.com/search?q=test"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["verdict"], "safe")
        self.assertEqual(data["source"], "allowlist")
        self.assertFalse(data["cached"])
        self.assertIn("google.com", data["url"])
        # Verify query string was stripped
        self.assertNotIn("?q=test", data["url"])

    def test_classify_phishing_url(self):
        res = self.client.post(
            "/v1/classify",
            json={"url": "http://paypal-security-update.account-verification.xyz/login"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn(data["verdict"], ["suspicious", "unsafe"])
        self.assertGreater(len(data["reasons"]), 0)

    def test_caching_behavior(self):
        url = "http://example-suspicious-domain-12345.xyz/path"
        res1 = self.client.post("/v1/classify", json={"url": url})
        self.assertEqual(res1.status_code, 200)
        self.assertFalse(res1.json()["cached"])

        res2 = self.client.post("/v1/classify", json={"url": url})
        self.assertEqual(res2.status_code, 200)
        self.assertTrue(res2.json()["cached"])

    def test_report_endpoint(self):
        res = self.client.post(
            "/v1/report",
            json={
                "url": "http://suspicious-site.com",
                "reported_verdict": "false_positive",
                "user_comment": "This is my internal company portal."
            }
        )
        self.assertEqual(res.status_code, 202)
        self.assertEqual(res.json()["status"], "quarantined")


if __name__ == "__main__":
    unittest.main()

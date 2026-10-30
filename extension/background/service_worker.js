// Sentinelkey Background Service Worker (Manifest V3)

const API_ENDPOINT = 'http://localhost:8000/v1/classify';

// Strips sensitive query parameters and fragments before transmission
function sanitizeUrl(rawUrl) {
  try {
    const url = new URL(rawUrl);
    return `${url.protocol}//${url.hostname}${url.pathname}`;
  } catch (e) {
    return rawUrl;
  }
}

// Update Chrome badge appearance based on security verdict
function updateBadge(tabId, verdict) {
  if (verdict === 'safe') {
    chrome.action.setBadgeText({ tabId, text: 'SAFE' });
    chrome.action.setBadgeBackgroundColor({ tabId, color: '#10b981' });
  } else if (verdict === 'suspicious') {
    chrome.action.setBadgeText({ tabId, text: 'SUSP' });
    chrome.action.setBadgeBackgroundColor({ tabId, color: '#f59e0b' });
  } else if (verdict === 'unsafe' || verdict === 'malicious') {
    chrome.action.setBadgeText({ tabId, text: 'RISK' });
    chrome.action.setBadgeBackgroundColor({ tabId, color: '#ef4444' });
  } else {
    chrome.action.setBadgeText({ tabId, text: '?' });
    chrome.action.setBadgeBackgroundColor({ tabId, color: '#6b7280' });
  }
}

// Intercept tab updates
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url && tab.url.startsWith('http')) {
    const sanitized = sanitizeUrl(tab.url);

    fetch(API_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: sanitized, client_version: '0.1.0' })
    })
      .then((res) => res.json())
      .then((data) => {
        updateBadge(tabId, data.verdict);
        chrome.storage.local.set({ [tabId]: data });
      })
      .catch((err) => {
        // Fail-open strategy: display unchecked badge rather than blocking navigation
        chrome.action.setBadgeText({ tabId, text: 'OFF' });
        chrome.action.setBadgeBackgroundColor({ tabId, color: '#9ca3af' });
      });
  }
});

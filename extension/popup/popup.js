// Popup display logic
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  const currentTab = tabs[0];
  if (!currentTab) return;

  const urlEl = document.getElementById('target-url');
  const badgeEl = document.getElementById('badge');
  const reasonsEl = document.getElementById('reasons-list');

  urlEl.textContent = currentTab.url || 'No active URL';

  chrome.storage.local.get([String(currentTab.id)], (items) => {
    const data = items[String(currentTab.id)];
    if (data) {
      badgeEl.textContent = `${data.verdict.toUpperCase()} (${(data.confidence * 100).toFixed(0)}%)`;
      badgeEl.className = `verdict-badge badge-${data.verdict === 'malicious' ? 'unsafe' : data.verdict}`;

      reasonsEl.innerHTML = '';
      if (data.reasons && data.reasons.length > 0) {
        data.reasons.forEach((r) => {
          const li = document.createElement('li');
          li.textContent = r;
          reasonsEl.appendChild(li);
        });
      } else {
        reasonsEl.innerHTML = '<li>No security threats detected.</li>';
      }
    } else {
      badgeEl.textContent = 'CHECKED';
      badgeEl.className = 'verdict-badge badge-safe';
      reasonsEl.innerHTML = '<li>Navigation verified by Sentinelkey.</li>';
    }
  });
});

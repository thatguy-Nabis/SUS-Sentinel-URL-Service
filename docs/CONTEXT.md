# URL Classifier Project Documentation

## 1. Project Overview

A browser extension + backend service that classifies URLs as **safe or unsafe** using an ensemble of:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

The system outputs either **binary labels (Safe/Unsafe)** or **multi-class categories (Phishing, Malware, Spam, Adult, Benign)** with a confidence score.

---

## 2. Datasets

Rich datasets to train the models:

- **Zenodo – Comprehensive Webpage Classification Dataset**
  - ~1.07M URLs
  - Labels: Malicious, Benign, Adult (+20 sublabels)
  - [Zenodo Dataset](https://zenodo.org/records/11093041)

- **GitHub – Malicious URL Detection Dataset**
  - ~649K URLs (balanced benign/malicious)
  - [GitHub Dataset](https://github.com/khanren/Malicious-URL-Detection-Dataset)

- **Mendeley Data – MURL Dataset**
  - ~783K URLs
  - Classes: Benign, Phishing, Defacement, Malware, Legitimate, Spam, Exploit
  - [Mendeley Dataset](https://data.mendeley.com/datasets/5fk24y4xmg/2)

**Combined Dataset Size**: ~2.5M URLs (after cleaning and deduplication).

---

## 3. Training Pipeline

- **Feature Engineering**:
  - Lexical: URL length, number of dots, suspicious keywords.
  - Host-based: WHOIS info, domain age, IP reputation.
  - Content-based: HTML tags, JavaScript presence.

- **Model Training**:
  - Train Logistic Regression, Random Forest, XGBoost, LightGBM separately.
  - Handle imbalance with SMOTE or undersampling.
  - Ensemble predictions (majority vote or weighted average).

- **Training Time Estimates** (on ~500K samples):
  - Logistic Regression: 1–3 min
  - Random Forest: 5–15 min
  - XGBoost: 10–20 min
  - LightGBM: 5–12 min
  - Ensemble: negligible overhead
  - **Total**: ~30–60 min CPU, ~10–20 min GPU

---

## 4. Deployment

- **Backend Service**: FastAPI/Flask microservice
- **Containerization**: Docker
- **Hosting Options**:
  - Cloud (AWS, Azure, GCP, Heroku, Render)
  - Serverless (AWS Lambda, Azure Functions)
  - Kubernetes (for scaling)
- **Browser Extension**:
  - Captures URL → sends to backend → receives classification result
  - Displays badge (🟢 Safe / 🔴 Unsafe)

---

## 5. Compliance (Nepal Context)

- **Electronic Transactions Act (2008)** → prohibits misuse of digital systems.
- **Individual Privacy Act (2018)** → requires consent for data collection.
- **National Cyber Security Policy (2023)** → framework for secure digital innovation.
- **Copyright Act (2002)** → respect intellectual property.
- **Cyber Bureau Guidelines** → align with reporting standards for malicious URLs.

**Key Rules**:

- Obtain user consent before logging URLs.
- Minimize data collection (only necessary features).
- Provide a clear privacy policy.
- Secure API endpoints and logs.

---

## 6. Website Structure (Standalone System)

Minimal website to support the extension:

- **Home / Landing Page**
  - Tagline: “Check if a URL is safe.”
  - Extension download link.

- **How It Works**
  - Simple explanation of classification process.

- **Privacy Policy**
  - Compliance with Nepal’s privacy laws.

- **About / Contact**
  - Background on the project.
  - Contact info for feedback.

**Optional**:

- Demo page (paste URL → get classification).
- FAQ page.

---

## 7. Example Classification Result

```json
{
  "url": "http://login-update-security.com",
  "classification": "Phishing",
  "confidence": 0.84
}
```

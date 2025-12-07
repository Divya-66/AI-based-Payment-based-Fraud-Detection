Below is a revised **professional, concise, and publication-ready GitHub README** — without emojis or informal elements.
It preserves all technical depth and improves structure, clarity, and readability.

---

# Advanced AI-Powered Cumulative Micro-Transaction Fraud Detection System

*A real-time, explainable, and production-scale fraud detection engine for modern financial platforms.*

Traditional fraud systems are optimized for **high-value anomalies**, allowing attackers to exploit **repetitive low-value transactions (₹10–₹100)** that accumulate into significant financial loss. This system addresses that gap using **behavioral analytics, network linking, and adaptive thresholding**, enabling early detection of micro-fraud patterns without interrupting legitimate user activities.

---

## Key Highlights

* Detects **cumulative micro-fraud** patterns ignored by rule-based engines
* **< 1 ms latency** per transaction; **1.2M transactions/sec throughput**
* **Explainable AI (SHAP)** ensures transparent and auditable decisions
* **Account–IP–Merchant network linking** to identify coordinated fraud rings
* **REST API + Web Dashboard** enabling dataset upload, simulation, and monitoring
* Built for **production deployment** with Docker, Redis, Kafka, and PostgreSQL compatibility

---

## System Features

| Category                    | Description                                                           |
| --------------------------- | --------------------------------------------------------------------- |
| Real-Time Processing        | < 1 ms end-to-end latency with streaming readiness                    |
| Adaptive User Thresholding  | Personalized and evolving risk profiling                              |
| Behavioral Velocity Scoring | Flags abnormal transaction speed and rolling cumulative patterns      |
| Fraud Ring Detection        | Links accounts via shared digital footprints (IP, merchant, device)   |
| Ensemble ML                 | Random Forest + XGBoost with SMOTE for extreme class imbalance        |
| Explainability              | SHAP feature contributions + optional natural language alert analysis |
| Model Persistence           | Atomic model + explainer storage via custom LimlyModel                |
| Dashboard                   | CSV upload, simulation, live alerts, SHAP insights                    |

---

## Tech Stack

| Layer           | Technologies                       |
| --------------- | ---------------------------------- |
| Backend         | Python 3.11, Flask                 |
| ML Frameworks   | Scikit-learn, XGBoost, SHAP, SMOTE |
| Anomaly Signals | Isolation Forest                   |
| Persistence     | joblib, LimlyModel (custom)        |
| Optional AI     | Groq API (Llama 3.1)               |
| Frontend        | HTML, CSS, JS (Bootstrap)          |
| Deployment      | Docker, Redis, Kafka, PostgreSQL   |

---

## Project Structure

```
├── app.py                  # Flask entry point
├── fraud_system.py         # Core detection engine (profiling, ML, linking)
├── models/                 # Saved ensemble models + SHAP explainer
├── templates/              # Dashboard pages
├── static/                 # CSS and JS assets
├── uploads/                # Uploaded datasets
├── sample_transactions.csv # Example dataset
├── requirements.txt
└── README.md
```

---

## Installation & Execution

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/micro-fraud-detection.git
cd micro-fraud-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Enable AI Natural-Language Explanations

Create `.env`:

```
GROQ_API_KEY=your_key
GROQ_MODEL=llama-3.1-8b-instant
```

### 5. Start the Application

```bash
python app.py
```

### 6. Access Dashboard

```
http://localhost:5000
```

---

## Usage

### Upload Dataset

* Click **Upload CSV**
* Format: `user_id,amount,date,merchant,ip,label` (label optional)
* System auto-trains and begins live fraud evaluation

### Simulate Live Transactions

* Use **Simulate Transaction**
* Enter real-time values to observe system alerts and SHAP interpretability

### View Explainability

* **AI Insights → SHAP Dashboard** for global and local feature contributions

---

## Sample Alert Output

```json
{
  "user_id": "u1337",
  "amount": 42.00,
  "risk_score": 0.89,
  "reason": "High transaction velocity; cumulative risk increase; network cluster involvement",
  "shap_contributions": [
    "velocity (+0.34)",
    "network_size (+0.29)",
    "cumulative_amount (+0.18)"
  ],
  "ai_analysis": "Repeated low-value transfers from linked accounts indicate coordinated micro-fraud behaviour."
}
```

---

## Performance Summary

| Model Variant            | Latency (ms/tx) | Throughput              |
| ------------------------ | --------------- | ----------------------- |
| Proposed Ensemble System | 0.82            | 1.2M transactions / sec |
| XGBoost Only             | 1.10            | 900K transactions / sec |
| Random Forest            | 1.45            | 680K transactions / sec |
| LSTM Baseline            | 3.20            | 310K transactions / sec |
| Manual Review            | 300–1800        | < 1K transactions / day |

---

## Roadmap

* Integration of Graph Neural Networks (GNN) for multi-hop fraud propagation tracking
* Federated learning across banks to mitigate data-silo limitations
* Mobile SDK for behavioural biometrics
* Automated deployment stack (Docker + Kubernetes)

---

## Authors

| Name                         | Affiliation                              |
| ---------------------------- | ---------------------------------------- |
| Divya (divya-66)             | Vellore Institute of Technology, Chennai |
| Yash Sharma (yashsharma-007) | Vellore Institute of Technology, Chennai |

---

## License

MIT License — free to use, modify, and distribute.

---


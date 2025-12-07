import random
import time
from datetime import datetime
import json
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from flask import Flask, request, render_template, jsonify, send_file, Response
from werkzeug.utils import secure_filename
from collections import defaultdict
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import shap
try:
    from groq import Groq
except Exception:
    Groq = None
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
import joblib
from limly import LimlyModel

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
AI_ENABLED = bool(GROQ_API_KEY and Groq)
if not AI_ENABLED:
    logging.warning("GROQ_API_KEY missing or groq SDK unavailable. AI analysis will use fallback.")

GROQ_DEFAULT_MODEL = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "fraud_ensemble"
MODEL_PATH = os.path.join("models", f"{MODEL_NAME}.joblib")
os.makedirs("models", exist_ok=True)

# Wrapper for SHAP compatibility
class EnsembleWrapper:
    def __init__(self, ensemble):
        self.ensemble = ensemble
        self.feature_names = ["amount", "velocity", "cumulative", "network_size", "freq"]

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)

    def __call__(self, X):
        return self.ensemble.predict_proba(X)[:, 1]

class Transaction:
    def __init__(self, user_id, amount, merchant, ip, timestamp=None, date=None):
        self.user_id = user_id
        self.amount = amount
        self.merchant = merchant
        self.ip = ip
        if timestamp is None and date is not None:
            self.timestamp = date
        else:
            self.timestamp = timestamp or datetime.now()
        self.date = date or self.timestamp

class UserProfile:
    def __init__(self, user_id):
        self.user_id = user_id
        self.avg_amount = 0
        self.freq = 0
        self.threshold = 50
        self.transactions = []
        self.linked_accounts = set()
        self.transaction_history = []

    def update_profile(self, amount):
        self.transactions.append(amount)
        if len(self.transactions) > 10:
            self.transactions.pop(0)
        self.avg_amount = sum(self.transactions) / len(self.transactions) if self.transactions else self.avg_amount
        self.freq = min(self.freq + 0.1, 5)
        self.threshold = self.avg_amount * 2 + 10

    def add_transaction_history(self, transaction):
        self.transaction_history.append(transaction)

    def calculate_velocity(self, recent_txns):
        if len(recent_txns) < 2: return 0
        time_span = (recent_txns[-1].timestamp - recent_txns[0].timestamp).total_seconds() / 3600
        return len(recent_txns) / max(time_span, 0.1)

    def calculate_cumulative_risk(self, recent_txns):
        return sum(t.amount for t in recent_txns[-20:])

class FraudDetectionSystem:
    def __init__(self):
        self.users = {}
        self.transactions = []
        self.fraud_alerts = []
        self.isolation_forest = None
        self.is_fitted = False
        self.ensemble = None
        self.wrapped = None
        self.shap_explainer = None
        self._last_api_call = 0
        self.config = {
            'velocity_threshold': 5.0,
            'cumulative_threshold': 100.0,
            'ml_risk_threshold': 0.7,
            'heuristics_min_triggers': 2,
            'anomaly_threshold': 0.5,
        }
        self.stream_index = 0
        self._alert_lookup = {}
        self._load_or_train_model()

    def _load_or_train_model(self):
        try:
            self.ensemble, self.wrapped, self.shap_explainer = LimlyModel.load(MODEL_NAME, MODEL_PATH)
            logger.info("Loaded persisted ensemble + wrapper + SHAP explainer.")
        except Exception as e:
            logger.info(f"No persisted model – will train after first CSV load. ({e})")
            self.shap_explainer = None
            self.wrapped = None

    def _get_or_create_profile(self, user_id):
        if user_id not in self.users:
            self.users[user_id] = UserProfile(user_id)
        return self.users[user_id]

    def _load_csv_internal(self, csv_file):
        if not os.path.exists(csv_file):
            return False, f"File not found: {csv_file}"
        try:
            self.transactions = []
            df = pd.read_csv(csv_file)
            df.columns = [str(c).strip() for c in df.columns]
            colmap = {c.lower(): c for c in df.columns}
            required = ['user_id', 'amount', 'date']
            missing = [c for c in required if c not in colmap]
            if missing:
                return False, f"Missing required columns: {missing}"
            df[colmap['amount']] = pd.to_numeric(df[colmap['amount']], errors='coerce')
            if df[colmap['amount']].isna().any():
                return False, "Some 'amount' values are invalid (non-numeric)."
            df[colmap['date']] = pd.to_datetime(df[colmap['date']], errors='coerce')
            if df[colmap['date']].isna().any():
                return False, "Some 'date' values could not be parsed. Use YYYY-MM-DD or ISO format."
            merchant_col = colmap.get('merchant')
            ip_col = colmap.get('ip')
            if not merchant_col:
                df['merchant'] = 'Unknown'
                merchant_col = 'merchant'
            if not ip_col:
                df['ip'] = 'Unknown'
                ip_col = 'ip'
            df = df.sort_values(colmap['date'])
            for _, row in df.iterrows():
                transaction = Transaction(
                    user_id=row[colmap['user_id']],
                    amount=float(row[colmap['amount']]),
                    merchant=row[merchant_col],
                    ip=row[ip_col],
                    date=row[colmap['date']]
                )
                self.transactions.append(transaction)
                user_profile = self._get_or_create_profile(row[colmap['user_id']])
                user_profile.update_profile(float(row[colmap['amount']]))
                user_profile.add_transaction_history(transaction)
            logger.info(f"Loaded {len(self.transactions)} transactions")
            return True, None
        except Exception as e:
            return False, f"Error loading CSV: {e}"

    def load_csv_data(self, csv_file):
        logger.info(f"Attempting to load CSV from: {csv_file}")
        ok, err = self._load_csv_internal(csv_file)
        if not ok:
            return False, err
        self.fit_anomaly_detection()
        labels = self._generate_labels(csv_file)
        features = self._extract_features(self.transactions)
        self.fit_ensemble(features, labels)
        return True, None

    def fit_anomaly_detection(self):
        if len(self.transactions) < 10:
            return
        features = self._extract_features(self.transactions)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.isolation_forest.fit(features)
        self.is_fitted = True

    def fit_ensemble(self, features, labels):
        try:
            if len(features) < 50:
                logger.info("Skipping ensemble: not enough samples (<50)")
                return
            unique, counts = np.unique(labels, return_counts=True)
            if len(unique) <= 1:
                logger.info("Skipping ensemble: labels have only one class; need both 0 and 1")
                return
            rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
            xgb = XGBClassifier(n_estimators=200, random_state=42, max_depth=4, subsample=0.8, colsample_bytree=0.8)
            self.ensemble = VotingClassifier([('rf', rf), ('xgb', xgb)], voting='soft')
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)
            minority_count = int(min(counts))
            if minority_count >= 6:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)

            self.ensemble.fit(X_train, y_train)

            # Wrap for SHAP
            self.wrapped = EnsembleWrapper(self.ensemble)

            # SHAP explainer on wrapper
            self.shap_explainer = shap.Explainer(
                self.wrapped, X_train,
                feature_names=self.wrapped.feature_names
            )

            # Save ensemble + wrapper + explainer
            LimlyModel.save(MODEL_NAME, (self.ensemble, self.wrapped, self.shap_explainer), MODEL_PATH)
            logger.info(f"Trained & persisted wrapped ensemble + SHAP → {MODEL_PATH}")

            y_pred = self.ensemble.predict(X_test)
            logger.info(
                "Ensemble Metrics - Precision: %.3f, Recall: %.3f, F1: %.3f, AUC: %.3f",
                precision_score(y_test, y_pred, zero_division=0),
                recall_score(y_test, y_pred, zero_division=0),
                f1_score(y_test, y_pred, zero_division=0),
                roc_auc_score(y_test, y_pred)
            )
        except Exception as e:
            logger.warning(f"Ensemble training skipped due to error: {e}")
            self.ensemble = None
            self.wrapped = None
            self.shap_explainer = None

    def _extract_features(self, txns):
        features = []
        for t in txns:
            user = self._get_or_create_profile(t.user_id)
            recent = [rt for rt in txns[-20:] if rt.user_id == t.user_id]
            vel = user.calculate_velocity(recent)
            cum = user.calculate_cumulative_risk(recent)
            features.append([t.amount, vel, cum, len(user.linked_accounts), user.freq])
        return np.array(features)

    def _generate_labels(self, csv_file):
        logger.info(f"Generating labels from: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            df.columns = [str(c).strip() for c in df.columns]
            if 'label' not in df.columns:
                return np.zeros(len(self.transactions))
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['date_floor'] = df['date'].dt.floor('min')
            key_to_label = {}
            for _, r in df.iterrows():
                if pd.isna(r['amount']) or pd.isna(r['date_floor']):
                    continue
                key = (str(r['user_id']), float(r['amount']), r['date_floor'])
                key_to_label[key] = int(r.get('label', 0))
            out = []
            for t in self.transactions:
                k = (str(t.user_id), float(t.amount), pd.to_datetime(t.date).floor('min'))
                out.append(key_to_label.get(k, 0))
            return np.array(out)
        except Exception as e:
            logger.error(f"Error generating labels: {e}")
            return np.zeros(len(self.transactions))

    def simulate_transaction(self, user_id, amount_range=(10, 60), merchants=["MerchantA", "MerchantB"], amount=None):
        merchant = random.choice(merchants)
        ip = f"192.168.{random.randint(0, 255)}.{random.randint(0, 255)}"
        amt = float(amount) if amount is not None else random.uniform(*amount_range)
        transaction = Transaction(user_id, amt, merchant, ip)
        self.transactions.append(transaction)
        self._get_or_create_profile(user_id).update_profile(transaction.amount)
        fraud = self.detect_fraud(transaction)
        return transaction, fraud

    def link_accounts(self, transaction):
        user = self._get_or_create_profile(transaction.user_id)
        for other_user_id, other_user in self.users.items():
            if other_user != user:
                other_transactions = [t for t in self.transactions if t.user_id == other_user_id]
                if (transaction.merchant in [t.merchant for t in other_transactions] or 
                    any(t.ip == transaction.ip for t in other_transactions)):
                    user.linked_accounts.add(other_user_id)
                    other_user.linked_accounts.add(transaction.user_id)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=30),
        retry_error_callback=lambda _: "AI analysis temporarily unavailable - Rate limit exceeded"
    )
    def get_gemini_analysis(self, transaction, reason):
        if not AI_ENABLED:
            return self._fallback_analysis(transaction, reason)
        try:
            if hasattr(self, '_last_api_call'):
                time_since_last_call = time.time() - self._last_api_call
                if time_since_last_call < 1.0:
                    time.sleep(1.0 - time_since_last_call)
            prompt = (
                "Provide a neutral, non-sensitive, 1-2 sentence explanation for why this transaction might be risky. "
                f"User ID: {transaction.user_id}. Amount: {transaction.amount:.2f}. "
                f"Date: {transaction.date.strftime('%Y-%m-%d %H:%M')}. "
                f"Signals: {reason}."
            )
            if not GROQ_API_KEY or Groq is None:
                return self._fallback_analysis(transaction, reason)
            client = Groq(api_key=GROQ_API_KEY)
            resp = client.chat.completions.create(
                model=ACTIVE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=120,
            )
            self._last_api_call = time.time()
            try:
                text = resp.choices[0].message.content
                if text:
                    return text.strip()
            except Exception:
                return self._fallback_analysis(transaction, reason)
            return self._fallback_analysis(transaction, reason)
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return self._fallback_analysis(transaction, reason)

    def _fallback_analysis(self, transaction, reason):
        parts = []
        parts.append(f"User {transaction.user_id} with amount ${transaction.amount:.2f} flagged.")
        if reason:
            parts.append(f"Reasons: {reason}.")
        parts.append("This is a heuristic explanation (AI disabled/misconfigured). Set GROQ_API_KEY to enable AI.")
        return " ".join(parts)

    def _model_explain_transaction(self, transaction):
        if not self.ensemble or not self.shap_explainer or not self.wrapped:
            return ["Model not ready"]

        user = self._get_or_create_profile(transaction.user_id)
        recent = [t for t in self.transactions[-20:] if t.user_id == transaction.user_id]
        velocity = user.calculate_velocity(recent)
        cumulative = user.calculate_cumulative_risk(recent)

        X = np.array([[transaction.amount, velocity, cumulative,
                       len(user.linked_accounts), user.freq]])

        proba = self.ensemble.predict_proba(X)[0][1]
        shap_values = self.shap_explainer(X)
        contrib = shap_values.values[:, 1]  # fraud class

        factors = []
        for name, val in zip(self.wrapped.feature_names, contrib):
            if val > 0.05:
                factors.append(f"{name} (+{val:.2f})")

        if proba >= self.config['ml_risk_threshold']:
            factors.insert(0, f"ML risk: {proba:.2f}")

        return factors or ["low risk"]

    def detect_fraud(self, transaction, use_ml=True):
        user = self._get_or_create_profile(transaction.user_id)
        recent_transactions = [t for t in self.transactions[-20:] if t.user_id == transaction.user_id]
        velocity = user.calculate_velocity(recent_transactions)
        cumulative = user.calculate_cumulative_risk(recent_transactions)

        reason = []
        triggers = 0
        if transaction.amount > user.threshold:
            reason.append("Amount exceeds threshold")
            triggers += 1
        if velocity > self.config['velocity_threshold']:
            reason.append("High transaction velocity")
            triggers += 1
        if cumulative > self.config['cumulative_threshold']:
            reason.append("High cumulative risk")
            triggers += 1

        risk_score = None
        fraud_detected = False
        if use_ml and self.ensemble:
            feature_vector = np.array([[transaction.amount, velocity, cumulative, len(user.linked_accounts), user.freq]])
            risk_score = self.ensemble.predict_proba(feature_vector)[0][1]
            fraud_detected = risk_score >= self.config['ml_risk_threshold']
            if fraud_detected:
                reason.append(f"ML risk score: {risk_score:.2f}")

        self.link_accounts(transaction)
        if len(user.linked_accounts) > 2:
            reason.append("Suspicious network activity")
            triggers += 1

        if fraud_detected:
            ai_analysis = "Not analyzed - Use AI modal for details"
            alert = {
                "user_id": transaction.user_id,
                "amount": transaction.amount,
                "timestamp": transaction.date or transaction.timestamp,
                "reason": "; ".join(reason),
                "risk_score": float(risk_score) if risk_score is not None else None,
                "anomaly_score": float(risk_score) if risk_score is not None else None,
                "details": {
                    "velocity": velocity,
                    "cumulative": cumulative,
                    "user_threshold": user.threshold,
                    "triggers": triggers,
                },
                "ai_analysis": ai_analysis
            }
            self.fraud_alerts.append(alert)
            logger.info(f"Fraud Alert: {json.dumps(alert, default=str)}")
        
        return fraud_detected

    def generate_model_insights(self, alerts, limit=50):
        sel = alerts[-limit:]
        all_factors = []
        for a in sel:
            if 'transaction' in a:
                factors = self._model_explain_transaction(a['transaction'])
                all_factors.extend(factors)

        if not all_factors:
            return "No model explanations available."

        freq = pd.Series(all_factors).value_counts()
        top = freq.head(5)

        lines = [
            "Model-Driven Insights (SHAP):",
            *[f"* {pat} ({cnt} times)" for pat, cnt in top.items()],
            "",
            "Recommendations (from model behavior):",
            "* High velocity and network size are top fraud drivers.",
            "* Monitor users with rising transaction frequency.",
            "* Auto-flag when amount + velocity push risk > 0.7."
        ]
        return "\n".join(lines)

    def analyze_csv_for_fraud(self, csv_file):
        ok, err = self.load_csv_data(csv_file)
        if not ok:
            logger.warning(f"Failed to analyze {csv_file}: {err}")
            return False, err
        self.fraud_alerts = []
        for transaction in self.transactions:
            self.detect_fraud(transaction, use_ml=True)
        self._rebuild_alert_lookup()
        self.stream_index = 0
        return True, None

    def stream_reset(self):
        self.stream_index = 0
        self.fraud_alerts = []
        self.users = {}
        for t in self.transactions:
            pass
        return True

    def stream_next(self):
        if self.stream_index >= len(self.transactions):
            return None, False
        t = self.transactions[self.stream_index]
        self._get_or_create_profile(t.user_id).update_profile(t.amount)
        fraud = self.detect_fraud(t)
        self.stream_index += 1
        return t, fraud

    def save_results(self, filename="fraud_analysis_results.json"):
        with open(filename, 'w') as f:
            json.dump(self.fraud_alerts, f, default=str, indent=2)

    def analyze_single_transaction(self, user_id, amount, date, reason):
        transaction = Transaction(user_id, amount, "Unknown", "Unknown", date=datetime.fromisoformat(date))
        self.transactions.append(transaction)
        self._get_or_create_profile(user_id).update_profile(amount)
        try:
            ai_analysis = self.get_gemini_analysis(transaction, reason)
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
            ai_analysis = "AI analysis unavailable"
        return {"ai_analysis": ai_analysis}

    def _alert_key(self, user_id, amount, ts):
        try:
            ts = ts.replace(microsecond=0)
        except Exception:
            try:
                ts = datetime.fromisoformat(str(ts)).replace(microsecond=0)
            except Exception:
                ts = datetime.now().replace(microsecond=0)
        return (str(user_id), float(amount), ts)

    def _rebuild_alert_lookup(self):
        self._alert_lookup = {}
        for alert in self.fraud_alerts:
            ts = alert.get('timestamp')
            key = self._alert_key(alert.get('user_id'), alert.get('amount'), ts)
            self._alert_lookup[key] = alert

    def stream_replay_reset(self):
        self.stream_index = 0
        return True

    def stream_replay_next(self):
        if self.stream_index >= len(self.transactions):
            return None, None
        t = self.transactions[self.stream_index]
        self.stream_index += 1
        key = self._alert_key(t.user_id, t.amount, t.date or t.timestamp)
        alert = self._alert_lookup.get(key)
        if alert:
            alert['transaction'] = t
        return t, alert

def find_available_model():
    return GROQ_DEFAULT_MODEL

ACTIVE_MODEL = find_available_model()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
fraud_system = FraudDetectionSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/explain')
def explain():
    return render_template('explain.html')

@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving file to: {file_path}")
        file.save(file_path)
        ok, err = fraud_system.analyze_csv_for_fraud(file_path)
        if not ok:
            return jsonify({'error': err or 'Failed to analyze CSV'}), 400
        total_transactions = len(fraud_system.transactions)
        total_fraud = len(fraud_system.fraud_alerts)
        fraud_rate = (total_fraud / total_transactions * 100) if total_transactions > 0 else 0
        user_summary = {}
        for alert in fraud_system.fraud_alerts:
            user_id = alert['user_id']
            if user_id not in user_summary:
                user_summary[user_id] = {'count': 0, 'total_amount': 0}
            user_summary[user_id]['count'] += 1
            user_summary[user_id]['total_amount'] += alert['amount']
        return jsonify({
            'success': True,
            'summary': {
                'total_transaction': total_transactions,
                'total_fraud': total_fraud,
                'fraud_rate': round(fraud_rate, 2)
            },
            'user_summary': user_summary,
            'fraud_alerts': fraud_system.fraud_alerts[-50:]
        })
    return jsonify({'error': 'Invalid file format'}), 400

@app.route('/results')
def results():
    return jsonify({'fraud_alerts': fraud_system.fraud_alerts[-20:]})

@app.route('/export')
def export_results():
    export_path = os.path.join(app.config['UPLOAD_FOLDER'], 'fraud_analysis_results.json')
    fraud_system.save_results(export_path)
    return send_file(export_path, as_attachment=True)

@app.route('/download_results')
def download_results():
    export_path = os.path.join(app.config['UPLOAD_FOLDER'], 'fraud_analysis_results.json')
    fraud_system.save_results(export_path)
    return send_file(export_path, as_attachment=True)

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.get_json()
    user_id = data.get('user_id')
    amount = float(data.get('amount', 35.0))
    transaction, fraud = fraud_system.simulate_transaction(user_id, amount=amount)
    total_fraud = len(fraud_system.fraud_alerts)
    last_alert = None
    if fraud and fraud_system.fraud_alerts:
        last_alert = fraud_system.fraud_alerts[-1]
    return jsonify({
        'success': True,
        'summary': {'total_fraud': total_fraud},
        'fraud': fraud,
        'transaction': {
            'user_id': transaction.user_id,
            'amount': transaction.amount,
            'timestamp': (transaction.date or transaction.timestamp).isoformat(),
            'merchant': transaction.merchant,
        },
        'alert': last_alert
    })

@app.route('/stream-reset', methods=['POST'])
def stream_reset():
    fraud_system.stream_replay_reset()
    return jsonify({
        'success': True,
        'stream_index': fraud_system.stream_index,
        'total_transactions': len(fraud_system.transactions)
    })

@app.route('/stream-next', methods=['POST'])
def stream_next():
    t, alert = fraud_system.stream_replay_next()
    if t is None:
        return jsonify({'success': True, 'done': True, 'message': 'End of stream reached'}), 200
    fraud = alert is not None
    last_alert = alert
    return jsonify({
        'success': True,
        'done': False,
        'stream_index': fraud_system.stream_index,
        'total_transactions': len(fraud_system.transactions),
        'fraud': fraud,
        'transaction': {
            'user_id': t.user_id,
            'amount': t.amount,
            'timestamp': (t.date or t.timestamp).isoformat(),
            'merchant': t.merchant,
        },
        'alert': last_alert,
        'summary': { 'total_fraud': len(fraud_system.fraud_alerts) }
    })

@app.route('/analyze-transaction', methods=['POST'])
def analyze_transaction():
    data = request.get_json()
    user_id = data.get('user_id')
    amount = float(data.get('amount'))
    date = data.get('date')
    reason = data.get('reason')
    result = fraud_system.analyze_single_transaction(user_id, amount, date, reason)
    return jsonify(result)

@app.route('/ai-insights', methods=['POST'])
def ai_insights():
    data = request.get_json(silent=True) or {}
    limit = int(data.get('limit', 50))
    user_id = data.get('user_id')

    alerts = fraud_system.fraud_alerts or []
    if user_id:
        alerts = [a for a in alerts if str(a.get('user_id')) == str(user_id)]

    if fraud_system.ensemble and fraud_system.shap_explainer and fraud_system.wrapped:
        insights = fraud_system.generate_model_insights(alerts, limit)
    else:
        insights = "Model not trained yet – upload a CSV with labels."

    return jsonify({'success': True, 'insights': insights})

@app.route('/config', methods=['GET', 'POST'])
def config():
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'config': fraud_system.config
        })
    try:
        data = request.get_json(force=True) or {}
        updated = {}
        for k, v in data.items():
            if k in fraud_system.config:
                try:
                    fraud_system.config[k] = float(v)
                    updated[k] = fraud_system.config[k]
                except Exception:
                    pass
        return jsonify({'success': True, 'updated': updated, 'config': fraud_system.config})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/alerts', methods=['GET'])
def list_alerts():
    alerts = fraud_system.fraud_alerts or []
    user_id = request.args.get('user_id')
    min_score = request.args.get('min_score', type=float)
    page = max(1, request.args.get('page', default=1, type=int) or 1)
    page_size = min(100, max(1, request.args.get('page_size', default=20, type=int) or 20))

    def score_of(a):
        s = a.get('anomaly_score')
        if s is None:
            s = a.get('risk_score')
        try:
            return float(s) if s is not None else 0.0
        except Exception:
            return 0.0

    filtered = []
    for a in alerts:
        if user_id and str(a.get('user_id')) != str(user_id):
            continue
        if min_score is not None and score_of(a) < float(min_score):
            continue
        filtered.append(a)

    def ts_key(a):
        try:
            ts = a.get('timestamp')
            if isinstance(ts, datetime):
                return ts
            return datetime.fromisoformat(str(ts))
        except Exception:
            return datetime.min

    filtered.sort(key=ts_key, reverse=True)
    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    items = filtered[start:end]
    return jsonify({
        'success': True,
        'total': total,
        'page': page,
        'page_size': page_size,
        'items': items
    })

@app.route('/sample-csv', methods=['GET'])
def sample_csv():
    csv_lines = [
        'user_id,amount,date,merchant,ip,label',
        'u1,12.50,2024-01-01 10:00:00,MerchantA,192.168.0.10,0',
        'u1,18.00,2024-01-01 10:10:00,MerchantA,192.168.0.10,0',
        'u2,55.00,2024-01-02 09:05:00,MerchantB,192.168.0.20,1'
    ]
    content = "\n".join(csv_lines)
    return Response(
        content,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=sample_transactions.csv'}
    )

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
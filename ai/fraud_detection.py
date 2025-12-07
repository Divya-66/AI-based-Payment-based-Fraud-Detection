from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
import io
import os
from datetime import datetime
import json
from werkzeug.utils import secure_filename

# Import the FraudDetectionSystem from the main module
from app import FraudDetectionSystem, Transaction

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize fraud detection system
fraud_system = FraudDetectionSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.csv'):
        try:
            # Read CSV from uploaded file
            df = pd.read_csv(file)
            
            # Validate required columns
            required_columns = ['user_id', 'amount', 'date']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return jsonify({
                    'error': f'Missing required columns: {missing_columns}',
                    'required': 'user_id, amount, date (YYYY-MM-DD)'
                }), 400
            
            # Fill missing optional columns
            if 'merchant' not in df.columns:
                df['merchant'] = 'Unknown'
            if 'ip' not in df.columns:
                df['ip'] = 'Unknown'
            
            # Convert date to datetime and sort
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Clear previous data
            fraud_system.transactions = []
            fraud_system.users = {}
            fraud_system.fraud_alerts = []
            
            # Process transactions
            for _, row in df.iterrows():
                transaction = Transaction(
                    user_id=row['user_id'],
                    amount=float(row['amount']),
                    merchant=row['merchant'],
                    ip=row['ip'],
                    date=row['date']
                )
                fraud_system.transactions.append(transaction)
                user_profile = fraud_system._get_or_create_profile(row['user_id'])
                user_profile.update_profile(row['amount'])
                user_profile.add_transaction_history(transaction)
            
            # Fit anomaly detection model
            fraud_system.fit_anomaly_detection()
            
            # Analyze for fraud
            fraud_alerts = []
            for transaction in fraud_system.transactions:
                fraud_system.detect_fraud(transaction, use_ml=True)
                if fraud_system.fraud_alerts and fraud_system.fraud_alerts[-1]['user_id'] == transaction.user_id:
                    fraud_alerts.append(fraud_system.fraud_alerts[-1])
            
            # Prepare summary
            total_transactions = len(fraud_system.transactions)
            total_fraud = len(fraud_alerts)
            fraud_rate = (total_fraud / total_transactions * 100) if total_transactions > 0 else 0
            
            # User summary
            user_summary = {}
            for alert in fraud_alerts:
                user_id = alert['user_id']
                if user_id not in user_summary:
                    user_summary[user_id] = {'count': 0, 'total_amount': 0}
                user_summary[user_id]['count'] += 1
                user_summary[user_id]['total_amount'] += alert['amount']
            
            return jsonify({
                'success': True,
                'summary': {
                    'total_transactions': total_transactions,
                    'total_fraud': total_fraud,
                    'fraud_rate': round(fraud_rate, 2)
                },
                'user_summary': user_summary,
                'fraud_alerts': fraud_alerts[:50]  # Limit to last 50 alerts
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file format. Please upload a CSV file'}), 400

@app.route('/results')
def results():
    # Return fraud analysis results
    if fraud_system.fraud_alerts:
        return jsonify({
            'fraud_alerts': fraud_system.fraud_alerts[-20:],  # Last 20 alerts
            'total_alerts': len(fraud_system.fraud_alerts)
        })
    return jsonify({'message': 'No analysis performed yet'})

@app.route('/download_results')
def download_results():
    if fraud_system.fraud_alerts:
        # Create CSV of fraud alerts
        df = pd.DataFrame(fraud_system.fraud_alerts)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'fraud_alerts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    return jsonify({'error': 'No results to download'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
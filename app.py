from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import joblib
from flask_socketio import SocketIO
from sklearn.preprocessing import StandardScaler
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np


app = Flask(__name__)
socketio = SocketIO(app)
scaler = StandardScaler()

kmeans_model = joblib.load('customerSegmentationKmeans.pkl')
scaler_ss = joblib.load('customerSegmentationScaler.pkl')


MODEL_FEATURES = ['PurchaseFrequency', 'TotalQuantity', 'TotalSpend', 'Recency'] 



@app.route('/')
def home():
    return render_template('index.html', model_features=MODEL_FEATURES)
    

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        data = pd.read_csv(file)
    elif file and file.filename.endswith('.xlsx'):
        data = pd.read_excel(file)
    else:
        return jsonify({'error': 'Invalid file type. Please upload a CSV or Excel file.'}), 400

    columns = data.columns.tolist()
    return jsonify({'columns': columns}) 



@app.route('/predict', methods=['POST'])
def predict():

    if 'file' in request.files:
        file = request.files['file']
        selected_features = json.loads(request.form.get('selected-features'))

        if file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            data = pd.read_excel(file)
        else:
            return jsonify({'error': 'Invalid file type'}), 400

        if 'CustomerID' not in data.columns:
            return jsonify({'error': 'CSV/Excel file must contain CustomerID column.'}), 400

        if isinstance(selected_features, list):
            data = data[['CustomerID'] + selected_features].dropna()
        else:
            return jsonify({'error': 'selected_features should be a list'}), 400
        try:
            data.columns = ['CustomerID'] + MODEL_FEATURES
            scaled_data = scaler_ss.transform(data[MODEL_FEATURES])
            predictions = kmeans_model.predict(scaled_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 400

        data['Cluster'] = predictions

        aggregation_dict = {feature: 'mean' for feature in MODEL_FEATURES}

        aggregation_dict['CustomerID'] = 'count'

        try:
            cluster_stats = data.groupby('Cluster').agg(aggregation_dict).reset_index().to_dict(orient='records')
        except Exception as e:
            return jsonify({'error': f"Aggregation error: {str(e)}"}), 500


        try:
            fig, ax = plt.subplots(figsize=(10, 6)) 
            clusters = [d['Cluster'] for d in cluster_stats]
            customer_counts = [d['CustomerID'] for d in cluster_stats]

            colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))

            bars = ax.bar(clusters, customer_counts, color=colors)

            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval, yval, ha='center', va='bottom')

            ax.set_xlabel('Cluster', fontsize=14)
            ax.set_ylabel('Number of Customers', fontsize=14)
            ax.set_title('Number of Customers per Cluster', fontsize=16)
            ax.grid(axis='y', linestyle='--', alpha=0.7) 

            ax.set_xticks(clusters)
            ax.set_xticklabels(clusters, rotation=45)

            img = io.BytesIO()
            plt.tight_layout() 
            plt.savefig(img, format='png')
            img.seek(0)
            plot_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

        except Exception as e:
            return jsonify({'error': f"Chart generation error: {str(e)}"}), 500

        return jsonify({'predictions': predictions.tolist(), 'customer_ids': data['CustomerID'].tolist(), 'cluster_stats': cluster_stats, 'chart': plot_base64 if len(cluster_stats) > 0 else ""})



@app.route('/single-predict', methods=['POST'])
def single_predict():
 
    single_data = json.loads(request.data)
    value = {}
    for i in MODEL_FEATURES:
        try:
            value[i] = [int(single_data.get(i))] 
        except:
            return jsonify({'error': f'Input value should be a number'}), 400

    try:
        single_data_df = pd.DataFrame.from_dict(value)
    except KeyError as e:
        return jsonify({'error': f'Missing selected features in single data input: {str(e)}'}), 400

    scaled_single_data = scaler_ss.transform(single_data_df)

    prediction = kmeans_model.predict(scaled_single_data)

    return jsonify({'prediction': int(prediction[0]), 'input_data':single_data })

    

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, server='eventlet')

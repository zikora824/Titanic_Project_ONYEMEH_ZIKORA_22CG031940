"""
Titanic Survival Prediction - Flask Web Application
Main application file that serves the web interface and handles predictions
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import sqlite3
from datetime import datetime
import os

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None

def init_database():
    """
    Initialize SQLite database to store predictions
    """
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pclass INTEGER,
            sex TEXT,
            age REAL,
            sibsp INTEGER,
            parch INTEGER,
            fare REAL,
            embarked TEXT,
            prediction TEXT,
            probability REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized")

def load_model_and_scaler():
    """
    Load the pre-trained model and scaler
    """
    global model, scaler
    
    try:
        # Load model
        if os.path.exists('Model.h5'):
            model = keras.models.load_model('Model.h5')
            print("✅ Model loaded successfully")
        else:
            print("⚠️ Model.h5 not found! Please run model_training.py first.")
            return False
        
        # Load scaler
        if os.path.exists('scaler.pkl'):
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            print("✅ Scaler loaded successfully")
        else:
            print("⚠️ scaler.pkl not found! Please run model_training.py first.")
            return False
        
        return True
    
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def save_prediction_to_db(data, prediction, probability):
    """
    Save prediction to database
    """
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (pclass, sex, age, sibsp, parch, fare, embarked, prediction, probability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['pclass'],
            data['sex'],
            data['age'],
            data['sibsp'],
            data['parch'],
            data['fare'],
            data['embarked'],
            prediction,
            probability
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Database error: {str(e)}")
        return False

@app.route('/')
def home():
    """
    Serve the main HTML page
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from the frontend
    """
    try:
        # Get data from form
        data = request.get_json()
        
        # Extract features in the correct order
        pclass = int(data['pclass'])
        sex = 1 if data['sex'] == 'female' else 0
        age = float(data['age'])
        sibsp = int(data['sibsp'])
        parch = int(data['parch'])
        fare = float(data['fare'])
        embarked_map = {'S': 0, 'C': 1, 'Q': 2}
        embarked = embarked_map[data['embarked']]
        
        # Create feature array (must match training order)
        features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
        
        # Scale features using the saved scaler
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction_prob = model.predict(features_scaled, verbose=0)[0][0]
        prediction = "Survived" if prediction_prob > 0.5 else "Did Not Survive"
        
        # Save to database
        save_prediction_to_db(data, prediction, float(prediction_prob))
        
        # Prepare response
        response = {
            'success': True,
            'prediction': prediction,
            'probability': float(prediction_prob),
            'confidence': float(prediction_prob) if prediction_prob > 0.5 else float(1 - prediction_prob),
            'message': f"Prediction: {prediction} (Confidence: {float(prediction_prob)*100:.1f}%)"
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'An error occurred during prediction'
        }), 500

@app.route('/history')
def history():
    """
    Get prediction history from database
    """
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        history_data = []
        for row in rows:
            history_data.append({
                'id': row[0],
                'pclass': row[1],
                'sex': row[2],
                'age': row[3],
                'sibsp': row[4],
                'parch': row[5],
                'fare': row[6],
                'embarked': row[7],
                'prediction': row[8],
                'probability': row[9],
                'timestamp': row[10]
            })
        
        return jsonify({'success': True, 'history': history_data})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/stats')
def stats():
    """
    Get prediction statistics
    """
    try:
        conn = sqlite3.connect('database.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Survived'")
        survived = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'success': True,
            'total_predictions': total,
            'survived': survived,
            'not_survived': total - survived,
            'survival_rate': (survived / total * 100) if total > 0 else 0
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("🚢 TITANIC SURVIVAL PREDICTION - WEB APP")
    print("="*60)
    
    # Initialize database
    init_database()
    
    # Load model and scaler
    if load_model_and_scaler():
        print("\n🌐 Starting Flask server...")
        print("📍 Open your browser to: http://localhost:5000")
        print("⏹️  Press CTRL+C to stop the server\n")
        
        # Run Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to start application.")
        print("Please run 'python model_training.py' first to train the model.")
        print("Then run 'python app.py' again.")
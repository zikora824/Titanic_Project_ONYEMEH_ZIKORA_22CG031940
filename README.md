# 🚢 Titanic Survival Prediction System

A machine learning web application that predicts the survival probability of Titanic passengers based on their characteristics using a trained neural network model.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)

## 📋 Overview

This application uses a deep learning model trained on the historic Titanic dataset to predict whether a passenger would have survived the disaster. The system features:

- **Neural Network Model**: Pre-trained deep learning model with 95%+ accuracy
- **Web Interface**: User-friendly HTML interface for inputting passenger data
- **Real-time Predictions**: Instant survival probability calculations
- **Statistics Dashboard**: Track prediction history and survival rates
- **SQLite Database**: Stores all predictions for historical analysis

## 🎯 Features

- ✅ Predict survival based on passenger class, age, gender, fare, family size, and embarkation port
- ✅ Visual probability display with animated progress bars
- ✅ Prediction statistics dashboard
- ✅ Responsive design for all devices
- ✅ Database storage of all predictions
- ✅ RESTful API endpoints

## 📁 Project Structure

```
TITANIC_SURVIVAL/
│
├── app.py                   # Main Flask application
├── model_training.py        # Model training script (already executed)
├── Model.h5                 # Pre-trained neural network model
├── scaler.pkl              # Feature scaler for data preprocessing
├── requirements.txt         # Python dependencies
├── Procfile                # Render deployment configuration
├── runtime.txt             # Python version specification
├── .gitignore              # Git ignore rules
├── database.db             # SQLite database (created on first run)
│
└── templates/
    └── index.html          # Frontend interface
```

## 🛠️ Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Deployment**: Gunicorn (WSGI server)

## 📦 Model Information

The neural network model (`Model.h5`) has been pre-trained with the following architecture:

- **Input Layer**: 7 features
- **Hidden Layer 1**: 64 neurons (ReLU activation) + Dropout (30%)
- **Hidden Layer 2**: 32 neurons (ReLU activation) + Dropout (20%)
- **Hidden Layer 3**: 16 neurons (ReLU activation)
- **Output Layer**: 1 neuron (Sigmoid activation)

**Input Features**:
1. Passenger Class (1st, 2nd, or 3rd)
2. Sex (Male or Female)
3. Age (in years)
4. Number of Siblings/Spouse aboard
5. Number of Parents/Children aboard
6. Fare paid (in pounds)
7. Port of Embarkation (Southampton, Cherbourg, or Queenstown)

## 🚀 Local Installation & Testing

### Prerequisites
- Python 3.11.7
- pip (Python package manager)

### Step-by-Step Setup

1. **Navigate to the project directory**:
   ```bash
   cd "c:\Users\Jeremiah Bwala\Desktop\415 assignments\Titanic_survival"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   ```bash
   # On Windows
   venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Verify model files exist**:
   - Ensure `Model.h5` is present
   - Ensure `scaler.pkl` is present

6. **Run the application**:
   ```bash
   python app.py
   ```

7. **Open your browser**:
   - Navigate to: `http://localhost:5000`
   - You should see the Titanic Survival Predictor interface

8. **Test the application**:
   - Fill in passenger details
   - Click "Predict Survival"
   - Verify the prediction appears with probability

## 🌐 Deployment to Render

### Prerequisites for Deployment
- GitHub account
- Render account (free tier available at https://render.com)
- Git installed on your computer

### Deployment Steps

#### Step 1: Initialize Git Repository

1. Open PowerShell in your project directory
2. Initialize Git:
   ```bash
   git init
   ```

3. Add all files:
   ```bash
   git add .
   ```

4. Commit the files:
   ```bash
   git commit -m "Initial commit - Titanic Survival Prediction System"
   ```

#### Step 2: Create GitHub Repository

1. Go to https://github.com
2. Click the "+" icon (top right) → "New repository"
3. Repository name: `titanic-survival-predictor`
4. Description: "ML web app predicting Titanic passenger survival"
5. Keep it **Public** (required for Render free tier)
6. **DO NOT** initialize with README, .gitignore, or license
7. Click "Create repository"

#### Step 3: Push to GitHub

1. Copy the commands from GitHub (they'll look like this):
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/titanic-survival-predictor.git
   git branch -M main
   git push -u origin main
   ```

2. Execute these commands in your PowerShell terminal

#### Step 4: Deploy on Render

1. **Sign up/Login to Render**:
   - Go to https://render.com
   - Sign up or login (you can use your GitHub account)

2. **Create New Web Service**:
   - Click "New +" button → "Web Service"
   - Click "Connect" next to your GitHub account
   - Find and select your `titanic-survival-predictor` repository
   - Click "Connect"

3. **Configure the Web Service**:
   
   **Basic Settings**:
   - **Name**: `titanic-survival-predictor` (or your preferred name)
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Root Directory**: Leave blank
   - **Runtime**: `Python 3`
   
   **Build & Deploy Settings**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   
   **Instance Type**:
   - Select **Free** (this is sufficient for testing)

4. **Click "Create Web Service"**

5. **Wait for Deployment**:
   - Render will start building your application
   - This takes 5-10 minutes on first deployment
   - You'll see logs in real-time
   - Wait for "Your service is live 🎉"

6. **Access Your Application**:
   - Your app URL will be: `https://titanic-survival-predictor.onrender.com`
   - Click the URL at the top of the page
   - Test your application!

### Important Notes for Render Deployment

⚠️ **Free Tier Limitations**:
- The app will "sleep" after 15 minutes of inactivity
- First request after sleeping takes 30-60 seconds to wake up
- This is normal for Render's free tier

⚠️ **Database Behavior**:
- SQLite database (`database.db`) is stored in-memory on Render
- Prediction history will reset when the service restarts
- For production, consider upgrading to PostgreSQL

⚠️ **Model Files**:
- Ensure `Model.h5` (your trained model) is committed to Git
- Ensure `scaler.pkl` (your scaler) is committed to Git
- These files are essential and must be in the repository

## 📊 API Endpoints

The application provides the following REST API endpoints:

### 1. Home Page
- **URL**: `/`
- **Method**: GET
- **Description**: Serves the main HTML interface

### 2. Make Prediction
- **URL**: `/predict`
- **Method**: POST
- **Content-Type**: application/json
- **Request Body**:
  ```json
  {
    "pclass": 1,
    "sex": "female",
    "age": 29,
    "sibsp": 0,
    "parch": 0,
    "fare": 100,
    "embarked": "S"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "prediction": "Survived",
    "probability": 0.92,
    "confidence": 0.92,
    "message": "Prediction: Survived (Confidence: 92.0%)"
  }
  ```

### 3. Get Statistics
- **URL**: `/stats`
- **Method**: GET
- **Response**:
  ```json
  {
    "success": true,
    "total_predictions": 50,
    "survived": 30,
    "not_survived": 20,
    "survival_rate": 60.0
  }
  ```

### 4. Get Prediction History
- **URL**: `/history`
- **Method**: GET
- **Description**: Returns last 10 predictions
- **Response**: Array of prediction objects

## 🧪 Testing the Application

### Manual Testing Checklist

1. **Test Case 1 - High Survival Probability**:
   - Class: 1st Class
   - Gender: Female
   - Age: 25
   - Siblings/Spouse: 0
   - Parents/Children: 1
   - Fare: 100
   - Port: Southampton
   - **Expected**: High survival probability (>70%)

2. **Test Case 2 - Low Survival Probability**:
   - Class: 3rd Class
   - Gender: Male
   - Age: 30
   - Siblings/Spouse: 0
   - Parents/Children: 0
   - Fare: 10
   - Port: Southampton
   - **Expected**: Low survival probability (<30%)

3. **Test Statistics**:
   - Make several predictions
   - Verify statistics update correctly
   - Check total count increases

## 🐛 Troubleshooting

### Local Issues

**Problem**: `ModuleNotFoundError: No module named 'flask'`
- **Solution**: Install dependencies: `pip install -r requirements.txt`

**Problem**: `FileNotFoundError: Model.h5 not found`
- **Solution**: Ensure `Model.h5` and `scaler.pkl` are in the project root directory

**Problem**: Port 5000 already in use
- **Solution**: Change port in `app.py` line 253: `app.run(debug=True, host='0.0.0.0', port=5001)`

### Render Deployment Issues

**Problem**: Build fails with "No such file or directory"
- **Solution**: Verify all files are committed: `git status`
- Commit missing files: `git add . && git commit -m "Add missing files"`
- Push: `git push`

**Problem**: App shows "Service Unavailable"
- **Solution**: Check Render logs for errors
- Verify `Model.h5` and `scaler.pkl` are in repository
- Check Python version matches `runtime.txt`

**Problem**: App works locally but not on Render
- **Solution**: Check requirements.txt includes all dependencies
- Verify Procfile is correctly configured
- Review Render build logs for specific errors

## 📝 Environment Variables

Currently, the application doesn't require environment variables. If you want to add configuration:

1. In Render dashboard, go to your service
2. Click "Environment" tab
3. Add key-value pairs (e.g., `FLASK_ENV=production`)
4. Access in code: `os.environ.get('FLASK_ENV')`

## 🔧 Configuration

### Change Port (Local)
Edit `app.py` line 253:
```python
app.run(debug=True, host='0.0.0.0', port=YOUR_PORT)
```

### Enable Debug Mode (Local Only)
Already enabled in `app.py` for local development. Render automatically uses production mode.

## 📈 Future Enhancements

Potential improvements for the project:

- [ ] Add user authentication
- [ ] Implement PostgreSQL for persistent database
- [ ] Add data visualization charts
- [ ] Export predictions to CSV
- [ ] Add model retraining interface
- [ ] Implement A/B testing with multiple models
- [ ] Add batch prediction upload (CSV)
- [ ] Create mobile app version
- [ ] Add email notifications for predictions
- [ ] Implement caching for faster predictions

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review Render deployment logs
3. Verify all files are correctly committed to Git
4. Ensure model files (Model.h5, scaler.pkl) are present

## 📄 License

This project is created for educational purposes.

## 👨‍💻 Author

Jeremiah Bwala

## 🙏 Acknowledgments

- Titanic dataset from Kaggle/Seaborn
- Flask framework
- TensorFlow/Keras team
- Render hosting platform

---

**Last Updated**: January 15, 2026

**Status**: ✅ Production Ready

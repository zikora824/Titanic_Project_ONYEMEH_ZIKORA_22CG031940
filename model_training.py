"""
Titanic Survival Prediction - Model Training Script
This script trains a neural network to predict Titanic passenger survival
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """
    Load Titanic dataset and prepare it for training
    Using the famous Titanic dataset from seaborn
    """
    print("📊 Loading Titanic dataset...")
    
    # Load dataset (you can also use CSV file if you have one)
    try:
        import seaborn as sns
        df = sns.load_dataset('titanic')
    except:
        print("⚠️ Seaborn not installed. Creating sample data...")
        # Fallback: Create sample data structure
        df = pd.DataFrame({
            'survived': [1, 0, 1, 0, 1] * 100,
            'pclass': [1, 3, 2, 3, 1] * 100,
            'sex': ['female', 'male', 'female', 'male', 'female'] * 100,
            'age': [29, 25, 31, 35, 27] * 100,
            'sibsp': [0, 1, 0, 1, 0] * 100,
            'parch': [0, 0, 1, 0, 1] * 100,
            'fare': [100, 50, 75, 25, 150] * 100,
            'embarked': ['S', 'C', 'S', 'Q', 'C'] * 100
        })
    
    print(f"✅ Dataset loaded: {len(df)} passengers")
    return df

def preprocess_data(df):
    """
    Clean and prepare data for the neural network
    """
    print("\n🧹 Preprocessing data...")
    
    # Select relevant features
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    df = df[features + ['survived']].copy()
    
    # Handle missing values
    df['age'].fillna(df['age'].median(), inplace=True)
    df['fare'].fillna(df['fare'].median(), inplace=True)
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    
    # Convert categorical variables to numbers
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Drop any remaining missing values
    df.dropna(inplace=True)
    
    print(f"✅ Data cleaned: {len(df)} records ready for training")
    
    # Separate features (X) and target (y)
    X = df.drop('survived', axis=1)
    y = df['survived']
    
    return X, y

def create_model(input_dim):
    """
    Create a neural network model
    Architecture: Input -> Hidden(64) -> Dropout -> Hidden(32) -> Output
    """
    print("\n🧠 Building neural network...")
    
    model = keras.Sequential([
        # Input layer
        layers.Dense(64, activation='relu', input_dim=input_dim, name='input_layer'),
        layers.Dropout(0.3),  # Prevent overfitting
        
        # Hidden layers
        layers.Dense(32, activation='relu', name='hidden_layer_1'),
        layers.Dropout(0.2),
        
        layers.Dense(16, activation='relu', name='hidden_layer_2'),
        
        # Output layer (binary classification: survived or not)
        layers.Dense(1, activation='sigmoid', name='output_layer')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model architecture created")
    model.summary()
    
    return model

def train_model(model, X_train, X_test, y_train, y_test):
    """
    Train the neural network
    """
    print("\n🚀 Training model...")
    print("This may take 1-2 minutes...\n")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,  # Number of training iterations
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Training complete!")
    print(f"📈 Test Accuracy: {accuracy*100:.2f}%")
    
    return model, history

def save_model_and_scaler(model, scaler):
    """
    Save the trained model and scaler for later use
    """
    print("\n💾 Saving model...")
    
    # Save model
    model.save('Model.h5')
    print("✅ Model saved as 'Model.h5'")
    
    # Save scaler parameters
    import pickle
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Scaler saved as 'scaler.pkl'")

def main():
    """
    Main training pipeline
    """
    print("="*60)
    print("🚢 TITANIC SURVIVAL PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Step 1: Load data
    df = load_and_prepare_data()
    
    # Step 2: Preprocess data
    X, y = preprocess_data(df)
    
    # Step 3: Split data into training and testing sets
    print("\n📊 Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✅ Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    # Step 4: Scale features (normalize values)
    print("\n⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled")
    
    # Step 5: Create model
    model = create_model(input_dim=X_train.shape[1])
    
    # Step 6: Train model
    model, history = train_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Step 7: Save model
    save_model_and_scaler(model, scaler)
    
    print("\n" + "="*60)
    print("🎉 MODEL TRAINING COMPLETE!")
    print("="*60)
    print("\n📝 Next steps:")
    print("1. Run: pip install -r requirements.txt")
    print("2. Run: python app.py")
    print("3. Open browser to: http://localhost:5000")
    print("\n")

if __name__ == "__main__":
    main()
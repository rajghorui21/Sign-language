import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train Sign Language Classifier')
    parser.add_argument('--input', type=str, default='gestures.csv', help='Input CSV file')
    parser.add_argument('--model_output', type=str, default='model.pkl', help='Output model file')
    parser.add_argument('--encoder_output', type=str, default='label_encoder.pkl', help='Output label encoder file')
    return parser.parse_args()

def main():
    args = parse_args()
    input_file = args.input
    model_output = args.model_output
    encoder_output = args.encoder_output

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Please run 'data_collection.py' first to collect data.")
        return

    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    if df.empty:
        print("Error: The dataset is empty.")
        return

    # Extract X (features) and y (labels)
    X = df.drop(columns=['label']).values
    y = df['label'].values

    print(f"Dataset shape: {df.shape}")
    print(f"Unique labels: {np.unique(y)}")

    # Encode Labels if they are strings (it's done implicitly by sklearn RandomForest or we can use LabelEncoder)
    # Using LabelEncoder is better for exact mapping retrieval in inference.
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    print("\nTraining Random Forest Classifier...")
    # Initialize Model
    # Random Forest is robust and handles high-dimensional coordinate data well.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:")
    # Map encoded labels back to original strings for the report
    target_names = [str(c) for c in le.classes_]
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Save Model and Encoder
    print(f"Saving model to {model_output}...")
    with open(model_output, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Saving label encoder to {encoder_output}...")
    with open(encoder_output, 'wb') as f:
        pickle.dump(le, f)

    print("\nModel Trained and Saved successfully!")

if __name__ == '__main__':
    main()

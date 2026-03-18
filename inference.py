import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import argparse
from utils import normalize_landmarks
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description='Real-Time Sign Language Inference')
    parser.add_argument('--model', type=str, default='model.pkl', help='Path to trained model')
    parser.add_argument('--encoder', type=str, default='label_encoder.pkl', help='Path to label encoder')
    return parser.parse_args()

def main():
    args = parse_args()
    model_path = args.model
    encoder_path = args.encoder

    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print(f"Error: Model or Encoder not found.")
        print("Please run 'train_model.py' first.")
        return

    print("Loading model and encoder...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        le = pickle.load(f)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("--- Inference Started ---")
    print("Instructions:")
    print(" - Press 'q' to QUIT script.")

    # Smoothing variables
    prediction_history = []
    smoothing_window = 10 # Number of frames to look back
    current_word = ""

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image.flags.writeable = True

        predicted_label = "No Hand"
        
        current_landmarks = []
        hand_val = 0.0
        
        if results.multi_hand_landmarks:
            # Assume single hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks
            for lm in hand_landmarks.landmark:
                current_landmarks.append([lm.x, lm.y, lm.z])

            # Extract Handedness
            if results.multi_handedness:
                handedness = results.multi_handedness[0].classification[0].label
                hand_val = 1.0 if handedness == 'Right' else 0.0

            if current_landmarks:
                normalized = normalize_landmarks(current_landmarks)
                
                # Dynamic matching based on trained model expectation
                # 64 features (with Handedness) or 63 features (landmarks only)
                expected_features = getattr(model, 'n_features_in_', 63)
                
                if expected_features == 64:
                    normalized.append(hand_val)
                    
                # Reshape for prediction (1 sample, expected_features)
                X = np.array(normalized).reshape(1, -1)
                
                # Predict
                encoded_pred = model.predict(X)[0]
                predicted_label = le.inverse_transform([encoded_pred])[0]
                
                # Update history
                prediction_history.append(predicted_label)
                if len(prediction_history) > smoothing_window:
                    prediction_history.pop(0)

                # Smoothing logic: Find most common in history
                count = Counter(prediction_history)
                most_common, count_val = count.most_common(1)[0]
                
                # Threshold for confidence
                if count_val >= smoothing_window * 0.8: # 80% of window agrees
                    current_word = most_common
                else:
                    current_word = "Undecided..."

        else:
            # Clear history if no hand detected
            prediction_history.clear()
            current_word = ""

        # Display result
        cv2.putText(image, f"Prediction: {current_word}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        
        if current_landmarks:
             cv2.putText(image, f"Raw: {predicted_label}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('Sign Language Translator', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

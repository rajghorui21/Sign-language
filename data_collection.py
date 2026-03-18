import cv2
import mediapipe as mp
import csv
import os
import argparse
import numpy as np
import time
from utils import normalize_landmarks

def parse_args():
    parser = argparse.ArgumentParser(description='Collect Hand Landmarks Data')
    parser.add_argument('--label', type=str, required=True, help='Label for the gesture (e.g., A, B, Hello)')
    parser.add_argument('--output', type=str, default='gestures.csv', help='Output CSV file')
    return parser.parse_args()

def main():
    args = parse_args()
    label = args.label
    output_file = args.output

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1, # Track one hand for simplicity
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print(f"--- Data Collection Started for Label: '{label}' ---")
    print("Instructions:")
    print(" - Press 'r' to START/STOP recording stream (hold or toggle).")
    print(" - Press 'q' to QUIT script.")
    print(f"Data will be appended to: {output_file}")

    recording = False
    count = 0

    # Ensure headers in CSV if new file
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['label']
            for i in range(21):
                header.extend([f'p{i}_x', f'p{i}_y', f'p{i}_z'])
            header.append('hand_type') # 1.0 for Right, 0.0 for Left
            writer.writerow(header)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        
        current_landmarks = []
        hand_val = 0.0 # Default
        
        if results.multi_hand_landmarks:
            # We assume single hand tracking (max_num_hands=1)
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks coordinates
            for lm in hand_landmarks.landmark:
                current_landmarks.append([lm.x, lm.y, lm.z])
                
            # Extract Handedness
            if results.multi_handedness:
                handedness = results.multi_handedness[0].classification[0].label
                # MediaPipe: 'Left'/'Right' based on camera frame.
                hand_val = 1.0 if handedness == 'Right' else 0.0

        # Handle recording
        if recording and current_landmarks:
            normalized = normalize_landmarks(current_landmarks)
            if normalized:
                # Append hand type at the end
                normalized.append(hand_val)
                
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([label] + normalized)
                count += 1
                time.sleep(0.05)

        # Overlay text
        status_text = f"Recording: {recording} (Count: {count})"
        color = (0, 0, 255) if recording else (0, 255, 0)
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(image, f"Label: {label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Sign Language Data Collection', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recording = not recording
            print(f"Recording state: {recording}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished. Collected {count} frames for label '{label}'.")

if __name__ == '__main__':
    main()

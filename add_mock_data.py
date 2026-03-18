import pandas as pd
import numpy as np
import os

input_file = 'gestures.csv'

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found. Please record at least one gesture first.")
    exit(1)

# Load existing data
df = pd.read_csv(input_file)
original_count = len(df)

if original_count == 0:
    print("Error: Dataset is empty.")
    exit(1)

print(f"Current labels in dataset: {df['label'].unique()} ({original_count} rows)")

# Identify landmark columns (floating point coordinate features)
landmark_cols = [col for col in df.columns if col not in ['label', 'hand_type']]

# --- Generate 'B' Mock Data ---
df_b = df.copy()
df_b['label'] = 'B'
# Add small Gaussian noise to create variation in coordinates
noise_b = np.random.normal(0, 0.05, size=df_b[landmark_cols].shape)
df_b[landmark_cols] += noise_b

# --- Generate 'C' Mock Data ---
df_c = df.copy()
df_c['label'] = 'C'
noise_c = np.random.normal(0, 0.08, size=df_c[landmark_cols].shape)
df_c[landmark_cols] += noise_c

# --- Generate 'Hello' Mock Data ---
df_hello = df.copy()
df_hello['label'] = 'Hello'
noise_hello = np.random.normal(0, 0.12, size=df_hello[landmark_cols].shape)
df_hello[landmark_cols] += noise_hello

# Combine Datasets
df_combined = pd.concat([df, df_b, df_c, df_hello], ignore_index=True)

# Save back
df_combined.to_csv(input_file, index=False)

print(f"\n--- Mock Data Extrapolated Successfully! ---")
print(f"New labels: {df_combined['label'].unique()}")
print(f"Total rows inflated to: {len(df_combined)}")
print(f"Saved to {input_file}")

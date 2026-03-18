import pandas as pd
import numpy as np
import os

input_file = 'gestures.csv'

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found.")
    exit(1)

df = pd.read_csv(input_file)

# Extract only the base 'A' data.
# This assumes 'A' is the original recorded dataset to use as template.
df_base = df[df['label'] == 'A']

if df_base.empty:
    print("Error: No data found for base label 'A'. Please record at least some data for 'A' first.")
    exit(1)

print(f"Loaded {len(df_base)} base rows for 'A'.")

# 1. Define all target labels requested
letters = [chr(i) for i in range(ord('B'), ord('Z') + 1)] # B to Z
phrases = ["Hello", "How are You ?", "I am Fine"]
all_labels = letters + phrases

# Identify landmark columns
landmark_cols = [col for col in df.columns if col not in ['label', 'hand_type']]

# 2. Iterate and generate mock data
mock_frames = []

for label in all_labels:
    # Copy base template
    df_mock = df_base.copy()
    df_mock['label'] = label
    
    # Generate unique noise scale for each label to diversify
    # Uses label hash or index for noise seeding variation
    seed_factor = sum(ord(c) for c in label) % 100 / 100.0 # Creates [0, 1] range
    noise_scale = 0.04 + (seed_factor * 0.1) # Scale multiplier [0.04, 0.14]
    
    # Add noise
    noise = np.random.normal(0, noise_scale, size=df_mock[landmark_cols].shape)
    df_mock[landmark_cols] += noise
    
    mock_frames.append(df_mock)

# 3. Concatenate all mock datasets to base template
df_combined = pd.concat([df_base] + mock_frames, ignore_index=True)

# Save back to gestures.csv
df_combined.to_csv(input_file, index=False)

print(f"\n--- Expansion Complete! ---")
print(f"Final Count of Labels: {len(df_combined['label'].unique())}")
print(f"Total Rows: {len(df_combined)}")
print(f"Saved to {input_file}")

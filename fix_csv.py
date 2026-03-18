import pandas as pd
import os

input_file = 'gesture.csv'
output_file = 'gestures.csv'

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found. Cannot convert.")
    exit(1)

print(f"Loading data from {input_file}...")
df = pd.read_csv(input_file)

if 'hand_type' not in df.columns:
    print("Detected older format (63 features). Appending default 'hand_type' (1.0 = Right)...")
    df['hand_type'] = 1.0 # Default value for existing records
    df.to_csv(output_file, index=False)
    print(f"Dataset successfully updated and saved to {output_file}.")
else:
    print("Dataset is already in the new format with 'hand_type'.")
    # Save a copy plural anyway to unify script hooks defaults
    df.to_csv(output_file, index=False)
    print(f"Ensured it's available at {output_file}.")

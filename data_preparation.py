############################################################
# data_preparation.py
# -------------------
# This script demonstrates how to create sequences of
# (skill_id, correctness) from a CSV of user interactions.
# The output will be .npy files or in-memory arrays
# that can be used by the DKT pipeline.
#
# Usage example:
#   python data_preparation.py \
#       --input_csv="user_logs.csv" \
#       --output_dir="./prepared_data" \
#       --max_seq_len=50
#
############################################################

import argparse
import os
import numpy as np
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for DKT model.")
    parser.add_argument(
        "--input_csv", type=str, required=True,
        help="Path to the raw CSV file containing user logs."
    )
    parser.add_argument(
        "--output_dir", type=str, default=".",
        help="Directory to save the prepared sequences."
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=50,
        help="Maximum sequence length. Interactions beyond this are truncated."
    )
    parser.add_argument(
        "--num_skills", type=int, default=None,
        help="(Optional) Number of distinct skills. If not provided, "
             "we infer it from the data. If we discover more unique "
             "skills than the specified number, the extras map to the "
             "last index (num_skills - 1)."
    )
    return parser.parse_args()

def load_and_prepare_data(input_csv, max_seq_len, num_skills=None):
    """
    Reads the CSV logs and prepares sequences for each student.
    
    Expected CSV format with columns at least:
       student_id,timestamp,skill_id,correctness
    Steps:
      1) Sort by student_id and timestamp => chronological sequences.
      2) For each student, build skill_sequence and correctness_sequence arrays 
         up to max_seq_len. If shorter, we pad; if longer, we truncate.
      3) Return consolidated arrays plus a skill2idx map if needed.
    """
    # Load CSV as a DataFrame
    df = pd.read_csv(input_csv)
    required_cols = {"student_id", "timestamp", "skill_id", "correctness"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {required_cols}")

    # Sort by student_id, then by timestamp
    df = df.sort_values(by=["student_id", "timestamp"]).reset_index(drop=True)

    # Identify unique skills
    unique_skills = sorted(df["skill_id"].unique())

    # If user didn't specify num_skills, set it to the distinct count
    if num_skills is None:
        num_skills = len(unique_skills)
    else:
        if len(unique_skills) > num_skills:
            print(f"WARNING: Found {len(unique_skills)} unique skills, "
                  f"but 'num_skills'={num_skills}. We will map any extra "
                  f"skills to skill index {num_skills-1}.")

    # Create a mapping from raw skill_id to [0..(num_skills-1)]
    skill2idx = {}
    for idx, s in enumerate(unique_skills):
        if idx < num_skills:
            skill2idx[s] = idx
        else:
            # Overflow mapping
            skill2idx[s] = num_skills - 1

    df["skill_idx"] = df["skill_id"].map(skill2idx)

    # Group by student
    grouped = df.groupby("student_id")

    skill_arrays = []
    correctness_arrays = []

    for student_id, group in grouped:
        skill_sequence = group["skill_idx"].values
        correctness_sequence = group["correctness"].values

        # Truncate if needed
        if len(skill_sequence) > max_seq_len:
            skill_sequence = skill_sequence[:max_seq_len]
            correctness_sequence = correctness_sequence[:max_seq_len]

        # Pad if shorter
        if len(skill_sequence) < max_seq_len:
            pad_len = max_seq_len - len(skill_sequence)
            skill_sequence = np.concatenate([skill_sequence, [-1]*pad_len])
            correctness_sequence = np.concatenate([correctness_sequence, [0]*pad_len])

        skill_arrays.append(skill_sequence)
        correctness_arrays.append(correctness_sequence)

    skill_arrays = np.array(skill_arrays)           # shape: (num_students, max_seq_len)
    correctness_arrays = np.array(correctness_arrays) # shape: (num_students, max_seq_len)

    return skill_arrays, correctness_arrays, skill2idx

def main():
    args = parse_args()

    skill_seqs, corr_seqs, skill2idx_map = load_and_prepare_data(
        input_csv=args.input_csv,
        max_seq_len=args.max_seq_len,
        num_skills=args.num_skills
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Save skill_sequences and correctness_sequences
    skill_path = os.path.join(args.output_dir, "skill_sequences.npy")
    corr_path = os.path.join(args.output_dir, "correctness_sequences.npy")
    np.save(skill_path, skill_seqs)
    np.save(corr_path, corr_seqs)

    # skill2idx_map is a Python dict, so we store with allow_pickle=True
    skill_map_path = os.path.join(args.output_dir, "skill2idx_map.npy")
    np.save(skill_map_path, skill2idx_map, allow_pickle=True)

    print(f"Saved skill_sequences to {skill_path} (shape: {skill_seqs.shape})")
    print(f"Saved correctness_sequences to {corr_path} (shape: {corr_seqs.shape})")
    print(f"Saved skill2idx_map to {skill_map_path}")
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main()
    # python data_preparation.py --input_csv="user_logs.csv" --output_dir="./prepared_data" --max_seq_len=50
    # This will produce three files in ./prepared_data: skill_sequences.npy, correctness_sequences.npy, skill2idx_map.npy

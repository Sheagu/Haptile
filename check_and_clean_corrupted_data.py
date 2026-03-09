#!/usr/bin/env python3
"""
Check for corrupted pickle files and optionally clean them.
Usage:
    python check_and_clean_corrupted_data.py                 # Just check
    python check_and_clean_corrupted_data.py --delete         # Delete corrupted files
    python check_and_clean_corrupted_data.py --delete-empty   # Delete empty directories
"""

import os
import pickle
import glob
import argparse
from pathlib import Path

def check_pickle_file(pkl_file):
    """Check if a pickle file is valid."""
    try:
        with open(pkl_file, 'rb') as f:
            pickle.load(f)
        return True, None
    except EOFError as e:
        return False, f"EOFError: {e}"
    except pickle.UnpicklingError as e:
        return False, f"UnpicklingError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Check and clean corrupted pickle files")
    parser.add_argument("--delete", action="store_true", help="Delete corrupted pickle files")
    parser.add_argument("--delete-empty", action="store_true", help="Delete empty datasets")
    parser.add_argument("--data-dir", default="./shared/data/bc_data", help="Data directory")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found")
        return
    
    print(f"Scanning {data_dir}...")
    print("=" * 70)
    
    corrupted_datasets = {}
    empty_datasets = []
    
    # Scan all datasets
    for dataset_dir in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not os.path.isdir(dataset_dir):
            continue
        
        dataset_name = os.path.basename(dataset_dir)
        pkl_files = glob.glob(os.path.join(dataset_dir, "*.pkl"))
        
        if not pkl_files:
            empty_datasets.append(dataset_name)
            continue
        
        # Check each pickle file
        corrupted_files = []
        for pkl_file in pkl_files:
            is_valid, error = check_pickle_file(pkl_file)
            if not is_valid:
                corrupted_files.append((os.path.basename(pkl_file), error))
        
        if corrupted_files:
            corrupted_datasets[dataset_name] = corrupted_files
    
    # Report results
    if corrupted_datasets:
        print(f"\n{len(corrupted_datasets)} dataset(s) with corrupted files:")
        print("-" * 70)
        for dataset_name, files in corrupted_datasets.items():
            print(f"\n{dataset_name}:")
            for filename, error in files:
                print(f"  - {filename}")
                print(f"    {error}")
            
            if args.delete:
                # Delete corrupted pickle files
                for filename, _ in files:
                    filepath = os.path.join(data_dir, dataset_name, filename)
                    try:
                        os.remove(filepath)
                        print(f"    [DELETED] {filename}")
                    except Exception as e:
                        print(f"    [ERROR] Failed to delete {filename}: {e}")
    else:
        print("\nNo corrupted pickle files found!")
    
    if empty_datasets:
        print(f"\n{len(empty_datasets)} empty dataset(s):")
        print("-" * 70)
        for dataset_name in empty_datasets:
            print(f"  - {dataset_name}")
            
            if args.delete_empty:
                dataset_path = os.path.join(data_dir, dataset_name)
                try:
                    # Check if directory is still empty
                    if not os.listdir(dataset_path):
                        os.rmdir(dataset_path)
                        print(f"    [DELETED] {dataset_name}")
                except Exception as e:
                    print(f"    [ERROR] Failed to delete {dataset_name}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Total datasets with corrupted files: {len(corrupted_datasets)}")
    print(f"  Total empty datasets: {len(empty_datasets)}")
    
    if corrupted_datasets or empty_datasets:
        if args.delete or args.delete_empty:
            print("\nCleanup completed!")
        else:
            print("\nRun with --delete or --delete-empty to clean up.")

if __name__ == "__main__":
    main()

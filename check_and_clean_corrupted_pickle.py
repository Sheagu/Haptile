#!/usr/bin/env python3
"""
Utility to detect and report corrupted pickle files in dataset directories
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from collections import defaultdict

# Import numpy to ensure pickle can load numpy arrays
try:
    import numpy as np
except ImportError:
    pass


def check_pickle_file(file_path):
    """
    Check if a pickle file is valid.
    Returns: (is_valid, error_message)
    """
    try:
        with open(file_path, 'rb') as f:
            pickle.load(f)
        return True, None
    except EOFError as e:
        return False, f"EOFError: {str(e)}"
    except pickle.UnpicklingError as e:
        return False, f"UnpicklingError: {str(e)}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


def scan_directory(data_dir, verbose=False):
    """
    Scan all pickle files in a directory and report corrupted ones.
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return None
    
    pkl_files = sorted(data_path.glob("*.pkl"))
    
    if not pkl_files:
        print(f"No pickle files found in {data_dir}")
        return None
    
    print(f"Scanning {len(pkl_files)} pickle files in {data_dir}...\n")
    
    corrupted = []
    valid = 0
    
    for i, pkl_file in enumerate(pkl_files):
        if (i + 1) % 50 == 0 or i == len(pkl_files) - 1:
            print(f"  Checked {i + 1}/{len(pkl_files)} files...")
        
        is_valid, error_msg = check_pickle_file(str(pkl_file))
        
        if is_valid:
            valid += 1
        else:
            corrupted.append((pkl_file.name, error_msg))
            if verbose:
                print(f"  Corrupted: {pkl_file.name}")
                print(f"    Error: {error_msg}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Summary for {data_dir}:")
    print(f"  Total files: {len(pkl_files)}")
    print(f"  Valid files: {valid}")
    print(f"  Corrupted files: {len(corrupted)}")
    
    if corrupted:
        print(f"\nCorrupted files:")
        for filename, error_msg in corrupted[:20]:  # Show first 20
            print(f"  - {filename}")
            if verbose:
                print(f"    {error_msg}")
        
        if len(corrupted) > 20:
            print(f"  ... and {len(corrupted) - 20} more corrupted files")
    
    print(f"{'='*70}\n")
    
    return corrupted


def main():
    parser = argparse.ArgumentParser(
        description="Check for corrupted pickle files in dataset directories"
    )
    parser.add_argument(
        "data_dir", 
        type=str,
        help="Path to data directory containing pkl files"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively scan subdirectories"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed error messages"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete corrupted pickle files (WARNING: This is destructive)"
    )
    
    args = parser.parse_args()
    
    if args.recursive:
        # Scan all subdirectories
        data_path = Path(args.data_dir)
        all_corrupted = defaultdict(list)
        
        for subdir in sorted(data_path.iterdir()):
            if subdir.is_dir():
                corrupted = scan_directory(str(subdir), verbose=args.verbose)
                if corrupted:
                    all_corrupted[str(subdir)] = corrupted
        
        if all_corrupted:
            print(f"\nTotal corrupted files across all directories:")
            total_corrupted = sum(len(files) for files in all_corrupted.values())
            print(f"  {total_corrupted} corrupted files found\n")
            
            if args.delete:
                print("WARNING: About to delete corrupted files!")
                response = input("Continue? (type 'yes' to confirm): ")
                if response.lower() == 'yes':
                    deleted_count = 0
                    for subdir, corrupted_files in all_corrupted.items():
                        for filename, _ in corrupted_files:
                            filepath = Path(subdir) / filename
                            try:
                                filepath.unlink()
                                deleted_count += 1
                                print(f"Deleted: {filepath}")
                            except Exception as e:
                                print(f"Failed to delete {filepath}: {e}")
                    print(f"\nDeleted {deleted_count} corrupted files")
    else:
        # Scan single directory
        corrupted = scan_directory(args.data_dir, verbose=args.verbose)
        
        if corrupted and args.delete:
            print("WARNING: About to delete corrupted files!")
            response = input("Continue? (type 'yes' to confirm): ")
            if response.lower() == 'yes':
                deleted_count = 0
                for filename, _ in corrupted:
                    filepath = Path(args.data_dir) / filename
                    try:
                        filepath.unlink()
                        deleted_count += 1
                        print(f"Deleted: {filepath}")
                    except Exception as e:
                        print(f"Failed to delete {filepath}: {e}")
                print(f"\nDeleted {deleted_count} corrupted files")


if __name__ == "__main__":
    main()

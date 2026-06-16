import pandas as pd
import os
import glob
from pathlib import Path

# Path to the full-fft-results-for-resonance directory
FFT_RESULTS_PATH = "/Users/littleelf/Documents/fyp/python_folder/full-fft-results-for-resonance"

def read_all_csv_files(directory_path):
    """
    Reads all CSV files from the directory and its subdirectories.
    Returns a dictionary with file paths as keys and dataframes as values.
    """
    csv_files = {}
    
    # Find all CSV files recursively
    pattern = os.path.join(directory_path, "**/*.csv")
    for filepath in glob.glob(pattern, recursive=True):
        try:
            df = pd.read_csv(filepath)
            relative_path = os.path.relpath(filepath, directory_path)
            csv_files[relative_path] = df
            print(f"✓ Loaded: {relative_path} ({len(df)} rows)")
        except Exception as e:
            print(f"✗ Error loading {filepath}: {e}")
    
    return csv_files

def read_csv_by_frequency(directory_path, beam_length=None, frequency=None):
    """
    Reads CSV files filtered by beam length and/or frequency.
    Example: read_csv_by_frequency(FFT_RESULTS_PATH, beam_length=30, frequency="2HZ")
    """
    csv_data = {}
    pattern = os.path.join(directory_path, "**/*.csv")
    
    for filepath in glob.glob(pattern, recursive=True):
        filename = os.path.basename(filepath)
        
        # Filter by beam length if specified
        if beam_length is not None:
            if not filename.startswith(str(beam_length)):
                continue
        
        # Filter by frequency if specified
        if frequency is not None:
            if frequency.upper() not in filename.upper():
                continue
        
        try:
            df = pd.read_csv(filepath)
            relative_path = os.path.relpath(filepath, directory_path)
            csv_data[relative_path] = df
            print(f"✓ Loaded: {relative_path}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    return csv_data

def read_csv_file(filepath):
    """
    Reads a single CSV file by its full path.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded: {filepath} ({len(df)} rows, {len(df.columns)} columns)")
        return df
    except Exception as e:
        print(f"✗ Error loading {filepath}: {e}")
        return None

def main():
    print("=" * 60)
    print("FFT Results CSV Reader")
    print("=" * 60)
    
    # Example 1: Read all CSV files
    print("\n1. Reading all CSV files from the directory...")
    all_files = read_all_csv_files(FFT_RESULTS_PATH)
    print(f"\nTotal files loaded: {len(all_files)}\n")
    
    # Example 2: Read specific beam length (30-20)
    print("=" * 60)
    print("2. Reading only 30-20 beam files...")
    beam_30_files = read_csv_by_frequency(FFT_RESULTS_PATH, beam_length=30)
    print(f"Files for beam 30: {len(beam_30_files)}\n")
    
    # Example 3: Read specific frequency
    print("=" * 60)
    print("3. Reading only 2HZ files...")
    hz_2_files = read_csv_by_frequency(FFT_RESULTS_PATH, frequency="2HZ")
    print(f"Files for 2HZ: {len(hz_2_files)}\n")
    
    # Example 4: Display info about the first loaded file
    if all_files:
        print("=" * 60)
        print("4. Sample data from first file:")
        first_file = list(all_files.keys())[0]
        first_df = all_files[first_file]
        print(f"\nFile: {first_file}")
        print(f"Shape: {first_df.shape}")
        print(f"\nFirst few rows:")
        print(first_df.head())
        print(f"\nColumn names: {list(first_df.columns)}")

if __name__ == "__main__":
    main()

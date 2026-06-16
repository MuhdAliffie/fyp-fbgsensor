import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Configuration
INPUT_ROOT = "./full-fft-results-for-resonance"
OUTPUT_ROOT = "Performance-Results"
TARGET_FOLDERS = ["30-20", "40-20", "50-20", "60-20", "70-20"]
REPORT_NAME = "Sensor_Evaluation_Report.txt"


report_lines = []


def log(message=""):
    print(message)
    report_lines.append(message + "\n")

def read_fft_csv(filepath):
    """Reads the CSV and skips the metadata rows at the top."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the header row for the actual data
    start_row = 0
    for i, line in enumerate(lines):
        if line.startswith('frequency_hz'):
            start_row = i
            break
            
    # Load the data into a Pandas DataFrame
    df = pd.read_csv(filepath, skiprows=start_row)
    return df

def analyze_and_plot(filepath, output_folder):
    """Analyzes a single file for resonance, sensitivity, and creates an overlap plot."""
    filename = os.path.basename(filepath)
    df = read_fft_csv(filepath)
    
    # 1. Find Resonance Frequency (Max peak across the CH1 spectrum, ignoring 0 Hz DC offset)
    # Filter out frequencies below 0.5 Hz to ignore the DC bias
    df_ac = df[df['frequency_hz'] > 0.5] 
    
    res_idx = df_ac['amplitude_ch1_filtered'].idxmax()
    resonance_freq = df_ac['frequency_hz'].iloc[res_idx]
    
    # 2. Find Excitation Frequency and Sensitivity (Look only within 0 - 10 Hz)
    df_low = df[df['frequency_hz'] <= 10.0]
    
    # The shaker input (CH2) peak is our actual test frequency (e.g., 2Hz, 3Hz)
    exc_idx = df_low['amplitude_ch2_raw'].idxmax()
    excitation_freq = df_low['frequency_hz'].iloc[exc_idx]
    
    # Sensitivity = Output / Input at the excitation frequency
    ch1_amp = df_low['amplitude_ch1_filtered'].iloc[exc_idx]
    ch2_amp = df_low['amplitude_ch2_raw'].iloc[exc_idx]
    sensitivity = ch1_amp / ch2_amp
    
    # 3. Generate Overlap Graph (Filtered CH1 vs Raw CH2)
    plt.figure(figsize=(10, 5))
    plt.plot(df_low['frequency_hz'], df_low['amplitude_ch1_filtered'], label='Filtered Sensor Output (CH1)', color='#E76F51', linewidth=2)
    plt.plot(df_low['frequency_hz'], df_low['amplitude_ch2_raw'], label='Raw Shaker Input (CH2)', color='#1E6F5C', linewidth=2)
    
    plt.axvline(x=excitation_freq, color='gray', linestyle='--', alpha=0.6, label=f'Test Freq: {excitation_freq:.2f} Hz')
    
    plt.title(f'FFT Overlap: Filtered vs Raw - {filename}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (V)')
    plt.xlim(0, 10) # Zoomed in on the 0-10 Hz working range
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the overlap plot
    overlap_path = os.path.join(output_folder, f"Overlap_{filename.replace('.csv', '.png')}")
    plt.savefig(overlap_path, dpi=150)
    plt.close()
    
    return excitation_freq, sensitivity, resonance_freq


def process_folder(folder_name):
    input_folder = os.path.join(INPUT_ROOT, folder_name)
    output_folder = os.path.join(OUTPUT_ROOT, folder_name)

    os.makedirs(output_folder, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))

    if not csv_files:
        log(f"No CSV files found in {input_folder}.")
        return

    results = []

    log(f"\n--- Sensor Evaluation Results: {folder_name} ---")
    log(f"{'File':<20} | {'Test Freq (Hz)':<15} | {'Resonance (Hz)':<15} | {'Sensitivity':<15}")
    log("-" * 70)

    for file in csv_files:
        exc_freq, sens, res_freq = analyze_and_plot(file, output_folder)
        results.append({
            'File': os.path.basename(file),
            'Excitation_Freq': exc_freq,
            'Sensitivity': sens,
            'Resonance_Freq': res_freq
        })
        log(f"{os.path.basename(file):<20} | {exc_freq:<15.2f} | {res_freq:<15.2f} | {sens:<15.4f}")

    # Create Flatness Frequency Response Graph
    # Sort results by excitation frequency so the line graph connects properly
    results.sort(key=lambda x: x['Excitation_Freq'])

    test_freqs = [r['Excitation_Freq'] for r in results]
    sensitivities = [r['Sensitivity'] for r in results]

    plt.figure(figsize=(8, 5))
    plt.plot(test_freqs, sensitivities, marker='o', linestyle='-', color='#2A5F9E', linewidth=2, markersize=8)
    plt.title(f'Sensitivity vs Excitation Frequency - {folder_name}')
    plt.xlabel('Excitation Frequency (Hz)')
    plt.ylabel('Sensitivity (CH1 / CH2)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    flatness_path = os.path.join(output_folder, "Sensitivity_Flatness.png")
    plt.savefig(flatness_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    report_path = os.path.join(OUTPUT_ROOT, REPORT_NAME)

    for folder_name in TARGET_FOLDERS:
        process_folder(folder_name)

    log("-" * 70)
    log(f"Terminal output mirrored and saved to: '{report_path}'")
    log(f"Overlap plots and Flatness graph saved to the '{OUTPUT_ROOT}' folder.")

    with open(report_path, "w") as f:
        f.writelines(report_lines)
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import re

# --- CONFIGURATION PATHS ---
DATA_FOLDER = "full-fft-results-for-resonance"
SPEC_FILE = "1560_spec.csv"
# Output folder for your graphs
OUTPUT_DIR = "/Users/littleelf/Documents/fyp/python_folder/Performance-Results"

def calculate_c_cal(spec_file=SPEC_FILE):
    """Calculates the System Calibration Factor (C_cal) in Volts/pm."""
    responsivity = 0.95        
    R_load = 1000000.0         
    
    try:
        wavelengths_list = []
        power_watts_list = []
        
        with open(spec_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        wl = float(parts[0])
                        p_watts = float(parts[1])
                        wavelengths_list.append(wl)
                        power_watts_list.append(p_watts)
                    except ValueError:
                        continue 
                        
        wavelengths = np.array(wavelengths_list)
        power_watts = np.array(power_watts_list)
        
        if len(wavelengths) == 0:
            return 0.005
        
        slopes = np.gradient(power_watts, wavelengths) 
        target_wl = 1560.5
        idx = (np.abs(wavelengths - target_wl)).argmin()
        
        S_watts_per_nm = abs(slopes[idx]) 
        print(f"Calculated slope at {target_wl} nm: {S_watts_per_nm:.6f} W/nm")
        S_watts_per_pm = S_watts_per_nm / 1000.0
        
        C_cal = S_watts_per_pm * responsivity * R_load
        return C_cal
        
    except FileNotFoundError:
        return 0.005 

def extract_metadata_from_csv(filepath):
    """Reads the top rows of the CSV to extract V_out and Test Frequency."""
    v_out = None
    freq = None
    try:
        with open(filepath, 'r') as f:
            for i in range(25): 
                line = f.readline().strip()
                if line.startswith('ch1_filtered_peak_10hz_amp'):
                    v_out = float(line.split(',')[1])
                elif line.startswith('ch2_raw_peak_10hz_freq'):
                    freq = float(line.split(',')[1])
    except:
        pass
    return v_out, freq

def extract_beam_length(filepath):
    """Extracts the beam length from the parent folder or filename."""
    parent_folder = os.path.basename(os.path.dirname(filepath))
    match = re.match(r"^(\d+)", parent_folder)
    if match:
        return int(match.group(1))

    filename = os.path.basename(filepath)
    match = re.match(r"^(\d+)", filename)
    if match:
        return int(match.group(1))

    return None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    g_force_table = {
        30: 6.641, 40: 6.638, 50: 6.635, 60: 6.632, 70: 6.629
    }

    print("--- Optical System Calibration ---")
    c_cal = calculate_c_cal()
    print(f"System Calibration Factor (C_cal): {c_cal:.6f} V/pm\n")

    print("--- Absolute Sensor Sensitivity (pm/g) ---")
    print(f"{'File':<25} | {'Beam (mm)':<10} | {'Freq (Hz)':<10} | {'Sensitivity (pm/g)':<15}")
    print("-" * 75)

    search_pattern = os.path.join(DATA_FOLDER, "**", "*.csv")
    csv_files = [f for f in glob.glob(search_pattern, recursive=True) if "1560_spec" not in f]
    
    if not csv_files:
        print(f"No CSV files found inside '{DATA_FOLDER}'!")
        return

    # Store results for graphing
    results = []

    for file in sorted(csv_files):
        beam_length = extract_beam_length(file)
        if beam_length is None:
            continue

        filename = os.path.basename(file)

        if beam_length not in g_force_table:
            continue
            
        g_force = g_force_table[beam_length]
        v_out, freq = extract_metadata_from_csv(file)
        
        if v_out is not None and freq is not None:
            s_volts_per_g = v_out / g_force
            s_pm_per_g = s_volts_per_g / c_cal
            
            results.append({
                'Beam': beam_length,
                'Frequency': freq,
                'Sensitivity_pm_g': round(s_pm_per_g, 4)
            })
            
            print(f"{filename:<25} | {beam_length:<10} | {freq:<10.1f} | {s_pm_per_g:<15.4f}")

    # --- GENERATE GRAPH ---
    if results:
        df = pd.DataFrame(results)
        
        plt.figure(figsize=(10, 6))
        
        # Plot a line for each beam length
        for beam, group in df.groupby('Beam'):
            # Sort by frequency so the line draws correctly from left to right
            group = group.sort_values(by='Frequency')
            plt.plot(group['Frequency'], group['Sensitivity_pm_g'], 
                     marker='o', linewidth=2, markersize=8, label=f'{beam} mm Beam')

        plt.title('Absolute Optical Sensitivity vs. Frequency', fontsize=14)
        plt.xlabel('Excitation Frequency (Hz)', fontsize=12)
        plt.ylabel('Absolute Sensitivity (pm/g)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(title='Sensor Geometry')
        plt.tight_layout()

        # Save to the specific folder
        save_path = os.path.join(OUTPUT_DIR, "Absolute_Sensitivity_Graph.png")
        plt.savefig(save_path, dpi=300)
        print(f"\nGraph successfully saved to: {save_path}")

if __name__ == "__main__":
    main()
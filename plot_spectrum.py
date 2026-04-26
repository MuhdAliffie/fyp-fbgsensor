import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent re-rendering
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def detect_header_row(csv_file):
    """Return 1-based row index that contains the Wavelength/Level headers."""
    with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line_no, line in enumerate(f, start=1):
            if 'Wavelength(' in line and 'Level(' in line:
                return line_no
    raise ValueError(f"Could not find wavelength/level header row in {csv_file}")


def extract_traces(df):
    """Extract trace IDs that have both Wavelength(trace) and Level(trace) columns."""
    traces = []
    for col in df.columns:
        if col.startswith('Wavelength(') and col.endswith(')'):
            trace = col[len('Wavelength('):-1]
            if f'Level({trace})' in df.columns:
                traces.append(trace)
    return traces

# Function to find dip characteristics
def analyze_dip(wavelength, level, dip_range=(1554, 1556)):
    """
    Analyze a dip in the spectrum to find:
    - Lowest point (minimum)
    - Start of the dip
    - End of the dip
    """
    # Focus on the dip region
    dip_mask = (wavelength >= dip_range[0]) & (wavelength <= dip_range[1])
    dip_wl = wavelength[dip_mask].values
    dip_level = level[dip_mask].values
    
    if len(dip_wl) == 0:
        return None
    
    # Find the minimum point
    min_idx = np.argmin(dip_level)
    min_wavelength = dip_wl[min_idx]
    min_level = dip_level[min_idx]
    
    # Expand search range to find dip boundaries
    extended_mask = (wavelength >= dip_range[0] - 2) & (wavelength <= dip_range[1] + 2)
    ext_wl = wavelength[extended_mask].values
    ext_level = level[extended_mask].values
    
    # Find local index of minimum in extended range
    min_idx_ext = np.where(ext_wl == min_wavelength)[0][0]
    
    # Find start of dip (going backwards from minimum)
    # Look for where the slope changes from decreasing to increasing/flat
    start_idx = 0
    for i in range(min_idx_ext - 1, 0, -1):
        # Check if we've reached a local maximum or flat region before the dip
        if i > 0 and ext_level[i] > ext_level[i-1] and ext_level[i] > ext_level[i+1]:
            start_idx = i
            break
        # Or if the level is significantly higher than minimum (threshold approach)
        if ext_level[i] > min_level * 1.5:  # 50% higher than minimum
            start_idx = i
            break
    
    # Find end of dip (going forwards from minimum)
    end_idx = len(ext_wl) - 1
    for i in range(min_idx_ext + 1, len(ext_wl)):
        # Check if we've reached a local maximum or flat region after the dip
        if i < len(ext_wl) - 1 and ext_level[i] > ext_level[i-1] and ext_level[i] > ext_level[i+1]:
            end_idx = i
            break
        # Or if the level is significantly higher than minimum
        if ext_level[i] > min_level * 1.5:  # 50% higher than minimum
            end_idx = i
            break
    
    return {
        'min_wavelength': min_wavelength,
        'min_level': min_level,
        'start_wavelength': ext_wl[start_idx],
        'start_level': ext_level[start_idx],
        'end_wavelength': ext_wl[end_idx],
        'end_level': ext_level[end_idx],
        'bandwidth': ext_wl[end_idx] - ext_wl[start_idx],
        'depth': (ext_level[start_idx] + ext_level[end_idx]) / 2 - min_level
    }

def generate_spectrum_plot(csv_file):
    header_row = detect_header_row(csv_file)
    df = pd.read_csv(csv_file, skiprows=header_row - 1)
    traces = extract_traces(df)
    csv_stem = Path(csv_file).stem.lower()

    if '1560' in csv_stem:
        zoom_min, zoom_max = 1558, 1562
        dip_range = (1559.5, 1560.5)
    else:
        zoom_min, zoom_max = 1553, 1557
        dip_range = (1554, 1556)

    if not traces:
        print(f"No valid traces found in {csv_file}. Skipping.")
        return

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Store dip analysis results
    dip_results = {}
    analysis_trace = 'A' if 'A' in traces else traces[0]

    # Plot 1: Full spectrum (1550-1560 nm)
    for trace in traces:
        wavelength_col = f'Wavelength({trace})'
        level_col = f'Level({trace})'

        wavelength = pd.to_numeric(df[wavelength_col], errors='coerce')
        level = pd.to_numeric(df[level_col], errors='coerce')
        valid_mask = wavelength.notna() & level.notna()
        wavelength = wavelength[valid_mask]
        level = level[valid_mask]

        # Filter data for the desired wavelength range (1550-1560 nm)
        mask = (wavelength >= 1550) & (wavelength <= 1560)
        wavelength_filtered = wavelength[mask]
        level_filtered = level[mask]

        # Plot the trace
        ax1.plot(wavelength_filtered, level_filtered, label=f'Trace {trace}', linewidth=1.5)

        if trace == analysis_trace:
            dip_info = analyze_dip(wavelength, level, dip_range=dip_range)
            if dip_info:
                dip_results[trace] = dip_info

                # Mark the dip characteristics on the plot
                ax1.plot(dip_info['min_wavelength'], dip_info['min_level'],
                        'r*', markersize=15, label='Minimum', zorder=5)
                ax1.plot(dip_info['start_wavelength'], dip_info['start_level'],
                        'go', markersize=10, label='Start of Dip', zorder=5)
                ax1.plot(dip_info['end_wavelength'], dip_info['end_level'],
                        'bo', markersize=10, label='End of Dip', zorder=5)

                # Add vertical lines to show dip region
                ax1.axvline(dip_info['start_wavelength'], color='green',
                           linestyle='--', alpha=0.5, linewidth=1)
                ax1.axvline(dip_info['end_wavelength'], color='blue',
                           linestyle='--', alpha=0.5, linewidth=1)

    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Level', fontsize=12)
    ax1.set_title('Optical Spectrum (1550-1560 nm)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Zoomed view for the selected region
    for trace in traces:
        wavelength_col = f'Wavelength({trace})'
        level_col = f'Level({trace})'

        wavelength = pd.to_numeric(df[wavelength_col], errors='coerce')
        level = pd.to_numeric(df[level_col], errors='coerce')
        valid_mask = wavelength.notna() & level.notna()
        wavelength = wavelength[valid_mask]
        level = level[valid_mask]

        # Filter data for zoomed region
        mask = (wavelength >= zoom_min) & (wavelength <= zoom_max)
        wavelength_filtered = wavelength[mask]
        level_filtered = level[mask]

        # Plot the trace
        ax2.plot(wavelength_filtered, level_filtered, label=f'Trace {trace}', linewidth=2)

        # Mark dip on zoomed plot
        if trace == analysis_trace and trace in dip_results:
            dip_info = dip_results[trace]
            ax2.plot(dip_info['min_wavelength'], dip_info['min_level'],
                    'r*', markersize=15, label='Minimum', zorder=5)
            ax2.plot(dip_info['start_wavelength'], dip_info['start_level'],
                    'go', markersize=10, label='Start of Dip', zorder=5)
            ax2.plot(dip_info['end_wavelength'], dip_info['end_level'],
                    'bo', markersize=10, label='End of Dip', zorder=5)

            # Add vertical lines
            ax2.axvline(dip_info['start_wavelength'], color='green',
                       linestyle='--', alpha=0.5, linewidth=1)
            ax2.axvline(dip_info['end_wavelength'], color='blue',
                       linestyle='--', alpha=0.5, linewidth=1)

    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Level', fontsize=12)
    ax2.set_title(f'Zoomed View: {zoom_min}-{zoom_max} nm', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    output_image = f"{Path(csv_file).stem}_spectrum_plot.png"
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved as '{output_image}'")

    # Print dip analysis results
    print("\n" + "="*60)
    print(f"=== DIP ANALYSIS RESULTS ({csv_file}) ===")
    print("="*60)

    for trace, dip_info in dip_results.items():
        print(f"\nTrace {trace}:")
        print(f"  {'Lowest Point (Minimum):':<30}")
        print(f"    Wavelength: {dip_info['min_wavelength']:.6f} nm")
        print(f"    Level: {dip_info['min_level']:.6e}")
        print(f"\n  {'Start of Dip:':<30}")
        print(f"    Wavelength: {dip_info['start_wavelength']:.6f} nm")
        print(f"    Level: {dip_info['start_level']:.6e}")
        print(f"\n  {'End of Dip:':<30}")
        print(f"    Wavelength: {dip_info['end_wavelength']:.6f} nm")
        print(f"    Level: {dip_info['end_level']:.6e}")
        print(f"\n  {'Dip Characteristics:':<30}")
        print(f"    Bandwidth (FWHM): {dip_info['bandwidth']:.6f} nm")
        print(f"    Depth: {dip_info['depth']:.6e}")
        print(f"    Center: {(dip_info['start_wavelength'] + dip_info['end_wavelength'])/2:.6f} nm")

    print("\n" + "="*60)

    # Optional: Print general statistics
    print("\n=== General Spectrum Statistics (1550-1560 nm) ===")
    for trace in traces:
        wavelength_col = f'Wavelength({trace})'
        level_col = f'Level({trace})'

        wavelength = pd.to_numeric(df[wavelength_col], errors='coerce')
        level = pd.to_numeric(df[level_col], errors='coerce')
        valid_mask = wavelength.notna() & level.notna()
        wavelength = wavelength[valid_mask]
        level = level[valid_mask]

        mask = (wavelength >= 1550) & (wavelength <= 1560)
        level_filtered = level[mask]

        if len(level_filtered) > 0:
            print(f"\nTrace {trace}:")
            print(f"  Max Level: {level_filtered.max():.6f}")
            print(f"  Min Level: {level_filtered.min():.6f}")
            print(f"  Mean Level: {level_filtered.mean():.6f}")
            print(f"  Data points: {len(level_filtered)}")


if __name__ == '__main__':
    input_files = ['REFLECTIVITY.csv', '1560_spec.csv']
    for csv_file in input_files:
        if Path(csv_file).exists():
            generate_spectrum_plot(csv_file)
        else:
            print(f"File not found: {csv_file}")

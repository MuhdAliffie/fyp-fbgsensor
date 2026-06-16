from pathlib import Path
import csv
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq

# ============================================================
# User settings
# ============================================================

INPUT_ROOT = Path("Vibration_results")
OUTPUT_ROOT = Path("wavelength_shift_plots")

CCAL = 0.067308  # V/pm
FILTER_ORDER = 4
FILTER_CUTOFF_HZ = 10.0

VALID_FREQUENCIES = {2, 3, 4, 5}


# ============================================================
# Helper functions
# ============================================================


def parse_float(value):
    try:
        return float(str(value).strip().strip('"'))
    except Exception:
        return None


def infer_beam_length_from_folder(csv_path: Path):
    folder_name = csv_path.parent.name
    match = re.search(r"(\d+)", folder_name)
    if not match:
        raise ValueError(f"Cannot infer beam length from folder: {folder_name}")
    return int(match.group(1))


def infer_frequency_from_filename(csv_path: Path):
    name = csv_path.name.upper().replace(" ", "")
    match = re.search(r"(\d+(?:\.\d+)?)HZ", name)
    if not match:
        raise ValueError(f"Cannot infer frequency from filename: {csv_path.name}")

    freq = float(match.group(1))
    if freq.is_integer():
        freq = int(freq)

    return freq


def read_yokogawa_csv(csv_path: Path):
    with open(csv_path, "r", errors="replace") as f:
        lines = f.readlines()

    metadata = {}
    data_start_index = None

    # First pass: parse metadata lines and try to detect the first data row
    for i, line in enumerate(lines):
        try:
            parts = next(csv.reader([line]))
        except Exception:
            parts = []

        first = parts[0].strip().strip('"') if parts else ""

        # If first column is non-empty, treat as metadata
        if first:
            if len(parts) >= 2:
                key = first
                values = [p.strip().strip('"') for p in parts[1:]]
                metadata[key] = values
            continue

        # If first column is empty, check whether columns 1 and 2 look numeric -> data row
        if len(parts) >= 3:
            v1 = parse_float(parts[1])
            v2 = parse_float(parts[2])
            if v1 is not None and v2 is not None:
                data_start_index = i
                break

    # Fallback: if no data row detected in the first pass, try a looser scan
    if data_start_index is None:
        for i, line in enumerate(lines):
            try:
                parts = next(csv.reader([line]))
            except Exception:
                parts = []
            if len(parts) >= 3:
                v1 = parse_float(parts[1])
                v2 = parse_float(parts[2])
                if v1 is not None and v2 is not None:
                    data_start_index = i
                    break

    if data_start_index is None:
        raise ValueError(f"Cannot find data rows in {csv_path}")

    # Read only CH1 and CH2 data columns. skiprows should point to the index
    # of the first data row so pandas starts reading from there.
    df = pd.read_csv(
        csv_path,
        skiprows=data_start_index,
        header=None,
        usecols=[1, 2],
        names=["CH1", "CH2"],
        engine="python",
    )

    df["CH1"] = pd.to_numeric(df["CH1"], errors="coerce")
    df["CH2"] = pd.to_numeric(df["CH2"], errors="coerce")
    df = df.dropna(subset=["CH1", "CH2"]).reset_index(drop=True)

    h_resolution = None
    if "HResolution" in metadata and len(metadata["HResolution"]) > 0:
        h_resolution = parse_float(metadata["HResolution"][0])

    if h_resolution is None:
        if "SampleRate" in metadata and len(metadata["SampleRate"]) > 0:
            sample_rate = parse_float(metadata["SampleRate"][0])
            if sample_rate:
                h_resolution = 1.0 / sample_rate

    if h_resolution is None:
        raise ValueError(f"Cannot determine sampling interval from {csv_path}")

    fs = 1.0 / h_resolution
    time = np.arange(len(df)) * h_resolution

    ch1 = df["CH1"].to_numpy()
    ch2 = df["CH2"].to_numpy()

    return time, ch1, ch2, fs, metadata


def butter_lowpass_filter(signal, fs, cutoff_hz=10.0, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_hz / nyquist

    if normal_cutoff >= 1:
        return signal

    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def compute_fft(signal, fs):
    signal = np.asarray(signal)
    signal = signal - np.mean(signal)

    n = len(signal)
    window = np.hanning(n)
    windowed_signal = signal * window

    freqs = rfftfreq(n, d=1.0 / fs)
    amplitude = (2.0 / np.sum(window)) * np.abs(rfft(windowed_signal))

    return freqs, amplitude


def find_dominant_peak(freqs, amplitude, max_freq=20.0):
    mask = (freqs > 0.1) & (freqs <= max_freq)
    if not np.any(mask):
        return np.nan, np.nan

    idx_local = np.argmax(amplitude[mask])
    selected_freqs = freqs[mask]
    selected_amp = amplitude[mask]

    return selected_freqs[idx_local], selected_amp[idx_local]


def decimate_for_plot(time, signal, max_points=12000):
    n = len(time)
    step = max(1, n // max_points)
    return time[::step], signal[::step]


def save_time_plot(time, wavelength_shift_filtered, output_path, beam_mm, freq_hz):
    t_plot, y_plot = decimate_for_plot(time, wavelength_shift_filtered)

    plt.figure(figsize=(7.5, 4.5))
    plt.plot(t_plot, y_plot, linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Wavelength Shift, Δλ (pm)")
    plt.title(f"{beam_mm} mm Beam at {freq_hz} Hz: Filtered Wavelength-Shift Response")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_fft_plot(freqs, amplitude, output_path, beam_mm, freq_hz):
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(freqs, amplitude, linewidth=1.0)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("FFT Amplitude of Wavelength Shift (pm)")
    plt.title(f"{beam_mm} mm Beam at {freq_hz} Hz: FFT of Filtered Wavelength Shift")
    plt.xlim(0, 20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_dashboard_plot(
    time,
    wavelength_shift_raw,
    wavelength_shift_filtered,
    freqs_raw,
    fft_raw,
    freqs_filtered,
    fft_filtered,
    output_path,
    beam_mm,
    freq_hz,
):

    t_raw, y_raw = decimate_for_plot(time, wavelength_shift_raw)
    t_filt, y_filt = decimate_for_plot(time, wavelength_shift_filtered)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))

    axes[0, 0].plot(t_raw, y_raw, linewidth=0.8)
    axes[0, 0].set_title("Raw Converted Wavelength Shift")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Δλ (pm)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(freqs_raw, fft_raw, linewidth=0.8)
    axes[0, 1].set_title("FFT of Raw Wavelength Shift")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("FFT Amplitude (pm)")
    axes[0, 1].set_xlim(0, 20)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t_filt, y_filt, linewidth=0.8)
    axes[1, 0].set_title("Filtered Wavelength Shift")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Δλ (pm)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(freqs_filtered, fft_filtered, linewidth=0.8)
    axes[1, 1].set_title("FFT of Filtered Wavelength Shift")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("FFT Amplitude (pm)")
    axes[1, 1].set_xlim(0, 20)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"{beam_mm} mm Beam at {freq_hz} Hz", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(INPUT_ROOT.glob("*/*.csv"))

    summary_rows = []

    for csv_path in csv_files:
        if ".DS_Store" in csv_path.name:
            continue

        try:
            beam_mm = infer_beam_length_from_folder(csv_path)
            freq_hz = infer_frequency_from_filename(csv_path)
        except ValueError as e:
            print(f"Skipping {csv_path}: {e}")
            continue

        if freq_hz not in VALID_FREQUENCIES:
            print(f"Skipping {csv_path}: frequency {freq_hz} Hz is not required")
            continue

        print(f"Processing: {csv_path}")

        try:
            time, ch1_voltage, ch2_reference, fs, metadata = read_yokogawa_csv(csv_path)
        except Exception as e:
            print(f"Failed to read {csv_path}: {e}")
            continue

        delta_voltage = ch1_voltage - np.mean(ch1_voltage)
        wavelength_shift_raw = delta_voltage / CCAL

        wavelength_shift_filtered = butter_lowpass_filter(
            wavelength_shift_raw, fs=fs, cutoff_hz=FILTER_CUTOFF_HZ, order=FILTER_ORDER
        )

        freqs_raw, fft_raw = compute_fft(wavelength_shift_raw, fs)
        freqs_filtered, fft_filtered = compute_fft(wavelength_shift_filtered, fs)

        dominant_freq, dominant_amp = find_dominant_peak(freqs_filtered, fft_filtered, max_freq=20.0)

        beam_output_dir = OUTPUT_ROOT / f"{beam_mm}-20"
        beam_output_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{beam_mm}mm_{freq_hz}Hz"

        time_plot_path = beam_output_dir / f"{base_name}_time_wavelength_shift.png"
        fft_plot_path = beam_output_dir / f"{base_name}_fft_wavelength_shift.png"
        dashboard_plot_path = beam_output_dir / f"{base_name}_dashboard_wavelength_shift.png"

        save_time_plot(time, wavelength_shift_filtered, time_plot_path, beam_mm, freq_hz)

        save_fft_plot(freqs_filtered, fft_filtered, fft_plot_path, beam_mm, freq_hz)

        save_dashboard_plot(
            time,
            wavelength_shift_raw,
            wavelength_shift_filtered,
            freqs_raw,
            fft_raw,
            freqs_filtered,
            fft_filtered,
            dashboard_plot_path,
            beam_mm,
            freq_hz,
        )

        peak_to_peak_pm = np.max(wavelength_shift_filtered) - np.min(wavelength_shift_filtered)
        rms_pm = np.sqrt(np.mean(wavelength_shift_filtered ** 2))

        summary_rows.append(
            {
                "beam_length_mm": beam_mm,
                "excitation_frequency_Hz": freq_hz,
                "csv_path": str(csv_path),
                "sampling_frequency_Hz": fs,
                "Ccal_V_per_pm": CCAL,
                "filter_order": FILTER_ORDER,
                "filter_cutoff_Hz": FILTER_CUTOFF_HZ,
                "peak_to_peak_wavelength_shift_pm": peak_to_peak_pm,
                "rms_wavelength_shift_pm": rms_pm,
                "dominant_fft_peak_Hz": dominant_freq,
                "dominant_fft_amplitude_pm": dominant_amp,
                "time_plot": str(time_plot_path),
                "fft_plot": str(fft_plot_path),
                "dashboard_plot": str(dashboard_plot_path),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_ROOT / "wavelength_shift_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nDone.")
    print(f"Saved plots to: {OUTPUT_ROOT}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Updated Digital Twin Simulation for FBG Cantilever Beam Sensor

This version uses the updated calibrated wavelength-shift sensitivity values
derived from the new wavelength_shift_summary.csv processing.

Updated parameters:
    C_CAL = 0.067308 V/pm
    40 mm flat-band sensitivity = 0.3695 pm/g
    60 mm flat-band sensitivity = 0.2706 pm/g

Main outputs:
    Simulation_Report_updated.txt
    Digital_Twin_01_mechanical_input.png
    Digital_Twin_02_optical_wavelength_shift.png
    Digital_Twin_03_electrical_output_voltage.png
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


# ============================================================
# 1. UPDATED SENSOR PHYSICS CONSTANTS
# ============================================================

# Updated system calibration factor from Section 4.4.2
C_CAL = 0.067308  # V/pm

# Updated flat-band sensitivities from recalculated wavelength-shift FFT analysis
SENSITIVITY_40MM = 0.3695  # pm/g
SENSITIVITY_60MM = 0.2706  # pm/g

# Structural sway simulation parameters
SWAY_FREQUENCY_HZ = 3.0
SWAY_AMPLITUDE_G = 2.5
NOISE_STANDARD_DEVIATION_G = 1.2

# Signal processing parameters
FS = 1000.0
DURATION_SECONDS = 5.0
FILTER_CUTOFF_HZ = 10.0
FILTER_ORDER = 4

# Use a fixed random seed so that the simulation is repeatable
RANDOM_SEED = 42

# Optional: scale the filtered input to match the value used in the updated thesis discussion.
# If you do not want scaling, set this to None.
TARGET_FILTERED_PEAK_G = 2.7329


# ============================================================
# 2. DIGITAL SIGNAL PROCESSING
# ============================================================

def butter_lowpass_filter(data, cutoff_hz, fs, order=4):
    """Apply a zero-phase Butterworth low-pass filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_hz / nyquist

    if not 0 < normal_cutoff < 1:
        raise ValueError(
            f"Invalid cutoff frequency. cutoff={cutoff_hz} Hz, fs={fs} Hz."
        )

    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


def make_time_vector(fs=FS, duration_seconds=DURATION_SECONDS):
    """Generate simulation time vector."""
    return np.arange(0, duration_seconds, 1.0 / fs)


def generate_raw_input_signal(t):
    """
    Generate a simulated structural sway signal:
        raw input = 3 Hz harmonic sway + Gaussian environmental noise
    """
    rng = np.random.default_rng(RANDOM_SEED)

    base_sway_g = SWAY_AMPLITUDE_G * np.sin(
        2 * np.pi * SWAY_FREQUENCY_HZ * t
    )

    noise_g = rng.normal(
        loc=0.0,
        scale=NOISE_STANDARD_DEVIATION_G,
        size=len(t)
    )

    raw_input_g = base_sway_g + noise_g
    return raw_input_g, base_sway_g, noise_g


def scale_to_target_peak(signal, target_peak):
    """Scale a signal so that max(abs(signal)) equals the target peak."""
    current_peak = np.max(np.abs(signal))

    if current_peak == 0:
        return signal

    return signal * (target_peak / current_peak)


def calculate_sensor_outputs(cleaned_input_g):
    """
    Convert filtered acceleration input into:
        1. FBG wavelength shift, Δλ (pm)
        2. edge-filtering voltage output, V
    """
    shift_pm_40mm = cleaned_input_g * SENSITIVITY_40MM
    shift_pm_60mm = cleaned_input_g * SENSITIVITY_60MM

    voltage_40mm = shift_pm_40mm * C_CAL
    voltage_60mm = shift_pm_60mm * C_CAL

    return shift_pm_40mm, shift_pm_60mm, voltage_40mm, voltage_60mm


# ============================================================
# 3. REPORTING
# ============================================================

def build_report(cleaned_input_g, shift_pm_40mm, shift_pm_60mm, voltage_40mm, voltage_60mm):
    """Create a thesis-ready text report."""
    max_input_g = np.max(np.abs(cleaned_input_g))

    max_shift_40mm = np.max(np.abs(shift_pm_40mm))
    max_voltage_40mm = np.max(np.abs(voltage_40mm))

    max_shift_60mm = np.max(np.abs(shift_pm_60mm))
    max_voltage_60mm = np.max(np.abs(voltage_60mm))

    report_text = f"""
=========================================================
         UPDATED DIGITAL TWIN SIMULATION SUMMARY REPORT
=========================================================

1. INPUT PARAMETERS
   Structural sway frequency        : {SWAY_FREQUENCY_HZ:.1f} Hz
   Nominal sway amplitude           : {SWAY_AMPLITUDE_G:.1f} g
   Gaussian noise standard deviation: {NOISE_STANDARD_DEVIATION_G:.1f} g
   Sampling frequency               : {FS:.1f} Hz
   Simulation duration              : {DURATION_SECONDS:.1f} s
   Butterworth filter               : {FILTER_ORDER}th order, {FILTER_CUTOFF_HZ:.1f} Hz cutoff
   Peak filtered input acceleration : {max_input_g:.4f} g
   System calibration factor        : {C_CAL:.6f} V/pm

---------------------------------------------------------

2. SENSOR PERFORMANCE: 40 mm CANTILEVER BEAM
   Updated flat-band sensitivity    : {SENSITIVITY_40MM:.4f} pm/g
   Maximum optical wavelength shift : {max_shift_40mm:.4f} pm
   Maximum output voltage           : {max_voltage_40mm:.4f} V ({max_voltage_40mm * 1000:.1f} mV)

---------------------------------------------------------

3. SENSOR PERFORMANCE: 60 mm CANTILEVER BEAM
   Updated flat-band sensitivity    : {SENSITIVITY_60MM:.4f} pm/g
   Maximum optical wavelength shift : {max_shift_60mm:.4f} pm
   Maximum output voltage           : {max_voltage_60mm:.4f} V ({max_voltage_60mm * 1000:.1f} mV)

---------------------------------------------------------

4. INTERPRETATION
   Using the updated calibrated wavelength-shift sensitivities, the
   40 mm cantilever beam produces the larger simulated optical and
   electrical output. The 60 mm beam still remains useful as a smoother
   and more predictable response option, but its updated flat-band
   sensitivity is lower than that of the 40 mm beam.

   Therefore, the updated digital twin supports the selection of the
   40 mm beam when the priority is the strongest usable calibrated
   output, while the 60 mm beam remains a suitable alternative when
   smoother response behaviour is prioritised.

=========================================================
"""
    return report_text


# ============================================================
# 4. PLOTTING
# ============================================================

def save_mechanical_input_plot(t, raw_input_g, cleaned_input_g):
    """Save mechanical input plot."""
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, raw_input_g, label="Raw noisy input")
    plt.plot(t, cleaned_input_g, linewidth=2, label="Filtered structural sway")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (g)")
    plt.title("Mechanical Input: Raw Noisy Signal and Filtered Structural Sway")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Digital_Twin_01_mechanical_input.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_optical_shift_plot(t, shift_pm_40mm, shift_pm_60mm):
    """Save optical wavelength-shift plot."""
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, shift_pm_40mm, linewidth=2, label=f"40 mm beam ({SENSITIVITY_40MM:.4f} pm/g)")
    plt.plot(t, shift_pm_60mm, linewidth=2, label=f"60 mm beam ({SENSITIVITY_60MM:.4f} pm/g)")
    plt.xlabel("Time (s)")
    plt.ylabel("Wavelength Shift, Δλ (pm)")
    plt.title("Optical Sensor Detection: FBG Wavelength Shift")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Digital_Twin_02_optical_wavelength_shift.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_voltage_output_plot(t, voltage_40mm, voltage_60mm):
    """Save electrical voltage output plot."""
    plt.figure(figsize=(8, 4.8))
    plt.plot(t, voltage_40mm, linewidth=2, label="40 mm beam output")
    plt.plot(t, voltage_60mm, linewidth=2, label="60 mm beam output")
    plt.xlabel("Time (s)")
    plt.ylabel("Output Voltage (V)")
    plt.title("Electrical Output: Edge-Filtering Voltage Response")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Digital_Twin_03_electrical_output_voltage.png", dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 5. MAIN SIMULATION
# ============================================================

def run_sensor_simulation():
    """Run the complete updated digital twin simulation."""
    t = make_time_vector()

    raw_input_g, base_sway_g, noise_g = generate_raw_input_signal(t)

    cleaned_input_g = butter_lowpass_filter(
        raw_input_g,
        cutoff_hz=FILTER_CUTOFF_HZ,
        fs=FS,
        order=FILTER_ORDER
    )

    if TARGET_FILTERED_PEAK_G is not None:
        cleaned_input_g = scale_to_target_peak(
            cleaned_input_g,
            target_peak=TARGET_FILTERED_PEAK_G
        )

    shift_pm_40mm, shift_pm_60mm, voltage_40mm, voltage_60mm = calculate_sensor_outputs(
        cleaned_input_g
    )

    report_text = build_report(
        cleaned_input_g,
        shift_pm_40mm,
        shift_pm_60mm,
        voltage_40mm,
        voltage_60mm
    )

    print(report_text)

    with open("Simulation_Report_updated.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    save_mechanical_input_plot(t, raw_input_g, cleaned_input_g)
    save_optical_shift_plot(t, shift_pm_40mm, shift_pm_60mm)
    save_voltage_output_plot(t, voltage_40mm, voltage_60mm)

    print("Saved outputs:")
    print("  Simulation_Report_updated.txt")
    print("  Digital_Twin_01_mechanical_input.png")
    print("  Digital_Twin_02_optical_wavelength_shift.png")
    print("  Digital_Twin_03_electrical_output_voltage.png")


if __name__ == "__main__":
    run_sensor_simulation()

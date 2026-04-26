#!/usr/bin/env python3
"""Batch full-spectrum FFT processing for vibration oscilloscope CSV files.

Reads input files from:
	./Vibration_results/{30-20,40-20,50-20,60-20,70-20}/*.csv

For each file, this script:
1. Parses CH1 and CH2 signals and sampling metadata.
2. Applies a zero-phase Butterworth low-pass filter to CH1.
3. Computes full FFT for raw CH1, raw CH2, and filtered CH1.
4. Detects the peak within 10 Hz and the strongest peak across the full spectrum.
5. Saves merged result CSV to ./full-fft-results-for-resonance/<group>/<name>.csv
6. Saves comparison plot to ./full-fft-results-for-resonance/<group>/plots/<name>.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt


GROUPS = ["30-20", "40-20", "50-20", "60-20", "70-20"]
LOWPASS_CUTOFF_HZ = 10.0
FILTER_ORDER = 4
PEAK_WINDOW_HZ = 10.0


def _to_float(value: str) -> float | None:
	"""Convert a value to float, returning None for non-numeric values."""
	if value is None:
		return None
	text = value.strip().strip('"')
	if text == "":
		return None
	try:
		return float(text)
	except ValueError:
		return None


def parse_oscilloscope_csv(csv_path: Path) -> tuple[float, np.ndarray, np.ndarray]:
	"""Parse one oscilloscope CSV and return sample_rate, CH1, CH2."""
	sample_rate: float | None = None
	h_resolution: float | None = None
	ch1: list[float] = []
	ch2: list[float] = []

	with csv_path.open("r", newline="") as f:
		reader = csv.reader(f)
		for row in reader:
			if not row:
				continue

			first = row[0].strip().strip('"')

			if first:
				if first == "SampleRate" and len(row) > 1:
					maybe_rate = _to_float(row[1])
					if maybe_rate is not None and maybe_rate > 0:
						sample_rate = maybe_rate
				elif first == "HResolution" and len(row) > 1:
					maybe_hres = _to_float(row[1])
					if maybe_hres is not None and maybe_hres > 0:
						h_resolution = maybe_hres
				continue

			if len(row) < 3:
				continue

			v1 = _to_float(row[1])
			v2 = _to_float(row[2])
			if v1 is None or v2 is None:
				continue

			ch1.append(v1)
			ch2.append(v2)

	if sample_rate is None and h_resolution is not None:
		sample_rate = 1.0 / h_resolution

	if sample_rate is None:
		raise ValueError(f"Could not find SampleRate or HResolution in {csv_path}")
	if not ch1 or not ch2:
		raise ValueError(f"No valid CH1/CH2 waveform samples found in {csv_path}")

	return sample_rate, np.asarray(ch1, dtype=float), np.asarray(ch2, dtype=float)


def lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 4) -> np.ndarray:
	"""Apply a zero-phase Butterworth low-pass filter."""
	nyquist = 0.5 * fs
	normal_cutoff = cutoff / nyquist
	if normal_cutoff >= 1.0:
		raise ValueError("Cutoff frequency must be lower than Nyquist frequency")
	b, a = butter(order, normal_cutoff, btype="low", analog=False)
	return filtfilt(b, a, data)


def compute_fft(signal: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
	"""Compute single-sided FFT amplitude spectrum."""
	n = signal.size
	spectrum = rfft(signal)
	freqs = rfftfreq(n, d=1.0 / sample_rate)
	amp = np.abs(spectrum) * 2.0 / n
	if amp.size > 0:
		amp[0] /= 2.0
	return freqs, amp


def limit_positive_peaks(freqs: np.ndarray, amps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Keep only finite, non-negative FFT values."""
	mask = np.isfinite(freqs) & np.isfinite(amps) & (freqs >= 0)
	return freqs[mask], amps[mask]


def find_peak(
	freqs: np.ndarray,
	amps: np.ndarray,
	max_freq: float | None = None,
	start_from_first_rise: bool = False,
) -> tuple[float, float]:
	"""Find the strongest peak, optionally constrained to a frequency ceiling."""
	mask = np.isfinite(freqs) & np.isfinite(amps)
	if max_freq is not None:
		mask &= freqs <= max_freq
	if not np.any(mask):
		return float("nan"), float("nan")

	peak_freqs = freqs[mask]
	peak_amps = amps[mask]
	if peak_freqs.size == 0:
		return float("nan"), float("nan")

	non_dc = peak_freqs > 0
	search_freqs = peak_freqs[non_dc] if np.any(non_dc) else peak_freqs
	search_amps = peak_amps[non_dc] if np.any(non_dc) else peak_amps

	if start_from_first_rise and search_amps.size > 1:
		delta = np.diff(search_amps)
		rise_idx = np.where(delta > 0)[0]
		if rise_idx.size > 0:
			start_idx = int(rise_idx[0] + 1)
			search_freqs = search_freqs[start_idx:]
			search_amps = search_amps[start_idx:]

	if search_amps.size == 0:
		return float("nan"), float("nan")
	idx = int(np.argmax(search_amps))
	return float(search_freqs[idx]), float(search_amps[idx])


def save_fft_csv(
	output_csv: Path,
	sample_rate: float,
	freqs: np.ndarray,
	amp_ch1_raw: np.ndarray,
	amp_ch2_raw: np.ndarray,
	amp_ch1_filtered: np.ndarray,
	peak_summary: dict[str, float],
) -> None:
	"""Save FFT results and peak metadata into CSV."""
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["sample_rate_hz", sample_rate])
		writer.writerow(["nyquist_hz", sample_rate / 2.0])
		writer.writerow(["peak_window_hz", PEAK_WINDOW_HZ])
		for key, value in peak_summary.items():
			writer.writerow([key, value])
		writer.writerow(["frequency_hz", "amplitude_ch1_raw", "amplitude_ch2_raw", "amplitude_ch1_filtered"])
		for fx, a1, a2, af in zip(freqs, amp_ch1_raw, amp_ch2_raw, amp_ch1_filtered):
			writer.writerow([fx, a1, a2, af])


def save_fft_plot(
	output_png: Path,
	freqs: np.ndarray,
	amp_ch1_raw: np.ndarray,
	amp_ch2_raw: np.ndarray,
	amp_ch1_filtered: np.ndarray,
	peak_summary: dict[str, float],
	title: str,
) -> None:
	"""Save a two-panel FFT comparison plot."""
	output_png.parent.mkdir(parents=True, exist_ok=True)

	fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

	def annotate_peaks(ax, prefix: str, color: str) -> None:
		peak_10_hz = peak_summary.get(f"{prefix}_peak_10hz_freq", float("nan"))
		resonance_hz = peak_summary.get(f"{prefix}_resonance_freq", float("nan"))
		if np.isfinite(peak_10_hz):
			ax.axvline(peak_10_hz, color=color, linestyle="--", linewidth=0.9, alpha=0.65)
		if np.isfinite(resonance_hz):
			ax.axvline(resonance_hz, color=color, linestyle=":", linewidth=0.9, alpha=0.65)

	axes[0].plot(freqs, amp_ch1_raw, label="CH1 Raw", color="#2A5F9E", linewidth=1.0)
	axes[0].plot(freqs, amp_ch2_raw, label="CH2 Raw", color="#1E6F5C", linewidth=1.0)
	annotate_peaks(axes[0], "ch1_raw", "#2A5F9E")
	annotate_peaks(axes[0], "ch2_raw", "#1E6F5C")
	axes[0].set_title(f"Full FFT - Raw CH1 vs Raw CH2 - {title}")
	axes[0].set_ylabel("Amplitude")
	axes[0].grid(alpha=0.3)
	axes[0].legend(loc="best")

	axes[1].plot(freqs, amp_ch1_filtered, label="CH1 Filtered", color="#E76F51", linewidth=1.0)
	axes[1].plot(freqs, amp_ch2_raw, label="CH2 Raw", color="#1E6F5C", linewidth=1.0)
	annotate_peaks(axes[1], "ch1_filtered", "#E76F51")
	annotate_peaks(axes[1], "ch2_raw", "#1E6F5C")
	axes[1].set_title(f"Full FFT - Filtered CH1 vs Raw CH2 - {title}")
	axes[1].set_xlabel("Frequency (Hz)")
	axes[1].set_ylabel("Amplitude")
	axes[1].grid(alpha=0.3)
	axes[1].legend(loc="best")

	fig.tight_layout()
	fig.savefig(output_png, dpi=160)
	plt.close(fig)


def process_file(input_csv: Path, output_csv: Path, output_png: Path) -> None:
	"""Process one CSV file and generate full-spectrum outputs."""
	sample_rate, ch1, ch2 = parse_oscilloscope_csv(input_csv)
	ch1_filtered = lowpass_filter(ch1, cutoff=LOWPASS_CUTOFF_HZ, fs=sample_rate, order=FILTER_ORDER)

	freq_raw, amp_ch1_raw = compute_fft(ch1, sample_rate)
	freq_ch2_raw, amp_ch2_raw = compute_fft(ch2, sample_rate)
	freq_filtered, amp_ch1_filtered = compute_fft(ch1_filtered, sample_rate)

	if not np.array_equal(freq_raw, freq_ch2_raw):
		raise ValueError(f"CH2 FFT frequency axis mismatch in {input_csv}")
	if not np.array_equal(freq_raw, freq_filtered):
		raise ValueError(f"Filtered FFT frequency axis mismatch in {input_csv}")

	freqs, amp_ch1_raw = limit_positive_peaks(freq_raw, amp_ch1_raw)
	_, amp_ch2_raw = limit_positive_peaks(freq_ch2_raw, amp_ch2_raw)
	_, amp_ch1_filtered = limit_positive_peaks(freq_filtered, amp_ch1_filtered)

	peak_summary = {
		"ch1_raw_peak_10hz_freq": find_peak(freqs, amp_ch1_raw, PEAK_WINDOW_HZ, start_from_first_rise=True)[0],
		"ch1_raw_peak_10hz_amp": find_peak(freqs, amp_ch1_raw, PEAK_WINDOW_HZ, start_from_first_rise=True)[1],
		"ch1_raw_resonance_freq": find_peak(freqs, amp_ch1_raw, None, start_from_first_rise=True)[0],
		"ch1_raw_resonance_amp": find_peak(freqs, amp_ch1_raw, None, start_from_first_rise=True)[1],
		"ch2_raw_peak_10hz_freq": find_peak(freqs, amp_ch2_raw, PEAK_WINDOW_HZ)[0],
		"ch2_raw_peak_10hz_amp": find_peak(freqs, amp_ch2_raw, PEAK_WINDOW_HZ)[1],
		"ch2_raw_resonance_freq": find_peak(freqs, amp_ch2_raw, None)[0],
		"ch2_raw_resonance_amp": find_peak(freqs, amp_ch2_raw, None)[1],
		"ch1_filtered_peak_10hz_freq": find_peak(freqs, amp_ch1_filtered, PEAK_WINDOW_HZ, start_from_first_rise=True)[0],
		"ch1_filtered_peak_10hz_amp": find_peak(freqs, amp_ch1_filtered, PEAK_WINDOW_HZ, start_from_first_rise=True)[1],
		"ch1_filtered_resonance_freq": find_peak(freqs, amp_ch1_filtered, None, start_from_first_rise=True)[0],
		"ch1_filtered_resonance_amp": find_peak(freqs, amp_ch1_filtered, None, start_from_first_rise=True)[1],
	}

	save_fft_csv(
		output_csv,
		sample_rate,
		freqs,
		amp_ch1_raw,
		amp_ch2_raw,
		amp_ch1_filtered,
		peak_summary,
	)

	save_fft_plot(
		output_png,
		freqs,
		amp_ch1_raw,
		amp_ch2_raw,
		amp_ch1_filtered,
		peak_summary,
		title=input_csv.name,
	)


def main() -> None:
	root = Path(__file__).resolve().parent
	in_root = root / "Vibration_results"
	out_root = root / "full-fft-results-for-resonance"

	total = 0
	for group in GROUPS:
		in_dir = in_root / group
		out_dir = out_root / group
		plot_dir = out_dir / "plots"
		out_dir.mkdir(parents=True, exist_ok=True)
		plot_dir.mkdir(parents=True, exist_ok=True)

		for input_csv in sorted(in_dir.glob("*.csv")):
			out_csv = out_dir / f"{input_csv.stem}.csv"
			out_png = plot_dir / f"{input_csv.stem}.png"
			try:
				process_file(input_csv, out_csv, out_png)
				total += 1
				print(f"Processed: {input_csv}")
			except Exception as exc:
				print(f"Failed: {input_csv} -> {exc}")

	print(f"Done. Processed {total} file(s).")


if __name__ == "__main__":
	main()
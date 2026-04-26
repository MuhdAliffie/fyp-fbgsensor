#!/usr/bin/env python3
"""Batch low-pass filtering for vibration oscilloscope CSV files.

Reads input files from:
	./Vibration_results/{30-20,40-20,50-20,60-20,70-20}/*.csv

For each file, this script:
1. Parses CH1 and CH2 signals and sampling metadata.
2. Applies a zero-phase Butterworth low-pass filter to CH1.
3. Computes FFT for raw CH1 and filtered CH1.
4. Saves merged result CSV to ./filter-results/<group>/<name>.csv
5. Saves comparison plot to ./filter-results/<group>/plots/<name>.png
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
FFT_MIN_HZ = 0.0
FFT_MAX_HZ = 40.0


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


def band_limit(freqs: np.ndarray, amps: np.ndarray, f_min: float, f_max: float) -> tuple[np.ndarray, np.ndarray]:
	"""Limit FFT arrays to a specified frequency band."""
	mask = (freqs >= f_min) & (freqs <= f_max)
	return freqs[mask], amps[mask]


def save_results_csv(
	out_csv: Path,
	fs: float,
	time_s: np.ndarray,
	ch1_raw: np.ndarray,
	ch2_raw: np.ndarray,
	ch1_filt: np.ndarray,
	fft_freq: np.ndarray,
	fft_ch1_raw: np.ndarray,
	fft_ch2_raw: np.ndarray,
	fft_filt: np.ndarray,
) -> None:
	"""Save time and FFT data in one CSV file."""
	out_csv.parent.mkdir(parents=True, exist_ok=True)

	max_len = max(time_s.size, fft_freq.size)
	with out_csv.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["sample_rate_hz", fs])
		writer.writerow(
			[
				"time_s",
				"ch1_raw_v",
				"ch2_raw_v",
				"ch1_filtered_v",
				"frequency_hz",
				"fft_ch1_raw",
				"fft_ch2_raw",
				"fft_ch1_filtered",
			]
		)

		for i in range(max_len):
			t = time_s[i] if i < time_s.size else ""
			c1 = ch1_raw[i] if i < ch1_raw.size else ""
			c2 = ch2_raw[i] if i < ch2_raw.size else ""
			c1f = ch1_filt[i] if i < ch1_filt.size else ""
			ff = fft_freq[i] if i < fft_freq.size else ""
			f1r = fft_ch1_raw[i] if i < fft_ch1_raw.size else ""
			f2r = fft_ch2_raw[i] if i < fft_ch2_raw.size else ""
			ffilt = fft_filt[i] if i < fft_filt.size else ""
			writer.writerow([t, c1, c2, c1f, ff, f1r, f2r, ffilt])


def save_comparison_plot(
	out_png: Path,
	time_s: np.ndarray,
	ch1_raw: np.ndarray,
	ch1_filt: np.ndarray,
	fft_freq: np.ndarray,
	fft_raw: np.ndarray,
	fft_filt: np.ndarray,
	title: str,
) -> None:
	"""Save before/after comparison in time and frequency domain."""
	out_png.parent.mkdir(parents=True, exist_ok=True)

	fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

	axes[0].plot(time_s, ch1_raw, label="CH1 Raw", color="#2A9D8F", linewidth=1.0)
	axes[0].plot(time_s, ch1_filt, label="CH1 Filtered", color="#E76F51", linewidth=1.1)
	axes[0].set_title(f"Time Domain - {title}")
	axes[0].set_xlabel("Time (s)")
	axes[0].set_ylabel("Voltage (V)")
	axes[0].grid(alpha=0.3)
	axes[0].legend(loc="best")

	axes[1].plot(fft_freq, fft_raw, label="FFT Raw CH1", color="#2A9D8F", linewidth=1.0)
	axes[1].plot(fft_freq, fft_filt, label="FFT Filtered CH1", color="#E76F51", linewidth=1.1)
	axes[1].set_title("Frequency Domain")
	axes[1].set_xlabel("Frequency (Hz)")
	axes[1].set_ylabel("Amplitude")
	axes[1].grid(alpha=0.3)
	axes[1].legend(loc="best")

	fig.tight_layout()
	fig.savefig(out_png, dpi=160)
	plt.close(fig)


def process_file(input_csv: Path, out_csv: Path, out_png: Path) -> None:
	"""Process one CSV file and generate filtered outputs."""
	fs, ch1, ch2 = parse_oscilloscope_csv(input_csv)
	time_s = np.arange(ch1.size, dtype=float) / fs
	ch1_filtered = lowpass_filter(ch1, cutoff=LOWPASS_CUTOFF_HZ, fs=fs, order=FILTER_ORDER)

	freq_raw, amp_ch1_raw = compute_fft(ch1, fs)
	freq_ch2_raw, amp_ch2_raw = compute_fft(ch2, fs)
	freq_filt, amp_filt = compute_fft(ch1_filtered, fs)

	# Both arrays should share the same frequency axis for equal-length signals.
	if not np.array_equal(freq_raw, freq_filt):
		raise ValueError(f"FFT frequency axis mismatch in {input_csv}")
	if not np.array_equal(freq_raw, freq_ch2_raw):
		raise ValueError(f"CH2 FFT frequency axis mismatch in {input_csv}")

	freq_band, amp_ch1_raw_band = band_limit(freq_raw, amp_ch1_raw, FFT_MIN_HZ, FFT_MAX_HZ)
	_, amp_ch2_raw_band = band_limit(freq_ch2_raw, amp_ch2_raw, FFT_MIN_HZ, FFT_MAX_HZ)
	_, amp_filt_band = band_limit(freq_filt, amp_filt, FFT_MIN_HZ, FFT_MAX_HZ)

	save_results_csv(
		out_csv,
		fs,
		time_s,
		ch1,
		ch2,
		ch1_filtered,
		freq_band,
		amp_ch1_raw_band,
		amp_ch2_raw_band,
		amp_filt_band,
	)

	save_comparison_plot(
		out_png,
		time_s,
		ch1,
		ch1_filtered,
		freq_band,
		amp_ch1_raw_band,
		amp_filt_band,
		title=input_csv.name,
	)


def main() -> None:
	root = Path(__file__).resolve().parent
	in_root = root / "Vibration_results"
	out_root = root / "filter-results"

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

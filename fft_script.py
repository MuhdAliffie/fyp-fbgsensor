#!/usr/bin/env python3
"""Batch FFT processing for oscilloscope vibration CSV files.

Reads all CSV files under:
	./Vibration_results/{30-20,40-20,50-20,60-20,70-20}/*.csv

For each file:
1. Parse waveform data (CH1/CH2) and sampling information.
2. Run FFT with scipy.
3. Save FFT results to ./fft_results/<group>/<name>.csv
4. Save FFT plots to ./fft_results/<group>/plots/<name>.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq


GROUPS = ["30-20", "40-20", "50-20", "60-20", "70-20"]
MIN_FREQ_HZ = 0
MAX_FREQ_HZ = 10


def _to_float(value: str) -> float | None:
	"""Convert a string to float, returning None for non-numeric values."""
	if value is None:
		return None
	text = value.strip().strip('"')
	if text == "":
		return None
	try:
		return float(text)
	except ValueError:
		return None


def parse_oscilloscope_csv(csv_path: Path) -> tuple[float, np.ndarray, np.ndarray | None]:
	"""Parse oscilloscope CSV and return sample_rate, CH1, CH2(optional)."""
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

			# Header rows include keys in first column (e.g., SampleRate, HResolution)
			if first:
				key = first
				if key == "SampleRate" and len(row) > 1:
					maybe_rate = _to_float(row[1])
					if maybe_rate is not None and maybe_rate > 0:
						sample_rate = maybe_rate
				elif key == "HResolution" and len(row) > 1:
					maybe_hres = _to_float(row[1])
					if maybe_hres is not None and maybe_hres > 0:
						h_resolution = maybe_hres
				continue

			# Data rows usually have first column empty, then CH1 and CH2 values.
			if len(row) < 2:
				continue

			v1 = _to_float(row[1])
			v2 = _to_float(row[2]) if len(row) > 2 else None

			if v1 is None:
				continue

			ch1.append(v1)
			if v2 is not None:
				ch2.append(v2)

	if sample_rate is None and h_resolution is not None:
		sample_rate = 1.0 / h_resolution

	if sample_rate is None:
		raise ValueError(f"Could not find SampleRate or HResolution in {csv_path}")
	if not ch1:
		raise ValueError(f"No waveform samples found in {csv_path}")

	ch1_arr = np.asarray(ch1, dtype=float)
	ch2_arr = np.asarray(ch2, dtype=float) if ch2 else None
	return sample_rate, ch1_arr, ch2_arr


def compute_fft(signal: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
	"""Compute single-sided FFT magnitude spectrum."""
	n = signal.size
	spectrum = rfft(signal)
	freqs = rfftfreq(n, d=1.0 / sample_rate)
	# Normalize to amplitude spectrum.
	amplitude = np.abs(spectrum) * 2.0 / n
	if amplitude.size > 0:
		amplitude[0] /= 2.0
	return freqs, amplitude


def save_fft_csv(
	output_csv: Path,
	freqs: np.ndarray,
	amp1: np.ndarray,
	amp2: np.ndarray | None,
) -> None:
	"""Save FFT results into CSV."""
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open("w", newline="") as f:
		writer = csv.writer(f)
		if amp2 is None:
			writer.writerow(["frequency_hz", "amplitude_ch1"])
			for fx, a1 in zip(freqs, amp1):
				writer.writerow([fx, a1])
		else:
			writer.writerow(["frequency_hz", "amplitude_ch1", "amplitude_ch2"])
			for fx, a1, a2 in zip(freqs, amp1, amp2):
				writer.writerow([fx, a1, a2])


def save_fft_plot(
	output_png: Path,
	freqs: np.ndarray,
	amp1: np.ndarray,
	amp2: np.ndarray | None,
	title: str,
) -> None:
	"""Save FFT plot as PNG."""
	output_png.parent.mkdir(parents=True, exist_ok=True)

	plt.figure(figsize=(10, 5))
	plt.plot(freqs, amp1, label="CH1", linewidth=1.0)
	if amp2 is not None:
		plt.plot(freqs, amp2, label="CH2", linewidth=1.0)

	plt.title(title)
	plt.xlabel("Frequency (Hz)")
	plt.ylabel("Amplitude")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_png, dpi=150)
	plt.close()


def process_file(input_csv: Path, output_csv: Path, output_png: Path) -> None:
	"""Process one waveform file and save FFT outputs."""
	sample_rate, ch1, ch2 = parse_oscilloscope_csv(input_csv)
	freqs, amp1 = compute_fft(ch1, sample_rate)

	amp2 = None
	if ch2 is not None and ch2.size == ch1.size:
		_, amp2 = compute_fft(ch2, sample_rate)

	# Keep only the requested frequency band.
	band_mask = (freqs >= MIN_FREQ_HZ) & (freqs <= MAX_FREQ_HZ)
	freqs = freqs[band_mask]
	amp1 = amp1[band_mask]
	if amp2 is not None:
		amp2 = amp2[band_mask]

	save_fft_csv(output_csv, freqs, amp1, amp2)
	save_fft_plot(output_png, freqs, amp1, amp2, title=input_csv.name)


def main() -> None:
	root = Path(__file__).resolve().parent
	input_root = root / "Vibration_results"
	output_root = root / "fft_results"

	total = 0
	for group in GROUPS:
		in_dir = input_root / group
		out_dir = output_root / group
		plot_dir = out_dir / "plots"
		out_dir.mkdir(parents=True, exist_ok=True)
		plot_dir.mkdir(parents=True, exist_ok=True)

		for input_csv in sorted(in_dir.glob("*.csv")):
			output_csv = out_dir / f"{input_csv.stem}.csv"
			output_png = plot_dir / f"{input_csv.stem}.png"

			try:
				process_file(input_csv, output_csv, output_png)
				total += 1
				print(f"Processed: {input_csv}")
			except Exception as exc:
				print(f"Failed: {input_csv} -> {exc}")

	print(f"Done. Processed {total} file(s).")


if __name__ == "__main__":
	main()

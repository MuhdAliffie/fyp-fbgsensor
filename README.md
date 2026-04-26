# FYP FBG Sensor Vibration Analysis Dashboard

This project processes oscilloscope vibration CSV files, generates FFT/filter outputs, and visualizes everything in a unified dashboard.

## Project Structure

- `Vibration_results/` : Raw oscilloscope CSV files grouped by test condition.
- `fft_results/` : FFT outputs (0-10 Hz view) from `fft_script.py`.
- `filter-results/` : Filtered signal outputs from `filter-signal.py`.
- `full-fft-results-for-resonance/` : Full-spectrum FFT outputs for resonance analysis from `full-fft-script.py`.
- `main-dashboard.html` : Main dashboard combining all visualizations.

## Requirements

- Python 3.10+
- Packages:
  - `numpy`
  - `scipy`
  - `matplotlib`

If needed:

```bash
pip install numpy scipy matplotlib
```

## Run Analysis Scripts

From project root:

```bash
python3 fft_script.py
python3 filter-signal.py
python3 full-fft-script.py
```

These scripts generate/update CSV and PNG outputs in their respective result folders.

## Open Dashboard Locally

Use a local web server (recommended):

```bash
python3 -m http.server 8000
```

Then open:

- `http://localhost:8000/main-dashboard.html`

Do not open `main-dashboard.html` directly with `file://` because browser fetch restrictions may block CSV loading.

## Full FFT Features in Dashboard

In the **full fft** section, you can:

- Toggle channel visibility (CH1/CH2).
- Set start and end frequency range for x-axis.
- Select peak rank (1st, 2nd, 3rd, etc.).
- View peak summaries in 4 lines with channel labels:
  - CH1 full-spectrum peak
  - CH1 peak within 10 Hz
  - CH2 full-spectrum peak
  - CH2 peak within 10 Hz

## Free Deployment (GitHub Pages)

1. Push this repository to GitHub.
2. Go to **Settings > Pages**.
3. Set:
   - Source: `Deploy from a branch`
   - Branch: `main`
   - Folder: `/(root)`
4. Save and wait for deployment.
5. Share:
   - `https://<your-username>.github.io/<repo-name>/main-dashboard.html`

## Notes

- Keep folder names and relative paths unchanged so dashboard fetch paths continue to work.
- If you regenerate results, commit and push updated output files so the hosted dashboard reflects latest data.

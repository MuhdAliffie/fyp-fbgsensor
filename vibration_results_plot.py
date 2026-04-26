import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET_FOLDERS = ["30-20", "40-20", "50-20", "60-20", "70-20"]
BASE_DIR = Path("Vibration_results")
INTERACTIVE_POINTS_MAX = 6000
DASHBOARD_HTML = Path("view_data_dashboard.html")


def parse_scope_csv(csv_path: Path):
    metadata = {}
    trace_names = ["CH1", "CH2"]

    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            key = row[0].strip().strip('"')
            values = [cell.strip().strip('"') for cell in row[1:] if cell.strip()]
            metadata[key] = values

    header_size = int(float(metadata.get("Header Size", [15])[0]))
    if "TraceName" in metadata and len(metadata["TraceName"]) >= 2:
        trace_names = [name.strip() for name in metadata["TraceName"][:2]]

    h_resolution = float(metadata.get("HResolution", ["1.0"])[0])
    h_offset = float(metadata.get("HOffset", ["0.0"])[0])

    data = pd.read_csv(
        csv_path,
        skiprows=header_size + 1,
        header=None,
        usecols=[1, 2],
        names=["ch1", "ch2"],
    )
    data = data.apply(pd.to_numeric, errors="coerce").dropna()

    ch1 = data["ch1"].to_numpy(dtype=float)
    ch2 = data["ch2"].to_numpy(dtype=float)
    t = h_offset + np.arange(ch1.size) * h_resolution

    return trace_names, t, ch1, ch2


def write_png(output_png: Path, title: str, t: np.ndarray, ch1: np.ndarray, ch2: np.ndarray, trace_names):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, ch2, label=f"{trace_names[1]} (Input)", linewidth=1.0)
    ax.plot(t, ch1, label=f"{trace_names[0]} (Output)", linewidth=1.0, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def _downsample_for_html(t: np.ndarray, y1: np.ndarray, y2: np.ndarray):
    if t.size <= INTERACTIVE_POINTS_MAX:
        return t, y1, y2
    step = int(np.ceil(t.size / INTERACTIVE_POINTS_MAX))
    return t[::step], y1[::step], y2[::step]


def _compute_shift_range(t: np.ndarray) -> float:
        if t.size < 2:
                return 0.001
        shift_range = float((t[-1] - t[0]) * 0.1)
        return max(shift_range, float(t[1] - t[0]) * 10.0)


def _dashboard_entry(csv_path: Path, trace_names, t: np.ndarray, ch1: np.ndarray, ch2: np.ndarray):
        t_ds, ch1_ds, ch2_ds = _downsample_for_html(t, ch1, ch2)
        return {
                "title": f"{csv_path.parent.name} - {csv_path.stem}",
                "trace_names": [trace_names[0], trace_names[1]],
                "x": t_ds.tolist(),
                "ch1": ch1_ds.tolist(),
                "ch2": ch2_ds.tolist(),
                "shift_range": _compute_shift_range(t_ds),
        }


def write_dashboard_html(output_html: Path, dashboard_data):
        html = f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Oscilloscope Interactive Dashboard</title>
    <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
    <style>
        body {{ font-family: Helvetica, Arial, sans-serif; margin: 16px; }}
        .section {{ border: 1px solid #d9d9d9; border-radius: 8px; padding: 12px; margin-bottom: 14px; }}
        .plot {{ width: 100%; height: 36vh; }}
        .row {{ display: flex; gap: 10px; align-items: center; margin: 10px 0; flex-wrap: wrap; }}
        input[type=range] {{ width: min(600px, 88vw); }}
        input[type=number] {{ width: 140px; padding: 4px; }}
        select {{ padding: 4px; min-width: 220px; }}
        code {{ background: #f2f2f2; padding: 2px 6px; border-radius: 4px; }}
    </style>
</head>
<body>
    <h3>Oscilloscope Interactive Dashboard</h3>
    <div class=\"section\">
        <div class=\"row\">
            <label for=\"folderSelect\">Folder:</label>
            <select id=\"folderSelect\"></select>
            <label for=\"fileSelect\">CSV file:</label>
            <select id=\"fileSelect\"></select>
            <code id=\"datasetLabel\"></code>
        </div>
    </div>

    <div class=\"section\">
        <h4 id=\"titleCh1\">CH1 (Output)</h4>
        <div class=\"row\">
            <label for=\"shiftCh1\">Horizontal shift (seconds):</label>
            <input id=\"shiftCh1\" type=\"range\" value=\"0\" />
            <input id=\"shiftCh1Number\" type=\"number\" value=\"0\" />
            <code id=\"shiftCh1Val\">0.000000</code>
        </div>
        <div id=\"plotCh1\" class=\"plot\"></div>
    </div>

    <div class=\"section\">
        <h4 id=\"titleCh2\">CH2 (Input)</h4>
        <div class=\"row\">
            <label for=\"shiftCh2\">Horizontal shift (seconds):</label>
            <input id=\"shiftCh2\" type=\"range\" value=\"0\" />
            <input id=\"shiftCh2Number\" type=\"number\" value=\"0\" />
            <code id=\"shiftCh2Val\">0.000000</code>
        </div>
        <div id=\"plotCh2\" class=\"plot\"></div>
    </div>

    <div class=\"section\">
        <h4>Overlapped View (Both Shifted)</h4>
        <div id=\"plotOverlap\" class=\"plot\"></div>
    </div>

    <script>
        const DATA = {json.dumps(dashboard_data)};

        const folderSelect = document.getElementById('folderSelect');
        const fileSelect = document.getElementById('fileSelect');
        const datasetLabel = document.getElementById('datasetLabel');

        const shiftCh1 = document.getElementById('shiftCh1');
        const shiftCh1Number = document.getElementById('shiftCh1Number');
        const shiftCh1Val = document.getElementById('shiftCh1Val');

        const shiftCh2 = document.getElementById('shiftCh2');
        const shiftCh2Number = document.getElementById('shiftCh2Number');
        const shiftCh2Val = document.getElementById('shiftCh2Val');

        const titleCh1 = document.getElementById('titleCh1');
        const titleCh2 = document.getElementById('titleCh2');

        let currentX = [];
        let currentY1 = [];
        let currentY2 = [];
        let currentTraceNames = ['CH1', 'CH2'];
        let shiftMin = -0.001;
        let shiftMax = 0.001;

        function clampShift(value) {{
            if (!Number.isFinite(value)) return 0;
            return Math.max(shiftMin, Math.min(shiftMax, value));
        }}

        function applyShiftBounds(range) {{
            const step = range / 200;
            shiftMin = -range;
            shiftMax = range;
            for (const el of [shiftCh1, shiftCh1Number, shiftCh2, shiftCh2Number]) {{
                el.min = String(shiftMin);
                el.max = String(shiftMax);
                el.step = String(step);
            }}
        }}

        function currentEntry() {{
            const folder = folderSelect.value;
            const file = fileSelect.value;
            return DATA[folder][file];
        }}

        function updateCh1(dx) {{
            const shift = clampShift(dx);
            const shiftedX = currentX.map(v => v + shift);
            shiftCh1.value = String(shift);
            shiftCh1Number.value = String(shift);
            shiftCh1Val.textContent = shift.toFixed(6);
            Plotly.restyle('plotCh1', {{ x: [shiftedX] }}, [0]);
            Plotly.restyle('plotOverlap', {{ x: [shiftedX] }}, [0]);
        }}

        function updateCh2(dx) {{
            const shift = clampShift(dx);
            const shiftedX = currentX.map(v => v + shift);
            shiftCh2.value = String(shift);
            shiftCh2Number.value = String(shift);
            shiftCh2Val.textContent = shift.toFixed(6);
            Plotly.restyle('plotCh2', {{ x: [shiftedX] }}, [0]);
            Plotly.restyle('plotOverlap', {{ x: [shiftedX] }}, [1]);
        }}

        function renderSelectedDataset(resetShifts = true) {{
            const entry = currentEntry();
            currentX = entry.x;
            currentY1 = entry.ch1;
            currentY2 = entry.ch2;
            currentTraceNames = entry.trace_names;

            datasetLabel.textContent = `${{folderSelect.value}} / ${{fileSelect.value}}`;
            titleCh1.textContent = `${{currentTraceNames[0]}} (Output)`;
            titleCh2.textContent = `${{currentTraceNames[1]}} (Input)`;

            applyShiftBounds(entry.shift_range);

            const xMinBase = Math.min(...currentX);
            const xMaxBase = Math.max(...currentX);
            const xPadding = entry.shift_range * 1.2;
            const xRange = [xMinBase - xPadding, xMaxBase + xPadding];

            const traceCh1 = {{
                x: [...currentX],
                y: currentY1,
                mode: 'lines',
                name: `${{currentTraceNames[0]}} (Output, shifted)`,
                line: {{ width: 1.4, color: '#E76F51' }}
            }};
            const traceCh2 = {{
                x: [...currentX],
                y: currentY2,
                mode: 'lines',
                name: `${{currentTraceNames[1]}} (Input, shifted)`,
                line: {{ width: 1.4, color: '#2A9D8F' }}
            }};
            const baseLayout = {{
                xaxis: {{ title: 'Time (s)', range: xRange }},
                yaxis: {{ title: 'Voltage (V)' }},
                hovermode: 'x unified',
                legend: {{ orientation: 'h' }},
                margin: {{ l: 64, r: 24, t: 40, b: 56 }}
            }};

            Plotly.react('plotCh1', [traceCh1], {{ ...baseLayout, title: `${{currentTraceNames[0]}} (Output)` }}, {{ responsive: true }});
            Plotly.react('plotCh2', [traceCh2], {{ ...baseLayout, title: `${{currentTraceNames[1]}} (Input)` }}, {{ responsive: true }});
            Plotly.react('plotOverlap', [traceCh1, traceCh2], {{ ...baseLayout, title: 'Overlapped View (Both Shifted)' }}, {{ responsive: true }});

            if (resetShifts) {{
                updateCh1(0);
                updateCh2(0);
            }}
        }}

        function populateFolders() {{
            const folders = Object.keys(DATA).sort();
            folderSelect.innerHTML = folders.map(folder => `<option value=\"${{folder}}\">${{folder}}</option>`).join('');
        }}

        function populateFilesAndRender(resetShifts = true) {{
            const folder = folderSelect.value;
            const files = Object.keys(DATA[folder]).sort();
            const prevFile = fileSelect.value;
            fileSelect.innerHTML = files.map(file => `<option value=\"${{file}}\">${{file}}</option>`).join('');
            if (files.includes(prevFile)) {{
                fileSelect.value = prevFile;
            }}
            renderSelectedDataset(resetShifts);
        }}

        folderSelect.addEventListener('change', () => populateFilesAndRender(true));
        fileSelect.addEventListener('change', () => renderSelectedDataset(true));

        shiftCh1.addEventListener('input', () => updateCh1(Number(shiftCh1.value)));
        shiftCh1Number.addEventListener('input', () => updateCh1(Number(shiftCh1Number.value)));
        shiftCh2.addEventListener('input', () => updateCh2(Number(shiftCh2.value)));
        shiftCh2Number.addEventListener('input', () => updateCh2(Number(shiftCh2Number.value)));

        populateFolders();
        populateFilesAndRender(true);
    </script>
</body>
</html>
"""
        output_html.write_text(html, encoding="utf-8")


def main():
    all_csv_files = []
    for folder in TARGET_FOLDERS:
        all_csv_files.extend(sorted((BASE_DIR / folder).glob("*.csv")))

    if not all_csv_files:
        print("No CSV files found in target folders.")
        return

    print(f"Processing {len(all_csv_files)} CSV files...")
    dashboard_data = {}

    for csv_file in all_csv_files:
        try:
            trace_names, t, ch1, ch2 = parse_scope_csv(csv_file)

            output_dir = csv_file.parent / "plots"
            output_dir.mkdir(parents=True, exist_ok=True)

            title = f"{csv_file.parent.name} - {csv_file.stem}"
            output_png = output_dir / f"{csv_file.stem}.png"
            write_png(output_png, title, t, ch1, ch2, trace_names)

            folder_name = csv_file.parent.name
            file_name = csv_file.name
            dashboard_data.setdefault(folder_name, {})[file_name] = _dashboard_entry(
                csv_file,
                trace_names,
                t,
                ch1,
                ch2,
            )

            print(f"OK: {csv_file} -> {output_png.name}")
        except Exception as exc:
            print(f"FAILED: {csv_file} -> {exc}")

    write_dashboard_html(DASHBOARD_HTML, dashboard_data)
    print(f"Dashboard HTML written to: {DASHBOARD_HTML}")


if __name__ == "__main__":
    main()
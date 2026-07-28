# Option 1 — Run Locally on Your Laptop (+ optional public link via ngrok)

No GitHub, no cloud account required for the core steps. Verified end-to-end in a clean virtual
environment right before writing this: `pip install -r requirements.txt` → zero errors → full pipeline run
→ exit code 0, every output file generated correctly.

## What's in this folder
```
main.py               <- runs the full pipeline in a terminal, generates outputs/models/figures/gis_data
app.py                <- Streamlit version: sliders + live map in your browser
flood_gb_10k_1.csv    <- Thor Valley data (must stay next to main.py / app.py)
requirements.txt      <- exact, tested package versions
```

---

## Step 1 — Install Python

You need **Python 3.11 or 3.12**.

Check what you already have:
```bash
python3 --version
```
(Windows: try `python --version` in PowerShell — Windows often uses `python` not `python3`.)

If it's missing or older than 3.11, download from **python.org/downloads** and install. On Windows, make sure
you tick **"Add Python to PATH"** during install.

## Step 2 — Put everything in one folder

Create a folder anywhere, e.g. `Documents/thor_valley_flood`, and put these 4 files in it:
`main.py`, `app.py`, `flood_gb_10k_1.csv`, `requirements.txt`

## Step 3 — Open a terminal in that folder

- **VS Code**: File → Open Folder → select it → open the built-in terminal (`` Ctrl+` `` on Windows/Linux,
  `` Cmd+` `` on Mac).
- **Without VS Code**: on Windows, open the folder in File Explorer, click the address bar, type `cmd`,
  press Enter. On Mac, right-click the folder → "New Terminal at Folder" (or open Terminal app and `cd` into it).

## Step 4 — Create and activate a virtual environment

This keeps these packages separate from anything else on your system.

**Windows (PowerShell or cmd):**
```powershell
python -m venv venv
venv\Scripts\activate
```
**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```
You'll know it worked when you see `(venv)` at the start of your terminal prompt.

> If Windows PowerShell blocks the activate script with a "running scripts is disabled" error, run this once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` then try `venv\Scripts\activate` again.

## Step 5 — Install the packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This takes a few minutes (TensorFlow is the biggest download, ~200-400 MB). You should see no red "ERROR"
lines at the end — warnings are fine, errors are not.

**If `geopandas`, `rasterio`, or `shapely` fail to install** (mainly a Windows issue — they need GDAL/PROJ
system libraries pip can't always fetch): install **Anaconda** or **Miniconda** instead, then run:
```bash
conda install -c conda-forge geopandas rasterio shapely pyproj scikit-image
pip install -r requirements.txt
```
The second `pip install` will skip the ones conda already installed and get the rest.

## Step 6 — Run it (pick one)

### 6a. Plain terminal version
```bash
python main.py
```
Watch it work through each stage in the terminal. Near the end it asks:
```
Rainfall (mm) [default 40]:
Duration (hr) [default 6]:
Date YYYY-MM-DD [default 2024-07-20]:
```
Type your own numbers, or just press **Enter** three times to accept the defaults. It finishes with a
deliverables summary. Note: you may see a harmless `Error in sys.excepthook` message right at the very end
— that's a known cosmetic TensorFlow shutdown quirk and appears *after* everything has already been saved;
it does not affect your results.

**Open your results:** double-click `outputs/Flood_Map.html` — opens straight in your default browser, no
server needed. Everything else (`models/Best_Model.pkl`, `figures/*.png`, `gis_data/*.geojson`, SHAP plots,
`Predictions.csv`) is sitting in the folders next to `main.py`.

### 6b. Streamlit browser version (recommended — nicer, interactive)
```bash
streamlit run app.py
```
Your browser opens automatically to `http://localhost:8501`. First load takes 2-5 minutes (it's running
the whole pipeline once, in the background, with a spinner showing progress) — after that, moving the
rainfall/duration/date sliders and clicking "Predict" is near-instant.

To stop it: go back to the terminal and press `Ctrl+C`.

## Step 7 — (Optional) Get a public shareable link with ngrok

If you want to send someone a link to your **currently-running** Streamlit app (only works while your
laptop + terminal stay open):

1. Go to **ngrok.com** → sign up free → download ngrok for your OS → follow their 1-line setup command to
   add your free auth token (shown right on their dashboard after signup), e.g.:
   ```bash
   ngrok config add-authtoken YOUR_TOKEN_HERE
   ```
2. With `streamlit run app.py` still running in one terminal, open a **second terminal** in the same folder
   and run:
   ```bash
   ngrok http 8501
   ```
3. ngrok prints something like:
   ```
   Forwarding    https://a1b2-203-0-113-5.ngrok-free.app -> http://localhost:8501
   ```
   Share that `https://....ngrok-free.app` URL with anyone — it tunnels straight to your laptop.
4. Close either terminal (or your laptop) and the link stops working — this is a "leave it running"
   solution, not a permanent deployment.

---

## Quick troubleshooting checklist

| Problem | Fix |
|---|---|
| `python3: command not found` | Use `python` instead (common on Windows), or reinstall Python with "Add to PATH" checked |
| `pip install` fails on geopandas/rasterio/shapely | Use conda for just those 3-5 packages (see Step 5) |
| PowerShell won't run `venv\Scripts\activate` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` (then use that port for ngrok too) |
| `flood_gb_10k_1.csv not found` | Make sure the CSV is in the *same folder* as `main.py`/`app.py`, not a subfolder |
| Everything seems to hang for minutes with no output | Normal — model training + terrain algorithms take a few minutes, especially on first Streamlit load |

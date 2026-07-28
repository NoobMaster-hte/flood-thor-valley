# Thor Valley Flood Prediction & Impact Mapping System

Plain-Python version of the notebook — no Jupyter required. Open the folder in VS Code and run it directly,
or deploy the included Streamlit app to Hugging Face Spaces for free.

## Folder contents

```
thor_valley_deploy/
├── main.py              <- the entire pipeline (data cleaning -> EDA -> 5 AI models -> SHAP ->
│                            GIS acquisition -> terrain/hydrology -> flood model -> impact/evacuation ->
│                            Folium map -> prediction panel -> exports), converted 1:1 from the notebook
├── app.py                <- Streamlit web UI wrapping main.py (used for local browser use AND Hugging Face)
├── flood_gb_10k_1.csv     <- Thor Valley source data
├── requirements.txt       <- pinned, tested package versions
└── README.md              <- this file
```

Running either file creates these output folders next to it: `outputs/`, `models/`, `figures/`, `gis_data/`.

---

## Option A — Run in VS Code (plain script, terminal output + generated files)

1. **Install Python 3.11 or 3.12** if you don't have it (python.org, or `python3 --version` to check).
2. **Open the folder in VS Code**: File → Open Folder → select `thor_valley_deploy`.
3. **Open a terminal in VS Code** (`` Ctrl+` `` / `` Cmd+` ``) and create a virtual environment:
   ```bash
   python3 -m venv venv
   # macOS/Linux:
   source venv/bin/activate
   # Windows (PowerShell):
   venv\Scripts\Activate.ps1
   ```
   VS Code will usually prompt "Select this environment as your workspace interpreter" — click **Yes**.
4. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   > If `geopandas` / `rasterio` / `shapely` fail to install on Windows, run
   > `conda install -c conda-forge geopandas rasterio shapely` first (they need GDAL/PROJ system libraries
   > that plain `pip` sometimes can't resolve on Windows), then re-run the `pip install -r requirements.txt`.
5. **Run it** — either click the ▷ "Run Python File" button in VS Code with `main.py` open, or from the
   terminal:
   ```bash
   python main.py
   ```
   It will print progress for every stage and, at the end, prompt you for a rainfall/duration/date to try
   your own flood scenario (press Enter at each prompt to accept the defaults shown).
6. **View the results:** open `outputs/Flood_Map.html` in any browser (double-click it, or right-click →
   "Open with Live Server" in VS Code). `outputs/`, `models/`, `figures/`, `gis_data/` contain everything
   else (SHAP plots, trained model, GeoJSON/Shapefile, metrics CSV, prediction CSV).

Total run time is a few minutes — model training and the DEM/terrain algorithms are the slow parts.

### Optional: real GIS downloads instead of the synthetic fallback layers
This sandbox had no internet route to SRTM/OpenTopography or OSM Overpass, so the script falls back to
clearly-labelled synthetic-but-realistic layers. On your laptop it will genuinely try the live services
first. For the DEM specifically, get a free key at opentopography.org and set it before running:
```bash
# macOS/Linux
export OPENTOPO_API_KEY="your_key_here"
# Windows PowerShell
$env:OPENTOPO_API_KEY="your_key_here"
```

---

## Option B — Run the Streamlit web app locally (nicer UI, live map + sliders in browser)

Same setup as above (steps 1–4), then instead of `python main.py` run:
```bash
streamlit run app.py
```
It opens `http://localhost:8501` in your browser automatically: a leaderboard of the 5 models, the live
interactive flood map embedded in the page, and rainfall/duration/date sliders that re-run the AI + flood
model on demand.

---

## Option C — Deploy to Hugging Face Spaces (free hosting, shareable link)

Hugging Face Spaces has a **free CPU tier** ("CPU basic": 2 vCPU, 16 GB RAM) — no credit card, no GPU
needed for this project since the dataset and models are small. This is the easiest way to get a public
URL you can share.

1. Go to https://huggingface.co/new-space (create a free account first if you don't have one).
2. Fill in:
   - **Space name**: e.g. `thor-valley-flood-prediction`
   - **SDK**: choose **Streamlit**
   - **Hardware**: leave it on the free **CPU basic** tier
   - **Visibility**: Public or Private, your choice
3. Click **Create Space**. Hugging Face gives you a git repo URL for the Space.
4. Push this folder's contents to that repo:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/thor-valley-flood-prediction
   cd thor-valley-flood-prediction
   cp /path/to/thor_valley_deploy/* .
   git add .
   git commit -m "Initial deploy: Thor Valley flood prediction system"
   git push
   ```
   (Alternatively, use the "Files" tab in the Space's web UI to drag-and-drop upload `main.py`, `app.py`,
   `flood_gb_10k_1.csv`, and `requirements.txt` — no git needed.)
5. Hugging Face automatically detects `requirements.txt`, builds the environment, and launches
   `streamlit run app.py`. The **first build/first load takes a few minutes** (installing TensorFlow etc.,
   then running the full pipeline once) — after that, Streamlit's caching keeps the trained models and map
   in memory so every visitor's page load and prediction is fast.
6. Your public URL will look like: `https://huggingface.co/spaces/<your-username>/thor-valley-flood-prediction`

### Notes on the free tier
- **Cost**: completely free on CPU basic, indefinitely, as long as the Space isn't heavily used (HF puts idle
  Spaces to sleep after a period of inactivity on the free tier; it wakes up again — with a ~1 min cold
  start — the next time someone visits).
- **Storage**: your repo (mainly the ~4.6 MB CSV) is tiny — no issue with HF's free storage limits.
- **No GPU needed**: the LSTM here is small and trains in seconds to low-minutes on CPU; you do not need to
  request GPU hardware (which is paid).
- If you want it to *not* retrain on every restart, you could instead train once locally, commit the
  `models/Best_Model.pkl` file to the repo, and modify `app.py` to load it directly with `pickle.load()`
  instead of re-running `main.py` — worth doing if you outgrow the free tier's cold-start time.

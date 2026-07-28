# Deploying Properly: GitHub + Streamlit Community Cloud

This is the cleanest, most reliable free path: push your project to GitHub, then point Streamlit Cloud at
the repo. No manual file uploads, no local server to keep running, no ngrok.

---

## PART A — Push your project to GitHub

### Step 1: Install Git
Check if you already have it:
```bash
git --version
```
If that fails: download from **git-scm.com/downloads** (Windows) or run `brew install git` (Mac) /
`sudo apt install git` (Ubuntu/Debian).

### Step 2: Tell Git who you are (one-time, ever)
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```
Use the same email you'll use for your GitHub account.

### Step 3: Create your GitHub account + a new empty repo
1. Go to **github.com** → Sign up (free) if you don't have an account.
2. Click the **+** icon top-right → **New repository**.
3. Repository name: e.g. `thor-valley-flood`
4. Keep it **Public** (required for Streamlit Cloud's free tier to read it).
5. **Do NOT** tick "Add a README" / "Add .gitignore" — leave it completely empty.
6. Click **Create repository**. GitHub now shows you a page with commands — ignore it, use the ones below instead (they're the same, just explained).

### Step 4: Get a Personal Access Token (replaces your password for Git operations)
GitHub no longer accepts your account password for `git push` — you need a token instead:
1. Go to **github.com/settings/tokens** → **"Generate new token" → "Generate new token (classic)"**
2. Name it anything, e.g. `laptop-push`
3. Expiration: pick 90 days or "No expiration," your choice
4. Under scopes, tick **`repo`** (this covers everything you need)
5. Click **Generate token** at the bottom
6. **Copy the token immediately and save it somewhere** (a notes file) — GitHub only shows it once.

### Step 5: Initialize Git in your project folder
Open a terminal **in your project folder** (the one with `main.py`, `app.py`, `flood_gb_10k_1.csv`,
`requirements.txt`, `.gitignore`):

```bash
git init
git add .
git commit -m "Initial commit: Thor Valley flood prediction system"
git branch -M main
```

### Step 6: Connect it to your GitHub repo and push
Replace `YOUR-USERNAME` and `thor-valley-flood` with your actual GitHub username and repo name:

```bash
git remote add origin https://github.com/YOUR-USERNAME/thor-valley-flood.git
git push -u origin main
```

When it asks for a **username**: type your GitHub username.
When it asks for a **password**: paste the **token** from Step 4 (not your actual GitHub password — that
will fail). The paste won't show any characters on screen — that's normal, just paste and press Enter.

### Step 7: Verify
Refresh your repo page on github.com — you should see `main.py`, `app.py`, `flood_gb_10k_1.csv`,
`requirements.txt` all listed. (`venv/`, `outputs/`, `models/`, `figures/`, `gis_data/` are correctly
excluded thanks to `.gitignore` — you don't want those in the repo; Streamlit Cloud regenerates them itself.)

### Making changes later
Any time you edit a file and want to update GitHub:
```bash
git add .
git commit -m "describe what you changed"
git push
```
(No need to repeat Steps 5-6 — the token/remote is remembered.)

---

## PART B — Deploy on Streamlit Community Cloud (free, from that repo)

1. Go to **share.streamlit.io**
2. Click **"Sign in with GitHub"** → authorize it (one click, uses your GitHub login, no separate password)
3. Click **"Create app"**
4. Choose:
   - **Repository**: `YOUR-USERNAME/thor-valley-flood`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy**

It builds automatically (reads your `requirements.txt`, installs everything, launches `app.py`). First
build takes several minutes — TensorFlow install + the full pipeline running once. You get a permanent URL like:

```
https://thor-valley-flood-<random-suffix>.streamlit.app
```

Share that with anyone. Every time you `git push` an update, Streamlit Cloud automatically redeploys.

---

## Avoiding common errors

| Symptom | Cause | Fix |
|---|---|---|
| `git push` says "Authentication failed" | Pasted your GitHub password instead of the token | Generate a token (Step 4), use that instead |
| `git push` says "repository not found" | Typo in the remote URL, or repo is Private and token lacks access | Double check the URL matches your repo exactly; make sure repo is Public |
| Streamlit Cloud build fails on `geopandas`/`rasterio` | Some geospatial packages need system libraries beyond pip | Add a `packages.txt` file (see below) — Streamlit Cloud reads it and installs system-level dependencies automatically |
| App builds but crashes on load | Check the "Manage app" logs in the Streamlit Cloud dashboard — usually a missing file (make sure `flood_gb_10k_1.csv` is actually in the repo, not gitignored by accident) |
| Build works locally but not in the cloud | Usually a Python version mismatch — Streamlit Cloud lets you pin the Python version in Advanced Settings when creating the app (choose 3.11 or 3.12) |

### If GeoPandas/Rasterio give the cloud build trouble
Add a file called `packages.txt` (no extension) to your repo root with:
```
libgdal-dev
gdal-bin
```
Streamlit Cloud automatically installs these system packages before your `pip install -r requirements.txt`
step. Just add it to your folder, then:
```bash
git add packages.txt
git commit -m "Add system dependencies for geopandas/rasterio"
git push
```
It'll auto-redeploy.

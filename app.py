"""
Streamlit front-end for the Thor Valley Flood Prediction & Impact Mapping System.

Run locally:      streamlit run app.py
Deploy on HF:      push this whole folder to a Hugging Face "Space" (SDK = Streamlit) — free tier works.

On first load this imports main.py, which runs the ENTIRE pipeline once (data cleaning, model
training, GIS layer generation, terrain/flood modelling, Folium map build). That takes a couple of
minutes on a free CPU instance. Streamlit's cache_resource keeps it in memory after that, so every
prediction you make afterwards is near-instant.
"""
import os
import sys
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Thor Valley Flood Prediction", page_icon="🌊", layout="wide")


@st.cache_resource(show_spinner="Running full pipeline (data cleaning -> model training -> GIS/terrain -> flood model)... first run only, ~2-5 min.")
def load_pipeline():
    import main  # noqa: executes the whole notebook-derived pipeline exactly once, side effects cached
    return main


st.title("🌊 Thor Valley Flood Prediction & Impact Mapping")
st.caption("Gilgit-Baltistan, Pakistan — AI-based early-warning demonstrator (DEM + rainfall + AI models)")

pipeline = load_pipeline()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Model leaderboard")
    st.dataframe(pipeline.results_df, use_container_width=True, hide_index=True)
    st.success(f"Best model selected: **{pipeline.BEST_MODEL_NAME}**  "
               f"(ROC-AUC={pipeline.results_df.iloc[0]['ROC-AUC']:.3f})")

    st.subheader("🔮 Run your own scenario")
    rainfall = st.slider("Rainfall (mm)", 0.0, 150.0, 40.0, 1.0)
    duration = st.slider("Duration (hours)", 1, 48, 6)
    date = st.date_input("Date")

    if st.button("Predict flood scenario", type="primary"):
        with st.spinner("Running AI prediction + terrain flood model..."):
            result = pipeline.predict_user_scenario(rainfall, duration, str(date))
        st.session_state["last_result"] = {k: v for k, v in result.items() if not k.startswith("_")}

    if "last_result" in st.session_state:
        st.subheader("Prediction result")
        for k, v in st.session_state["last_result"].items():
            st.metric(k.replace("_", " ").title(), v) if isinstance(v, (int, float)) else st.write(f"**{k}**: {v}")

with col2:
    st.subheader("Interactive flood map (reference scenario)")
    map_path = pipeline.FLOOD_MAP_PATH
    with open(map_path, "r", encoding="utf-8") as f:
        components.html(f.read(), height=720, scrolling=True)

st.divider()
st.caption(
    "GIS layers (DEM, rivers, buildings, roads, schools, hospitals, bridges, population, LULC, boundary) "
    "are downloaded live where a network route exists, otherwise generated as clearly-labelled synthetic "
    "layers calibrated to this dataset's real Thor Valley attributes. See the console/log output on first "
    "load for the provenance of every layer."
)

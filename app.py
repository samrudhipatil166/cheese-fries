
import os
import subprocess
import streamlit as st

st.set_page_config(page_title="Tesla Location Sales – Model Playground", layout="wide")

st.title("🚗 Tesla Location Sales – Model Playground")
st.write(
    "Use this app to run different **Python training scripts** (different model versions) "
    "and then make predictions for any address using the active model."
)

st.markdown("---")

# ---------- Configuration ----------
# Look for any Python files in the current folder that start with 'train_'
SCRIPT_PATTERN_PREFIX = "train_"
PREDICTION_SCRIPT = "predict_for_address.py"  # shared prediction entrypoint

# ---------- Helper functions ----------
@st.cache_data(show_spinner=False)
def list_training_scripts():
    scripts = []
    for fname in os.listdir("."):
        if fname.startswith(SCRIPT_PATTERN_PREFIX) and fname.endswith(".py"):
            scripts.append(fname)
    return sorted(scripts)

def run_python_script(path):
    """Run a Python script and return (stdout, stderr, returncode)."""
    result = subprocess.run(
        ["python", path],
        capture_output=True,
        text=True
    )
    return result.stdout, result.stderr, result.returncode

def run_prediction(address: str, year: int):
    if not os.path.exists(PREDICTION_SCRIPT):
        raise FileNotFoundError(
            f"Prediction script '{PREDICTION_SCRIPT}' not found in current directory."
        )
    result = subprocess.run(
        ["python", PREDICTION_SCRIPT, "--address", address, "--year", str(year)],
        capture_output=True,
        text=True
    )
    return result.stdout, result.stderr, result.returncode

# ---------- Sidebar: script selection ----------
st.sidebar.header("Model Version")
scripts = list_training_scripts()

if not scripts:
    st.sidebar.error(
        "No training scripts found.\n\n"
        "Place one or more files like `train_model_v1.py`, `train_model_v2.py` "
        "in the same folder as this app."
    )
    chosen_script = None
else:
    chosen_script = st.sidebar.selectbox(
        "Choose a training script (.py):",
        scripts,
        format_func=lambda x: x
    )

    st.sidebar.markdown("**Selected script:**")
    st.sidebar.code(chosen_script, language="bash")

st.markdown("### 1️⃣ Train a model version")

if scripts and chosen_script:
    st.write(
        "Pick a training script in the sidebar, then click **Run training script**. "
        "Each script can have different features, hyperparameters, or logic.\n\n"
        "All scripts are expected to save a model as `tesla_sales_model.joblib` "
        "and metadata as `model_metadata.json` in the current folder."
    )

    if st.button("▶ Run training script", type="primary"):
        with st.spinner(f"Running `{chosen_script}`..."):
            stdout, stderr, code = run_python_script(chosen_script)

        st.markdown("#### Training output")
        st.code(stdout or "(no stdout)", language="bash")

        if stderr:
            st.markdown("#### Training errors / warnings")
            st.code(stderr, language="bash")

        if code == 0:
            if os.path.exists("tesla_sales_model.joblib"):
                st.success("Training completed and `tesla_sales_model.joblib` found.")
            else:
                st.warning(
                    "Training script finished, but `tesla_sales_model.joblib` was not found.\n"
                    "Make sure your training script saves the model with that name."
                )
        else:
            st.error(f"Script exited with non-zero return code: {code}")

st.markdown("---")
st.markdown("### 2️⃣ Predict sales for an address")

col1, col2 = st.columns([2, 1])

with col1:
    address = st.text_input(
        "Address",
        placeholder="e.g. 500 E St Elmo Rd, Austin, TX"
    )
with col2:
    year = st.number_input(
        "Target year",
        min_value=2020,
        max_value=2100,
        value=2026,
        step=1
    )

can_predict = bool(address.strip())

if st.button("📈 Run prediction", disabled=not can_predict):
    if not can_predict:
        st.warning("Please enter an address first.")
    else:
        try:
            with st.spinner("Running prediction via predict_for_address.py..."):
                stdout, stderr, code = run_prediction(address.strip(), int(year))
            st.markdown("#### Prediction output")
            st.code(stdout or "(no stdout)", language="bash")

            if stderr:
                st.markdown("#### Prediction errors / warnings")
                st.code(stderr, language="bash")

            if code != 0:
                st.error(f"Prediction script exited with non-zero return code: {code}")
        except FileNotFoundError as e:
            st.error(str(e))

st.markdown("---")
st.caption(
    "Tip: create multiple training scripts like `train_model_baseline.py`, "
    "`train_model_competitors.py`, `train_model_demographics.py`, etc. "
    "This app lets you switch between them without touching the CLI."
)

import streamlit as st
from typing import Dict, Any

DEFAULT_CONFIG: Dict[str, Any]= {
    "temperature": 0.0,
    "top_k" : 50,
    "max_new_tokens": 80,
    "window_size": 20,
    "z_threshold" : 2.0
}

def init_settings()-> None:
    if "settings" not in st. session_state:
        st.session_state.settings= DEFAULT_CONFIG.copy()

def render_settings() -> None:
    init_settings()

    st.subheader("⚙️ Model & Telemetry Configuration")
    st.caption("Adjust generation sampling and uncertanity quantification thresholds")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Generation Parameters")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.5,
            value=float(st.session_state.settings["temperature"]),
            step= 0.05,
            help= "controls randomness. 0.0 is deterministic greedy search"
        )

        top_k = st.number_input(
            "Top-K Sampling",
            min_value=1,
            max_value=100,
            value=int(st.session_state.settings["top_k"]),
            step=1,
            help="limits sampling pool to top K tokens when temperature>0"
        )

        max_new_tokens = st.slider(
            "Max New Tokens",
            min_value=10,
            max_value=250,
            value=int(st.session_state.settings["max_new_tokens"]),
            step=10,
            help="maximum number of tokens the model will generate per prompt."
        )

    with col2:
        st.markdown("#### Telemetry & Anomaly Bounds")

        window_size = st.slider(
            "Moving Window Size",
            min_value=5,
            max_value=50,
            value=int(st.session_state.settings["window_size"]),
            step=1,
            help="number of recent tokens used to calculate moving mean and std deviation"
        )

        z_threshold = st.slider(
            "Z-Score Anomaly Threshold",
            min_value=1.0,
            max_value=4.0,
            value=float(st.session_state.settings["z_threshold"]),
            step=0.1,
            help="spikes above this value trigger red uncertanity warnings"
        )

    st.divider()

    btn_col1, btn_col2, __ = st.columns([1,1,3])

    with btn_col1:
        if st.button("Save Settings", type="primary", use_container_width=True):
            st.session_state.settings["temperature"]=temperature
            st.session_state.settings["top_k"]=top_k
            st.session_state.settings["max_new_tokens"]=max_new_tokens
            st.session_state.settings["window_size"]= window_size
            st.session_state.settings["z_threshold"]= z_threshold
            st.success("Settings updated successfully")

    with btn_col2:
        if st.button("Reset to Defaults", use_container_width=True):
            st.session_state.settings = DEFAULT_CONFIG.copy()
            st.info("Reset to factory defaults")
            st.rerun()
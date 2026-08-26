import re
import time
import streamlit as st
from auth import authenticate_user, register_user, delete_user
from settings import init_settings, render_settings
from model_loader import load_model_and_tokenizer
from generator import generate_stream
from visuals import create_candidate_distribution_plot

st.set_page_config(
    page_title="LLM ENTROPY ANALYZER",
    page_icon="🔮",
    layout="wide"
)

SESSION_TIMEOUT_SECONDS = 1800  # 30 minutes inactivity timeout

USER_AVATAR = ":material/person:"
BOT_AVATAR = ":material/smart_toy:"

@st.cache_resource(show_spinner="Loading model into memory...")
def get_cached_model():
    return load_model_and_tokenizer()

# initialize session variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "current_page" not in st.session_state:
    st.session_state.current_page = "dashboard"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_telemetry" not in st.session_state:
    st.session_state.latest_telemetry = []
if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()

init_settings()
EMAIL_REGEX = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

def is_valid_email(email: str) -> bool:
    return bool(re.match(EMAIL_REGEX, email.strip()))

def clear_session():
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.session_state.messages = []
    st.session_state.latest_telemetry = []
    st.session_state.last_activity = time.time()

def check_session_timeout():
    if st.session_state.logged_in:
        elapsed = time.time() - st.session_state.last_activity
        if elapsed > SESSION_TIMEOUT_SECONDS:
            clear_session()
            st.warning("⏱Session expired due to inactivity. Please log in again.")
            st.rerun()
        else:
            st.session_state.last_activity = time.time()

def login_gate():
    st.title("LLM Entropy Analyzer")
    st.caption("real-time uncertainty quantification and token stream monitoring")
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        st.subheader("Account Login")
        login_email = st.text_input("Email", key="login_email_input")
        login_pass = st.text_input("Password", type="password", key="login_pass_input")

        if st.button("Sign In", type="primary", use_container_width=True):
            clean_email = login_email.strip().lower()
            if not clean_email or not login_pass:
                st.warning("Please fill in all fields")
            elif not is_valid_email(clean_email):
                st.error("Please enter a valid email address")
            else:
                is_auth, auth_msg = authenticate_user(clean_email, login_pass)
                if is_auth:
                    st.session_state.logged_in = True
                    st.session_state.user_email = clean_email
                    st.session_state.last_activity = time.time()
                    st.success("Authentication successful")
                    st.rerun()
                else:
                    st.error(auth_msg)

    with tab_register:
        st.subheader("Create New Account")
        reg_email = st.text_input("Email", key="reg_email_input")
        reg_pass = st.text_input("Password", type="password", key="reg_pass_input")
        reg_pass_confirm = st.text_input("Confirm Password", type="password", key="reg_pass_confirm_input")

        if st.button("Register", use_container_width=True):
            clean_reg_email = reg_email.strip().lower()
            if not clean_reg_email or not reg_pass or not reg_pass_confirm:
                st.warning("Please fill in all fields")
            elif not is_valid_email(clean_reg_email):
                st.error("Please enter a valid email address format")
            elif reg_pass != reg_pass_confirm:
                st.error("Passwords do not match")
            elif len(reg_pass) < 6:
                st.warning("Password must be at least 6 characters")
            else:
                success = register_user(clean_reg_email, reg_pass)
                if success:
                    st.success("Account created. You can now log in.")
                else:
                    st.error("Registration failed. Email might already be in use.")

def render_dashboard():
    col_chat, col_stats = st.columns([1.2, 1], gap="medium")

    with col_stats:
        st.markdown("##### Telemetry Monitor")

        top_row = st.columns([1, 1, 1.2])
        metric_h = top_row[0].empty()
        metric_z = top_row[1].empty()
        status_box = top_row[2].empty()

        chart_placeholder = st.empty()
        candidates_placeholder = st.empty()

        if not st.session_state.latest_telemetry:
            metric_h.metric("Entropy (H)", "--")
            metric_z.metric("Z-Score", "--")
            status_box.info("Idle")
            chart_placeholder.caption("Entropy & Z-Score trajectory will appear during generation.")
        else:
            latest = st.session_state.latest_telemetry[-1]
            z_thresh = float(st.session_state.settings["z_threshold"])
            
            metric_h.metric("Entropy (H)", f"{float(latest['entropy']):.2f}")
            metric_z.metric("Z-Score", f"{float(latest['z_score']):.2f}")

            if latest["z_score"] >= z_thresh:
                status_box.error(f"Spike (Z ≥ {z_thresh:.1f})")
            elif latest["z_score"] >= 1.5:
                status_box.warning("Shift (Z ≥ 1.5)")
            else:
                status_box.success("Stable")

            historical_payload = {
                "Entropy (H)": [float(d["entropy"]) for d in st.session_state.latest_telemetry],
                "Z-Score": [float(d["z_score"]) for d in st.session_state.latest_telemetry]
            }
            chart_placeholder.line_chart(historical_payload, height=150)

            if "candidates" in latest and latest["candidates"]:
                candidates_placeholder.plotly_chart(
                    create_candidate_distribution_plot(latest["candidates"]),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

    with col_chat:
        st.markdown("##### Live Model Stream")
        chat_box = st.container(height=420)

        with chat_box:
            if not st.session_state.messages:
                with st.chat_message("assistant", avatar=BOT_AVATAR):
                    st.markdown("Enter a prompt")
            else:
                for msg in st.session_state.messages:
                    avatar_icon = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
                    with st.chat_message(msg["role"], avatar=avatar_icon):
                        st.markdown(msg["content"])

        prompt = st.chat_input("Ask a question to monitor uncertainty:")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with chat_box:
                with st.chat_message("user", avatar=USER_AVATAR):
                    st.markdown(prompt)

            model, tokenizer = get_cached_model()
            config = st.session_state.settings
            z_thresh = float(config["z_threshold"])

            with chat_box:
                with st.chat_message("assistant", avatar=BOT_AVATAR):
                    text_placeholder = st.empty()
                    full_response = ""
                    current_stream_data = []

                    stream = generate_stream(
                        prompt=prompt,
                        max_new_tokens=int(config["max_new_tokens"]),
                        model=model,
                        tokenizer=tokenizer,
                        temperature=float(config["temperature"]),
                        top_k=int(config["top_k"]),
                        window_size=int(config["window_size"])
                    )

                    for packet in stream:
                        full_response += packet["token"]
                        current_stream_data.append(packet)
                        text_placeholder.markdown(full_response + "▌")

                        metric_h.metric("Entropy (H)", f"{float(packet['entropy']):.2f}")
                        metric_z.metric("Z-Score", f"{float(packet['z_score']):.2f}")

                        if packet["z_score"] >= z_thresh:
                            status_box.error(f"Spike (Z ≥ {z_thresh:.1f})")
                        elif packet["z_score"] >= 1.5:
                            status_box.warning("Shift (Z ≥ 1.5)")
                        else:
                            status_box.success("Stable")

                        chart_payload = {
                            "Entropy (H)": [float(d["entropy"]) for d in current_stream_data],
                            "Z-Score": [float(d["z_score"]) for d in current_stream_data]
                        }
                        chart_placeholder.line_chart(chart_payload, height=150)

                        if "candidates" in packet and packet["candidates"]:
                            candidates_placeholder.plotly_chart(
                                create_candidate_distribution_plot(packet["candidates"]),
                                use_container_width=True,
                                config={"displayModeBar": False},
                                key=f"cand_live_{len(current_stream_data)}"
                            )

                    text_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.session_state.latest_telemetry = current_stream_data

def render_app():
    check_session_timeout()

    st.sidebar.title("Telemetry Engine")
    st.sidebar.write(f"Active User: **{st.session_state.user_email}**")

    st.sidebar.markdown("### Menu")
    
    col_dash = st.sidebar.button(
        "Dashboard", 
        use_container_width=True, 
        type="primary" if st.session_state.current_page == "dashboard" else "secondary"
    )
    if col_dash:
        st.session_state.current_page = "dashboard"
        st.rerun()

    col_sett = st.sidebar.button(
        "Settings", 
        use_container_width=True, 
        type="primary" if st.session_state.current_page == "settings" else "secondary"
    )
    if col_sett:
        st.session_state.current_page = "settings"
        st.rerun()

    st.sidebar.markdown("---")

    if st.sidebar.button("Logout", use_container_width=True):
        clear_session()
        st.rerun()

    with st.sidebar.expander("Account Settings"):
        st.markdown("##### Danger Zone")
        confirm_pass = st.text_input("Confirm password to delete", type="password", key="delete_pass_input")
        if st.button("Delete Account", type="primary", use_container_width=True):
            if not confirm_pass:
                st.warning("Enter your password to confirm.")
            else:
                if delete_user(st.session_state.user_email, confirm_pass):
                    clear_session()
                    st.success("Account deleted successfully.")
                    st.rerun()
                else:
                    st.error("Incorrect password. Deletion rejected.")

    if st.session_state.current_page == "dashboard":
        render_dashboard()
    elif st.session_state.current_page == "settings":
        render_settings()

if not st.session_state.logged_in:
    login_gate()
else:
    render_app()
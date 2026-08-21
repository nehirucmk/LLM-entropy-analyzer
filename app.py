import re
import streamlit as st
from auth import authenticate_user, register_user, delete_user
from settings import init_settings, render_settings
from model_loader import load_model_and_tokenizer
from generator import generate_stream

st.set_page_config(
    page_title="LLM ENTROPY ANALYZER",
    page_icon="🔮",
    layout="wide"
)

# cache model in ram to prevent reloading on each step
@st.cache_resource(show_spinner="loading model into memory...")
def get_cached_model():
    return load_model_and_tokenizer()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "current_page" not in st.session_state:
    st.session_state.current_page = "dashboard"
if "messages" not in st.session_state:
    st.session_state.messages =[]
if "latest_telemetry" not in st.session_state:
    st.session_state.latest_telemetry=[]

init_settings()
EMAIL_REGEX = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

def is_valid_email(email: str) -> bool:
    # validate email format using standard regex pattern
    return bool(re.match(EMAIL_REGEX, email.strip()))

def clear_session():
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.session_state.messages =[]
    st.session_state.latest_telemetry=[]

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
                if authenticate_user(clean_email, login_pass):
                    st.session_state.logged_in = True
                    st.session_state.user_email = clean_email
                    st.success("Authentication successful")
                    st.rerun()
                else:
                    st.error("Invalid email or password")

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
    col_chat, col_stats = st.columns([2,1], gap="large")

    with col_chat:
        st.subheader("Live Model Stream")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask a question to monitor uncertainty:")
        if prompt:
            # display and save user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            model, tokenizer = get_cached_model()
            config = st.session_state.settings

            with st.chat_message("assistant"):
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

                text_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.latest_telemetry = current_stream_data

        with col_stats:
            st.subheader("Telemetry Monitor")
            if not st.session_state.latest_telemetry:
                st.info("will be added")
            else:
                latest = st.session_state.latest_telemetry[-1]
                z_thresh = st.session_state.settings["z_threshold"]

                m1,m2 = st.columns(2)
                m1.metric("Current Entropy (H)",f"{latest['entropy']: .2f}" )
                m2.metric("Moving Z-Score", f"{latest['z_score']:.2f}")

                if latest["z_score"] >= z_thresh:
                    st.error(f"🔴 High Uncertainty Spike Detected (Z >= {z_thresh})")
                elif latest["z_score"] >= 1.5:
                    st.warning("🟡 Moderate Entropy Shift")
                else:
                    st.success("🟢 Stable Token Generation")
def render_app():
    st.sidebar.title("Telemetry Engine")
    st.sidebar.write(f"Active User: **{st.session_state.user_email}**")

    # native button navigation with active state highlighting
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

    # account settings and secure account deletion
    with st.sidebar.expander("⚙️ Account Settings"):
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
import re
import streamlit as st
from auth import authenticate_user, register_user, delete_user
from settings import init_settings, render_settings

st.set_page_config(
    page_title="LLM ENTROPY ANALYZER",
    page_icon="🔮",
    layout="wide"
)

# initialize session variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

EMAIL_REGEX = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

def is_valid_email(email: str) -> bool:
    # validate email format using standard regex pattern
    return bool(re.match(EMAIL_REGEX, email.strip()))

def clear_session():
    # securely purge session variables on logout or deletion
    st.session_state.logged_in = False
    st.session_state.user_email = ""

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

def render_app():
    st.sidebar.title("Telemetry Engine")
    st.sidebar.write(f"Active User: **{st.session_state.user_email}**")

    navigation = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Settings"],
        index=0
    )

    if st.sidebar.button("Logout", use_container_width=True):
        clear_session()
        st.rerun()

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
    if navigation== "Dashboard":
        st.title("Telemetry Dashboard")
    elif navigation == "Settings":
        render_settings()
if not st.session_state.logged_in:
    login_gate()
else:
    render_app()
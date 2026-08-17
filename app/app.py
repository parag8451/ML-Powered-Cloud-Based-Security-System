from __future__ import annotations

from pathlib import Path
import hashlib
import io
import importlib.util
import os
import sys
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

import json
import gdown

files = {
    "selected_features.pkl": "1DKCp1uFkehbNxdUHhlI9UXLYZEab1YSr",
    "scaler.pkl": "1lJSGZ4LrE9cl6Wdchi-9cIyl4BKNaN7h",
    "rf_model.pkl": "1ODqvk27c_e94Aa_qNczFzAX7VsIlEN91",
    "label_encoder.pkl": "1nUFUMskIG-y20nIXVEXlUyO0fPr6bIsR",
    "final_scaler.pkl": "115F_6DzwhA1DxcA3Bm4TGLp6l6PZxSl3",
    "final_model.pkl": "1BhKdYricdz3zhWLgLuQdMS6dwaJOfqEA",
    "final_label_encoder.pkl": "1eKtu31Ai7Lf-V0FZvdEJb-uKR8oWY-qH"
}

for file_name, file_id in files.items():
    if not os.path.exists(file_name):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, file_name, quiet=False)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.pipeline import load_pipeline
    from src.utils import DATA_DIR, REPORTS_DIR, setup_logging
except ModuleNotFoundError:
    # Fallback for Streamlit launches that do not preserve the repo root on sys.path.
    pipeline_spec = importlib.util.spec_from_file_location(
        "src.pipeline",
        PROJECT_ROOT / "src" / "pipeline.py",
    )
    utils_spec = importlib.util.spec_from_file_location(
        "src.utils",
        PROJECT_ROOT / "src" / "utils.py",
    )
    if pipeline_spec is None or pipeline_spec.loader is None:
        raise
    if utils_spec is None or utils_spec.loader is None:
        raise

    utils_module = importlib.util.module_from_spec(utils_spec)
    sys.modules["src.utils"] = utils_module
    utils_spec.loader.exec_module(utils_module)

    pipeline_module = importlib.util.module_from_spec(pipeline_spec)
    sys.modules["src.pipeline"] = pipeline_module
    pipeline_spec.loader.exec_module(pipeline_module)

    load_pipeline = pipeline_module.load_pipeline
    DATA_DIR = utils_module.DATA_DIR
    REPORTS_DIR = utils_module.REPORTS_DIR
    setup_logging = utils_module.setup_logging


st.set_page_config(
    page_title="ML Powered Cloud-Based Security",
    page_icon="Shield",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = setup_logging("dashboard")
AUTH_STORAGE_KEY = "mlcloud_admin_auth"

CUSTOM_CSS = """
<style>
:root {
    --bg-1: #031427;
    --bg-2: #0d2f57;
    --bg-3: #1d5b96;
    --panel: rgba(255, 255, 255, 0.12);
    --panel-2: rgba(255, 255, 255, 0.08);
    --border: rgba(255, 255, 255, 0.18);
    --text: #f7fbff;
    --muted: #d9efff;
    --accent: #7ed6ff;
}

.stApp {
    color: var(--text);
    font-family: "Times New Roman", serif;
    background:
        radial-gradient(circle at 15% 20%, rgba(126, 214, 255, 0.28), transparent 22%),
        radial-gradient(circle at 88% 12%, rgba(255, 255, 255, 0.22), transparent 16%),
        linear-gradient(135deg, var(--bg-1), var(--bg-2) 52%, var(--bg-3));
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.12), rgba(255,255,255,0.05));
    backdrop-filter: blur(20px);
    border-right: 1px solid var(--border);
}



.hero {
    position: relative;
    overflow: hidden;
    isolation: isolate;
    padding: 34px 34px 26px;
    border-radius: 28px;
    margin-bottom: 18px;
    background: linear-gradient(135deg, rgba(255,255,255,0.18), rgba(255,255,255,0.08));
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 30px 90px rgba(1, 19, 42, 0.34);
}

.metric-card {
    min-height: 132px;
    border-radius: 22px;
    padding: 18px 20px;
    background: linear-gradient(180deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
    border: 1px solid rgba(255,255,255,0.16);
}

.metric-title {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
}

.metric-value {
    margin-top: 10px;
    font-size: 2.45rem;
    font-weight: 700;
    color: white;
}

.metric-sub {
    color: var(--muted);
}

.status-pill {
    display: inline-block;
    padding: 6px 12px;
    margin: 8px 10px 0 0;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.18);
    background: rgba(255,255,255,0.09);
}

.login-shell {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 32px 16px;
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(circle at 20% 18%, rgba(126, 214, 255, 0.18), transparent 24%),
        radial-gradient(circle at 80% 14%, rgba(255, 255, 255, 0.10), transparent 16%),
        linear-gradient(145deg, #071b33 0%, #0d2f57 52%, #163f6b 100%);
    border-radius: 28px;
}

.login-shell::before,
.login-shell::after {
    content: "";
    position: absolute;
    inset: auto;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.38);
    pointer-events: none;
}

.login-shell::before {
    width: 520px;
    height: 520px;
    top: -220px;
    right: -140px;
    background: radial-gradient(circle, rgba(126, 214, 255, 0.18), transparent 72%);
    border: none;
}

.login-shell::after {
    width: 440px;
    height: 440px;
    bottom: -200px;
    left: -120px;
    background: radial-gradient(circle, rgba(255, 255, 255, 0.09), transparent 74%);
    border: none;
}

.login-grid {
    position: absolute;
    inset: 0;
    background:
        linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
    background-size: 36px 36px;
    mask-image: linear-gradient(180deg, rgba(0,0,0,0.72), transparent 92%);
    pointer-events: none;
}

.login-stage {
    width: min(440px, 100%);
    position: relative;
    z-index: 1;
}

.login-card {
    width: min(440px, 100%);
    padding: 34px 32px 28px;
    border-radius: 24px;
    background: linear-gradient(180deg, rgba(8, 24, 45, 0.90), rgba(12, 39, 70, 0.82));
    backdrop-filter: blur(18px);
    border: 1px solid rgba(126, 214, 255, 0.18);
    box-shadow: 0 26px 70px rgba(3, 10, 20, 0.38);
    text-align: center;
}

.login-icon-wrap {
    display: flex;
    justify-content: center;
    margin-bottom: 18px;
}

.login-icon {
    width: 62px;
    height: 62px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(126, 214, 255, 0.22), rgba(126, 214, 255, 0.10));
    color: #dff6ff;
    font-size: 1.15rem;
    font-weight: 700;
    border: 1px solid rgba(126, 214, 255, 0.28);
    box-shadow: 0 12px 26px rgba(4, 18, 36, 0.28);
}

.login-title {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 24px;
    color: #f8fbff;
}

.login-form-wrap div[data-testid="stForm"] {
    border: none;
    padding: 0;
    background: transparent;
}

.login-form-wrap label,
.login-form-wrap [data-testid="stWidgetLabel"] {
    display: none !important;
}

.login-form-wrap [data-baseweb="input"] > div {
    min-height: 46px;
    background: rgba(255,255,255,0.09);
    border-radius: 14px;
    border: 1px solid rgba(126, 214, 255, 0.14);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}

.login-form-wrap input {
    color: #f7fbff !important;
    font-size: 0.95rem;
}

.login-form-wrap input::placeholder {
    color: #8fbad1;
}

.login-form-wrap .stButton > button,
.login-form-wrap .stFormSubmitButton > button {
    height: 3rem;
    border: none;
    border-radius: 14px;
    background: linear-gradient(180deg, #7ed6ff, #48a8d6);
    color: #05203a;
    font-weight: 800;
    letter-spacing: 0.02em;
    box-shadow: 0 14px 28px rgba(72, 168, 214, 0.24);
}

.login-form-wrap .stButton > button:hover,
.login-form-wrap .stFormSubmitButton > button:hover {
    background: linear-gradient(180deg, #96e0ff, #5dbce9);
    color: #041a30;
}

.login-error-wrap {
    margin-top: 14px;
}

.login-hint {
    margin-top: 16px;
    font-size: 0.83rem;
    color: #9bcbe2;
}
.login-form-wrap > div[data-testid="stVerticalBlock"] {
    gap: 0.65rem;
}

.login-form-wrap .stTextInput {
    margin-bottom: 0.2rem;
}

.login-footer {
    margin-top: 10px;
}

.login-accent-line {
    width: 92px;
    height: 4px;
    margin: 0 auto 20px;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(126, 214, 255, 0.18), #7ed6ff, rgba(126, 214, 255, 0.18));
}

.cinematic-frame {
    position: relative;
    overflow: hidden;
    isolation: isolate;
}

.cinematic-frame::before {
    content: "";
    position: absolute;
    inset: 0;
    z-index: -1;
    pointer-events: none;
    background:
        linear-gradient(115deg, transparent 0%, rgba(126, 214, 255, 0.08) 39%, transparent 58%),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.018) 0 1px, transparent 1px 72px);
    opacity: 0.78;
    transform: translateX(-18%);
    animation: cinematicSweep 14s ease-in-out infinite;
}

.cinematic-frame::after {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background: linear-gradient(180deg, rgba(255,255,255,0.035), transparent 24%, transparent 76%, rgba(0,0,0,0.16));
    mix-blend-mode: screen;
}

@keyframes cinematicSweep {
    0%, 100% { transform: translateX(-22%); opacity: 0.4; }
    50% { transform: translateX(22%); opacity: 0.9; }
}

@keyframes signalPulse {
    0%, 100% { transform: scale(0.82); opacity: 0.32; }
    50% { transform: scale(1.12); opacity: 1; }
}

@keyframes orbitDrift {
    from { transform: rotate(0deg) translateX(2px) rotate(0deg); }
    to { transform: rotate(360deg) translateX(2px) rotate(-360deg); }
}

.cinematic-orbit {
    position: absolute;
    width: 210px;
    height: 210px;
    right: 5%;
    top: 14%;
    border: 1px solid rgba(126, 214, 255, 0.18);
    border-radius: 50%;
    transform: rotate(-22deg) skewX(-12deg);
    pointer-events: none;
    opacity: 0.82;
}

.cinematic-orbit::before,
.cinematic-orbit::after {
    content: "";
    position: absolute;
    inset: 21px;
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 50%;
}

.cinematic-orbit::after {
    inset: 48%;
    border: 0;
    width: 7px;
    height: 7px;
    background: #a8e8ff;
    box-shadow: 0 0 18px 5px rgba(126,214,255,0.42);
    animation: signalPulse 3.2s ease-in-out infinite;
}

.signal-trace {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 18px;
    color: rgba(223, 246, 255, 0.78);
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.69rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.signal-trace::before {
    content: "";
    display: inline-block;
    width: 44px;
    height: 1px;
    background: linear-gradient(90deg, transparent, #7ed6ff);
}

.signal-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    margin-right: 6px;
    border-radius: 50%;
    background: #86e5b3;
    box-shadow: 0 0 12px rgba(134,229,179,0.82);
    animation: signalPulse 2.4s ease-in-out infinite;
}

.hero-readout {
    display: flex;
    justify-content: space-between;
    gap: 18px;
    margin-top: 22px;
    padding-top: 14px;
    border-top: 1px solid rgba(255,255,255,0.14);
    color: rgba(223,246,255,0.68);
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.hero-readout strong {
    color: #f7fbff;
    font-weight: 500;
}

.glass-card {
    position: relative;
    overflow: hidden;
    padding: 22px;
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(255,255,255,0.13), rgba(255,255,255,0.045));
    box-shadow: 0 24px 70px rgba(1, 19, 42, 0.24), inset 0 1px 0 rgba(255,255,255,0.08);
}

.glass-card::before {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    top: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(168,232,255,0.72), transparent);
    opacity: 0.72;
}

.glass-card h2,
.glass-card h3 {
    letter-spacing: -0.02em;
}

.metric-card {
    position: relative;
    overflow: hidden;
    transition: transform 260ms ease, border-color 260ms ease, box-shadow 260ms ease;
}

.metric-card::after {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    left: -40%;
    width: 24%;
    background: linear-gradient(90deg, transparent, rgba(168,232,255,0.28), transparent);
    transform: skewX(-18deg);
    transition: left 650ms ease;
    pointer-events: none;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(168,232,255,0.42);
    box-shadow: 0 18px 42px rgba(1, 19, 42, 0.28), inset 0 1px 0 rgba(255,255,255,0.12);
}

.metric-card:hover::after {
    left: 130%;
}

.signal-rail {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 1px;
    margin: 0 0 24px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.10);
}

.signal-rail__node {
    min-height: 74px;
    padding: 14px 16px;
    background: rgba(3,20,39,0.34);
    transition: background 220ms ease, transform 220ms ease;
}

.signal-rail__node:hover {
    background: rgba(126,214,255,0.10);
    transform: translateY(-2px);
}

.signal-rail__label {
    display: block;
    color: rgba(223,246,255,0.58);
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.signal-rail__value {
    display: block;
    margin-top: 7px;
    color: #f7fbff;
    font-size: 0.9rem;
    font-weight: 600;
}

@media (max-width: 760px) {
    .signal-rail {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .hero-readout {
        flex-direction: column;
        gap: 8px;
    }
}

@media (prefers-reduced-motion: reduce) {
    .cinematic-frame::before,
    .cinematic-orbit::after,
    .signal-dot {
        animation: none !important;
    }
}

@media (max-width: 640px) {
    .login-shell {
        min-height: 96vh;
        padding: 16px 10px;
        border-radius: 20px;
    }

    .login-card {
        padding: 28px 20px 22px;
    }

    .login-title {
        font-size: 1.55rem;
    }
}

.login-form-wrap .stButton > button,
.login-form-wrap .stFormSubmitButton > button {
    transition: all 0.18s ease;
}

.login-form-wrap .stButton > button:focus,
.login-form-wrap .stFormSubmitButton > button:focus {
    box-shadow: 0 0 0 0.14rem rgba(126, 214, 255, 0.22);
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

AUTH_STORAGE_COMPONENT = st.components.v2.component(
    "auth_storage_sync",
    html="""
    <div id="auth-storage-sync" aria-hidden="true"></div>
    """,
    css="""
    :host {
        display: block;
        width: 0;
        height: 0;
        overflow: hidden;
    }

    #auth-storage-sync {
        width: 0;
        height: 0;
        overflow: hidden;
    }
    """,
    js="""
    export default function ({ data, setStateValue }) {
      const storageKey = data?.storageKey || "mlcloud_admin_auth";
      const action = data?.action || "read";
      const token = data?.token || "";

      let storedToken = "";

      try {
        if (action === "save") {
          window.localStorage.setItem(storageKey, token);
        } else if (action === "clear") {
          window.localStorage.removeItem(storageKey);
        }

        storedToken = window.localStorage.getItem(storageKey) || "";
      } catch (error) {
        storedToken = "";
      }

      setStateValue("stored_token", storedToken);
      setStateValue("ready", true);
    }
    """,
)


@st.cache_resource(show_spinner=False)
def get_pipeline():
    return load_pipeline()


@st.cache_data(show_spinner=False)
def get_reference_dataset() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("*.csv"))
    frames = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = frame.columns.str.strip()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def get_training_report() -> dict | None:
    report_path = REPORTS_DIR / "training_report.json"
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def init_session_state() -> None:
    defaults = {
        "authenticated": False,
        "auth_storage_action": "read",
        "auth_storage_ready": False,
        "auth_storage_token": "",
        "live_index": 0,
        "live_running": False,
        "live_log": [],
        "live_history_rows": [],
        "last_result": None,
        "last_input_row": None,
        "stream_limit": 180,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_admin_credentials() -> tuple[str, str]:
    secret_username = None
    secret_password = None

    try:
        secret_username = st.secrets.get("ADMIN_USERNAME")
        secret_password = st.secrets.get("ADMIN_PASSWORD")
    except StreamlitSecretNotFoundError:
        pass

    username = secret_username or os.getenv("ADMIN_USERNAME", "admin")
    password = secret_password or os.getenv("ADMIN_PASSWORD", "admin123")
    return username, password


def build_auth_token(username: str, password: str) -> str:
    token_source = f"{username}:{password}:{PROJECT_ROOT.name}:v1"
    return hashlib.sha256(token_source.encode("utf-8")).hexdigest()


def sync_persistent_auth(expected_token: str) -> bool:
    action = st.session_state.get("auth_storage_action", "read")
    prior_token = st.session_state.get("auth_storage_token", "")
    prior_ready = st.session_state.get("auth_storage_ready", False)

    result = AUTH_STORAGE_COMPONENT(
        key="auth_storage_sync",
        data={
            "storageKey": AUTH_STORAGE_KEY,
            "action": action,
            "token": expected_token if action == "save" else "",
        },
        default={
            "stored_token": prior_token,
            "ready": prior_ready,
        },
        on_stored_token_change=lambda: None,
        on_ready_change=lambda: None,
        width=1,
        height=1,
    )

    stored_token = getattr(result, "stored_token", prior_token) or ""
    ready = bool(getattr(result, "ready", prior_ready))

    st.session_state.auth_storage_token = stored_token
    st.session_state.auth_storage_ready = ready

    if action in {"save", "clear"}:
        st.session_state.auth_storage_action = "read"

    if stored_token == expected_token:
        st.session_state.authenticated = True
    elif ready and action != "save":
        st.session_state.authenticated = False

    return ready


def render_login_page() -> None:
    admin_username, admin_password = get_admin_credentials()
    expected_token = build_auth_token(admin_username, admin_password)

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none;}
        section.main > div {padding-top: 0.5rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    shell_col = st.columns([1.15, 1.5, 1.15])[1]

    with shell_col:
        st.markdown(
            """
            <div class="login-shell">
                <div class="login-grid"></div>
                <div class="login-stage">
                    <div class="login-card">
                        <div class="login-icon-wrap">
                            <div class="login-icon">ML</div>
                        </div>
                        <div class="login-accent-line"></div>
                        <div class="login-title">ML Powered Cloud-Based Security</div>
                        <div class="login-form-wrap">
            """,
            unsafe_allow_html=True,
        )

        with st.form("admin_login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        st.markdown(
            """
                            <div class="login-footer"></div>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if submitted:
        if username == admin_username and password == admin_password:
            st.session_state.authenticated = True
            st.session_state.auth_storage_action = "save"
            st.session_state.auth_storage_token = expected_token
            st.session_state.login_error = ""
            st.rerun()
        else:
            st.session_state.login_error = "Invalid admin username or password."

    if st.session_state.get("login_error"):
        st.markdown('<div class="login-error-wrap">', unsafe_allow_html=True)
        st.error(st.session_state.login_error)
        st.markdown("</div>", unsafe_allow_html=True)

def metric_card(title: str, value: str, subtitle: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{subtitle}</div>
    </div>
    """


def build_hero(pipeline) -> None:
    report = get_training_report()
    macro_f1 = pipeline.metadata.get("macro_f1")
    if macro_f1 is None and report:
        macro_f1 = report.get("metadata", {}).get("macro_f1")

    st.markdown(
        """
        <style>
        .glow-title {
            text-align: center;
            font-size: 3.5rem;
            font-weight: bold;
            letter-spacing: 0.04em;
            position: relative;
            color: #fff;
            background: none;
            -webkit-text-fill-color: #fff;
            text-shadow: 0 0 10px rgba(255,255,255,0.45), 0 1px 1px rgba(0,0,0,0.08);
        }
        </style>
        <div class="hero cinematic-frame">
            <div class="cinematic-orbit" aria-hidden="true"></div>
            <div class="signal-trace"><span class="signal-dot"></span>Threat intelligence / live inference layer</div>
            <div class="glow-title">ML Powered Cloud-Based Security System</div>
            <div style="max-width:760px;margin:12px auto 0;text-align:center;color:rgba(223,246,255,0.78);font-size:1.03rem;line-height:1.65;">
                A cinematic command surface for reading network behavior before it becomes an incident.
            </div>
            <div class="hero-readout">
                <span>Pipeline <strong>RANDOM FOREST / MULTI-CLASS</strong></span>
                <span>Signal <strong>ENCRYPTED / READY</strong></span>
                <span>Mode <strong>FLOW ANALYSIS</strong></span>
            </div>
        </div>
        <div class="signal-rail" aria-label="Security system status">
            <div class="signal-rail__node"><span class="signal-rail__label">Input fabric</span><span class="signal-rail__value">Network flow vectors</span></div>
            <div class="signal-rail__node"><span class="signal-rail__label">Decision engine</span><span class="signal-rail__value">35 selected features</span></div>
            <div class="signal-rail__node"><span class="signal-rail__label">Classification</span><span class="signal-rail__value">Multi-class detection</span></div>
            <div class="signal-rail__node"><span class="signal-rail__label">Operator state</span><span class="signal-rail__value"><span class="signal-dot"></span>Awaiting signal</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def append_results_to_log(result_df: pd.DataFrame, source: str, source_rows: pd.DataFrame) -> None:
    for row_index, (_, result_row) in enumerate(result_df.iterrows()):
        sequence = len(st.session_state.live_log) + 1
        source_row = source_rows.iloc[row_index].to_dict()
        entry = {
            "sequence": sequence,
            "source": source,
            "predicted_label": result_row["predicted_label"],
            "confidence": float(result_row["confidence"]),
            "severity": result_row["severity"],
            "severity_score": int(result_row["severity_score"]),
            "top_features": ", ".join(result_row["top_features"]),
            "threat_pattern": result_row["threat_pattern"],
        }
        entry.update({f"prob_{key}": value for key, value in result_row["probabilities"].items()})
        st.session_state.live_log.append(entry)
        st.session_state.live_history_rows.append(source_row)
        st.session_state.last_result = entry
        st.session_state.last_input_row = source_row


def run_prediction(pipeline, df: pd.DataFrame, source: str) -> pd.DataFrame:
    result_df = pipeline.predict_dataframe(df)
    append_results_to_log(result_df, source=source, source_rows=df.reset_index(drop=True))
    return result_df


def render_dashboard(pipeline) -> None:
    # Prefer simulation log from Upload Analysis if available
    if 'simulate_upload_log' in st.session_state and st.session_state.simulate_upload_log:
        log_df = pd.DataFrame(st.session_state.simulate_upload_log)
        # Try to map columns to expected names for dashboard
        if 'prediction' in log_df.columns:
            log_df = log_df.rename(columns={'prediction': 'predicted_label'})
        if 'severity' not in log_df.columns:
            log_df['severity'] = log_df.get('predicted_label', '').map(lambda x: 'High' if x and x != 'BENIGN' else 'Low')
        if 'confidence' not in log_df.columns:
            log_df['confidence'] = 100.0
        if 'severity_score' not in log_df.columns:
            log_df['severity_score'] = log_df['severity'].map(lambda x: 90 if x == 'High' else 10)
        if 'threat_pattern' not in log_df.columns:
            log_df['threat_pattern'] = ''
        latest = log_df.iloc[-1].to_dict() if not log_df.empty else {}
        source = 'simulation'
    else:
        log_df = pd.DataFrame(st.session_state.live_log)
        latest = st.session_state.last_result or {}
        source = 'live'
    total_flows = len(log_df)
    attack_flows = int((log_df["predicted_label"] != "BENIGN").sum()) if not log_df.empty else 0
    attack_rate = f"{(attack_flows / total_flows * 100):.1f}%" if total_flows else "0.0%"

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(metric_card("Model Health", "Ready", "Unified pipeline loaded"), unsafe_allow_html=True)
    col2.markdown(metric_card("Flows Processed", str(total_flows), f"Source: {source}"), unsafe_allow_html=True)
    col3.markdown(metric_card("Threat Detections", str(attack_flows), f"Attack rate {attack_rate}"), unsafe_allow_html=True)
    col4.markdown(metric_card("Latest Severity", latest.get("severity", "Low"), latest.get("predicted_label", "BENIGN")), unsafe_allow_html=True)

    centered = st.columns([1, 2, 1])
    with centered[1]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Attack Distribution")
        if log_df.empty:
            st.info("Distribution will update after predictions are generated.")
        else:
            distribution = log_df["predicted_label"].value_counts().reset_index()
            distribution.columns = ["label", "count"]
            fig = px.pie(
                distribution,
                names="label",
                values="count",
                hole=0.55,
                color_discrete_sequence=px.colors.sequential.Blues_r,
            )
            fig.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(family="Times New Roman", color="black"),
                legend=dict(
                    font=dict(color="black"),
                    bgcolor="white",
                    bordercolor="#e0e0e0",
                    borderwidth=1
                ),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Latest Alerts")
        if log_df.empty:
            st.info("Alerts appear here when attacks are detected.")
        else:
            alerts = log_df[log_df["predicted_label"] != "BENIGN"].tail(8).iloc[::-1]
            if alerts.empty:
                st.success("No attack alerts in the current session.")
            else:
                for _, row in alerts.iterrows():
                    st.markdown(
                        f"""
                        <div style="padding:12px 14px; margin-bottom:10px; border-radius:16px;
                            background:rgba(255,255,255,0.10); border:1px solid rgba(255,255,255,0.16);">
                            <div style="font-size:1.04rem; color:#fff;">{row['predicted_label']} | {row['severity']}</div>
                            <div style="color:#dff7ff;">Confidence {row['confidence']:.2f}% | Severity score {int(row['severity_score'])}</div>
                            <div style="color:#dff7ff;">{row['threat_pattern']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        st.markdown("</div>", unsafe_allow_html=True)


def render_upload_analysis(pipeline) -> None:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Upload Analysis")
    st.write("Upload a traffic CSV. Single-row files work for targeted checks, and multi-row files work for batch analysis.")

    template_df = pd.DataFrame(columns=pipeline.feature_names)
    st.download_button(
        "Download Input Template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="input_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload traffic CSV", type=["csv"], key="upload_csv")
    # Keep uploaded CSV in session state until reset or new upload
    if uploaded:
        st.session_state.uploaded_csv_bytes = uploaded.getvalue()
        st.session_state.uploaded_csv_name = uploaded.name
    if 'uploaded_csv_bytes' in st.session_state:
        try:
            df = pd.read_csv(io.BytesIO(st.session_state.uploaded_csv_bytes))
            # Always run as real-time simulation
            if 'simulate_upload_index' not in st.session_state or st.session_state.get('simulate_upload_reset', False):
                st.session_state.simulate_upload_index = 0
                st.session_state.simulate_upload_log = []
                st.session_state.simulate_upload_reset = False

            st.write(f"""
                <div style='margin-bottom:10px;'>
                <b>Real-Time Simulation: Each row is processed and shown one by one at 2-second intervals.</b><br>
                <span style='color:#7ed6ff;'>Current file: <b>{st.session_state.get('uploaded_csv_name','')}</b></span>
                </div>
            """, unsafe_allow_html=True)

            col_reset, col_remove = st.columns(2)
            if col_reset.button("Reset Simulation", use_container_width=True):
                st.session_state.simulate_upload_index = 0
                st.session_state.simulate_upload_log = []
                st.session_state.simulate_upload_reset = True
                st.rerun()
            if col_remove.button("Remove Uploaded CSV", use_container_width=True):
                del st.session_state['uploaded_csv_bytes']
                if 'uploaded_csv_name' in st.session_state:
                    del st.session_state['uploaded_csv_name']
                st.session_state.simulate_upload_index = 0
                st.session_state.simulate_upload_log = []
                st.session_state.simulate_upload_reset = True
                st.rerun()

            if st.session_state.simulate_upload_index < len(df):
                row = df.iloc[[st.session_state.simulate_upload_index]]
                result_df = pipeline.predict_dataframe(row)
                display_df = pd.concat(
                    [result_df.drop(columns=["probabilities", "schema_validation"], errors="ignore"), pd.DataFrame(result_df["probabilities"].tolist())],
                    axis=1,
                )
                st.session_state.simulate_upload_log.append(display_df.iloc[0].to_dict())
                st.session_state.simulate_upload_index += 1
                # Show all attacks so far in a growing table with serial number
                all_df = pd.DataFrame(st.session_state.simulate_upload_log)
                all_df.insert(0, 'S.No.', range(1, len(all_df) + 1))
                all_df = all_df.iloc[::-1].reset_index(drop=True)  # Latest attack first
                st.dataframe(all_df, use_container_width=True, hide_index=True)
                time.sleep(1.5)
                st.rerun()
            else:
                st.success("Simulation completed.")
                # Show all rows after simulation
                if st.session_state.simulate_upload_log:
                    all_df = pd.DataFrame(st.session_state.simulate_upload_log)
                    all_df.insert(0, 'S.No.', range(1, len(all_df) + 1))
                    all_df = all_df.iloc[::-1].reset_index(drop=True)  # Latest attack first
                    st.dataframe(all_df, use_container_width=True, hide_index=True)

            # Optionally, allow download after simulation
            if st.session_state.simulate_upload_log:
                all_df = pd.DataFrame(st.session_state.simulate_upload_log)
                csv_bytes = all_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Simulation Report",
                    data=csv_bytes,
                    file_name="simulation_report.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            logger.exception("Upload analysis failed")
            st.error(f"Upload analysis failed: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_live_monitor(pipeline) -> None:
    st.subheader("Real-Time Monitoring")
    st.write("This page streams reference traffic through the same inference function used for uploads, with controlled reruns to avoid flicker.")

    col1, col2, col3 = st.columns(3)
    interval = col1.slider("Refresh interval (sec)", 0.5, 3.0, 1.0, 0.1)
    batch_size = 1  # Always process one row at a time for realistic simulation
    st.session_state.stream_limit = col3.slider("Max rows this session", 50, 500, st.session_state.stream_limit, 10)

    start_col, stop_col, reset_col = st.columns(3)
    if start_col.button("Start Stream", use_container_width=True):
        st.session_state.live_running = True
    if stop_col.button("Pause Stream", use_container_width=True):
        st.session_state.live_running = False
    if reset_col.button("Reset Stream", use_container_width=True):
        st.session_state.live_running = False
        st.session_state.live_index = 0
        st.session_state.live_log = []
        st.session_state.live_history_rows = []
        st.session_state.last_result = None
        st.session_state.last_input_row = None
        st.rerun()


    # Use uploaded CSV if present, else use reference dataset
    if 'uploaded_csv_bytes' in st.session_state:
        try:
            uploaded_df = pd.read_csv(io.BytesIO(st.session_state['uploaded_csv_bytes']))
            feature_data = uploaded_df.drop(columns=["Label"], errors="ignore")
        except Exception as e:
            st.warning(f"Failed to read uploaded CSV for simulation: {e}. Using reference dataset.")
            data = get_reference_dataset()
            feature_data = data.drop(columns=["Label"], errors="ignore")
    else:
        data = get_reference_dataset()
        feature_data = data.drop(columns=["Label"], errors="ignore")

    if st.session_state.live_running and st.session_state.live_index < min(len(feature_data), st.session_state.stream_limit):
        start = st.session_state.live_index
        end = min(start + batch_size, st.session_state.stream_limit, len(feature_data))
        batch = feature_data.iloc[start:end].reset_index(drop=True)
        run_prediction(pipeline, batch, source="stream")
        st.session_state.live_index = end
        time.sleep(interval)
        st.rerun()
    elif st.session_state.live_index >= st.session_state.stream_limit:
        st.session_state.live_running = False
        st.success("Streaming session completed.")

    live_df = pd.DataFrame(st.session_state.live_log)
    if live_df.empty:
        st.info("Press Start Stream to begin the simulation.")
    else:
        top_metrics = st.columns(3)
        top_metrics[0].metric("Processed", st.session_state.live_index)
        top_metrics[1].metric("Attack Types Seen", live_df["predicted_label"].nunique())
        top_metrics[2].metric("Critical / High Alerts", int(live_df["severity"].isin(["Critical", "High"]).sum()))

        chart_df = (
            live_df.melt(
                id_vars=["sequence", "predicted_label"],
                value_vars=[column for column in live_df.columns if column.startswith("prob_")],
                var_name="class_name",
                value_name="probability",
            )
            .assign(class_name=lambda frame: frame["class_name"].str.replace("prob_", "", regex=False))
        )
        fig = px.line(
            chart_df,
            x="sequence",
            y="probability",
            color="class_name",
            markers=True,
            color_discrete_sequence=px.colors.sequential.Bluered_r,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Times New Roman", color="#f7fbff"),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show only the latest row (or last 3 for context)
        latest_rows = live_df.tail(3).iloc[::-1]  # Show last 3, newest on top
        st.dataframe(latest_rows, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_threat_intelligence() -> None:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Threat Intelligence")
    intelligence = pd.DataFrame(
        {
            "Threat": ["BENIGN", "Bot", "BruteForce", "DDoS", "DoS", "Infiltration", "PortScan", "WebAttack"],
            "Operational Impact": [
                "Normal traffic",
                "Botnet automation or malware beaconing",
                "Credential stuffing and password guessing",
                "Service exhaustion and availability loss",
                "Targeted service disruption",
                "Deep compromise and persistence risk",
                "Reconnaissance and exposure mapping",
                "Application compromise attempt",
            ],
            "Recommended Action": [
                "Continue monitoring baselines",
                "Inspect command-and-control indicators",
                "Force resets and review access policies",
                "Apply rate limiting and upstream filtering",
                "Inspect hosts and throttle suspicious segments",
                "Escalate to incident response and threat hunting",
                "Tighten exposed services and scan policies",
                "Inspect application logs and patch vulnerable endpoints",
            ],
        }
    )
    st.dataframe(intelligence, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_model_insights(pipeline) -> None:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Model Insights")

    metadata_df = pd.DataFrame([{"Metric": key, "Value": value} for key, value in pipeline.metadata.items()])
    st.dataframe(metadata_df, use_container_width=True, hide_index=True)

    report = get_training_report()
    if report:
        class_report = (
            pd.DataFrame(report["classification_report"])
            .transpose()
            .reset_index()
            .rename(columns={"index": "label"})
        )
        st.write("Per-class validation metrics")
        st.dataframe(class_report, use_container_width=True, hide_index=True)

    source = st.radio("Explainability source", ["Latest prediction", "Upload single-row CSV"], horizontal=True)
    row_df = None
    if source == "Latest prediction" and st.session_state.last_input_row:
        row_df = pd.DataFrame([st.session_state.last_input_row])
    elif source == "Upload single-row CSV":
        uploaded = st.file_uploader("Upload one-row CSV", type=["csv"], key="insight_csv")
        if uploaded:
            candidate = pd.read_csv(uploaded)
            if len(candidate) != 1:
                st.error("Please upload exactly one row for explainability.")
            else:
                row_df = candidate

    if row_df is not None:
        explanation = pipeline.explain_row(row_df)
        if explanation:
            explain_df = pd.DataFrame({"feature": list(explanation.keys()), "importance": list(explanation.values())})
            fig = px.bar(
                explain_df,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Blues",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Times New Roman", color="#f7fbff"),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        prediction = pipeline.predict_dataframe(row_df).iloc[0].to_dict()
        st.json(prediction)
    st.markdown("</div>", unsafe_allow_html=True)


def render_reporting() -> None:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Downloadable Reports")
    live_df = pd.DataFrame(st.session_state.live_log)
    if live_df.empty:
        st.info("Generate predictions first to create a report.")
    else:
        buffer = io.StringIO()
        live_df.to_csv(buffer, index=False)
        st.download_button(
            "Download Session Report",
            data=buffer.getvalue().encode("utf-8"),
            file_name="security_session_report.csv",
            mime="text/csv",
        )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    init_session_state()
    admin_username, admin_password = get_admin_credentials()
    auth_ready = sync_persistent_auth(build_auth_token(admin_username, admin_password))

    if not auth_ready:
        st.markdown(
            """
            <div style="min-height: 60vh; display:flex; align-items:center; justify-content:center; color:#dff6ff;">
                Loading secure session...
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if not st.session_state.authenticated:
        render_login_page()
        return

    pipeline = get_pipeline()
    build_hero(pipeline)

    with st.sidebar:
        st.title("Security Console")
        page = st.radio(
            "Navigation",
            ["Dashboard", "Upload Analysis", "Threat Intelligence", "Model Insights", "Reports"],
        )
        st.markdown("---")
        st.write("Protected classes")
        for label in pipeline.metadata["classes"]:
            st.write(f"- {label}")
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.auth_storage_action = "clear"
            st.session_state.auth_storage_ready = False
            st.session_state.auth_storage_token = ""
            st.session_state.login_error = ""
            st.rerun()

    if page == "Dashboard":
        render_dashboard(pipeline)
    elif page == "Upload Analysis":
        render_upload_analysis(pipeline)
    elif page == "Threat Intelligence":
        render_threat_intelligence()
    elif page == "Model Insights":
        render_model_insights(pipeline)
    else:
        render_reporting()


if __name__ == "__main__":
    main()

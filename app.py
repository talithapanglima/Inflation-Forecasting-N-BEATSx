import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="INFLASI.N-BEATSx — Prediksi Inflasi Indonesia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background: #0F1117; color: #E8EAF0; }

    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
        border: 1px solid #2D3748; border-radius: 16px;
        padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    }
    .main-title {
        font-family: 'Space Mono', monospace; font-size: 1.8rem;
        font-weight: 700; color: #63B3ED; margin: 0;
    }
    .main-subtitle { color: #718096; font-size: 0.9rem; margin-top: 0.3rem; }

    .metric-card {
        background: #1A202C; border: 1px solid #2D3748;
        border-radius: 12px; padding: 1.2rem 1.5rem;
        text-align: center; transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #63B3ED; }
    .metric-label {
        font-size: 0.75rem; color: #718096; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'Space Mono', monospace; font-size: 1.6rem;
        font-weight: 700; color: #63B3ED;
    }
    .metric-sub { font-size: 0.72rem; color: #4A5568; margin-top: 0.2rem; }

    .section-header {
        font-family: 'Space Mono', monospace; font-size: 0.85rem;
        font-weight: 700; color: #63B3ED; text-transform: uppercase;
        letter-spacing: 0.12em; border-bottom: 1px solid #2D3748;
        padding-bottom: 0.5rem; margin-bottom: 1rem;
    }

    .info-box {
        background: #1A202C; border-left: 3px solid #63B3ED;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        font-size: 0.85rem; color: #A0AEC0; margin: 0.5rem 0;
    }
    .warning-box {
        background: #1A202C; border-left: 3px solid #F6AD55;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        font-size: 0.85rem; color: #A0AEC0; margin: 0.5rem 0;
    }
    .success-box {
        background: #1A202C; border-left: 3px solid #68D391;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        font-size: 0.85rem; color: #A0AEC0; margin: 0.5rem 0;
    }
    .error-box {
        background: #1A202C; border-left: 3px solid #FC8181;
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
        font-size: 0.85rem; color: #A0AEC0; margin: 0.5rem 0;
    }

    .pred-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 0.5rem; }
    .pred-table th {
        background: #2D3748; color: #A0AEC0; font-weight: 600;
        font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 0.06em; padding: 0.6rem 0.8rem; text-align: left;
    }
    .pred-table td {
        padding: 0.55rem 0.8rem; border-bottom: 1px solid #1A202C;
        color: #E8EAF0; font-family: 'Space Mono', monospace; font-size: 0.82rem;
    }
    .pred-table tr:nth-child(even) td { background: #1A202C; }
    .pred-table tr:hover td { background: #2D3748; }

    /* Nav menu */
    .nav-btn {
        display: block; width: 100%; padding: 0.6rem 1rem;
        border-radius: 8px; margin-bottom: 4px;
        font-size: 0.85rem; font-weight: 500;
        cursor: pointer; transition: background 0.15s;
        border: none; text-align: left;
        background: transparent; color: #718096;
    }
    .nav-btn.active { background: #2D3748 !important; color: #63B3ED !important; }

    [data-testid="stSidebar"] { background: #0D1117; border-right: 1px solid #2D3748; }

    .stButton > button {
        background: #2B6CB0; color: white; border: none;
        border-radius: 8px; font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600; font-size: 0.875rem; padding: 0.5rem 1.5rem;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #3182CE; }

    [data-testid="stFileUploader"] {
        background: #1A202C; border: 1px dashed #2D3748; border-radius: 12px;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #1A202C; border-radius: 10px; padding: 4px; gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent; color: #718096; border-radius: 7px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.85rem; font-weight: 500;
    }
    .stTabs [aria-selected="true"] { background: #2D3748 !important; color: #63B3ED !important; }

    hr { border-color: #2D3748; }

    /* Home cards */
    .feature-card {
        background: #1A202C; border: 1px solid #2D3748;
        border-radius: 12px; padding: 1.4rem 1.5rem;
        height: 100%; transition: border-color 0.2s;
    }
    .feature-card:hover { border-color: #63B3ED; }
    .feature-icon { font-size: 2rem; margin-bottom: 0.6rem; }
    .feature-title {
        font-family: 'Space Mono', monospace; font-size: 0.9rem;
        font-weight: 700; color: #63B3ED; margin-bottom: 0.4rem;
    }
    .feature-desc { font-size: 0.83rem; color: #718096; line-height: 1.6; }

    /* About card */
    .about-card {
        background: #1A202C; border: 1px solid #2D3748;
        border-radius: 12px; padding: 1.5rem;
    }
    .about-title {
        font-family: 'Space Mono', monospace; font-size: 0.85rem;
        font-weight: 700; color: #63B3ED; margin-bottom: 0.6rem;
        text-transform: uppercase; letter-spacing: 0.1em;
    }
    .about-text { font-size: 0.85rem; color: #A0AEC0; line-height: 1.75; }

    .badge {
        display: inline-block; padding: 2px 10px;
        border-radius: 20px; font-size: 0.72rem; font-weight: 600;
        margin: 2px;
    }
    .badge-blue { background: #1A365D; color: #63B3ED; }
    .badge-green { background: #1C4532; color: #68D391; }
    .badge-yellow { background: #3D2800; color: #F6AD55; }

    .val-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.4rem 0; border-bottom: 1px solid #2D3748;
        font-size: 0.83rem;
    }
    .val-key { color: #718096; }
    .val-val { font-family: 'Space Mono', monospace; color: #E8EAF0; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# ── Load Model & Config ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    from neuralforecast import NeuralForecast
    import torch
    # Patch torch.load untuk kompatibilitas
    _orig = torch.load
    def _patched(*a, **kw):
        kw.setdefault('weights_only', False)
        return _orig(*a, **kw)
    torch.load = _patched

    nf          = NeuralForecast.load('saved_model/nf_final')
    with open('saved_model/scaler_y.pkl',    'rb') as f: scaler_y    = pickle.load(f)
    with open('saved_model/scaler_exog.pkl', 'rb') as f: scaler_exog = pickle.load(f)
    with open('saved_model/best_params.pkl', 'rb') as f: best        = pickle.load(f)
    with open('saved_model/config.json',     'r')  as f: config      = json.load(f)

    # full_data adalah data yang SUDAH di-scale (normalized)
    # digunakan langsung sebagai input model tanpa di-scale lagi
    full_data_scaled = pd.read_parquet('saved_model/full_data.parquet')

    # Buat juga versi raw (inverse transform) untuk keperluan UI
    # agar default nilai eksogen tampil dalam skala asli
    NUM_COLS = ['Harga Minyak Dunia', 'BI Rate', 'Kurs USD/IDR',
                'lag1', 'lag3', 'lag6', 'lag12']
    full_data_raw = full_data_scaled.copy()
    full_data_raw['y'] = scaler_y.inverse_transform(
        full_data_scaled[['y']]).flatten()
    full_data_raw[NUM_COLS] = scaler_exog.inverse_transform(
        full_data_scaled[NUM_COLS])

    return nf, scaler_y, scaler_exog, best, config, full_data_scaled, full_data_raw


# ── Helper Functions ──────────────────────────────────────────────
def make_dummy_col(index_series, month_list):
    return index_series.dt.to_period('M').astype(str).isin(month_list).astype(int)

def build_features(df_raw, config):
    df = df_raw.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds').reset_index(drop=True)
    df['Ramadhan']  = make_dummy_col(df['ds'], config['ramadhan_months'])
    df['Idulfitri'] = make_dummy_col(df['ds'], config['idulfitri_months'])
    df['Natal']     = make_dummy_col(df['ds'], config['natal_months'])
    df['Imlek']     = make_dummy_col(df['ds'], config['imlek_months'])
    df['lag1']      = df['y'].shift(1)
    df['lag3']      = df['y'].shift(3)
    df['lag6']      = df['y'].shift(6)
    df['lag12']     = df['y'].shift(12)
    df = df.dropna().reset_index(drop=True)
    df['unique_id'] = 'inflasi'
    return df

def scale_df(df, scaler_y, scaler_exog, num_cols):
    df = df.copy()
    df['y']      = scaler_y.transform(df[['y']])
    df[num_cols] = scaler_exog.transform(df[num_cols])
    return df

def make_future_dummy(last_date, h, config):
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1), periods=h, freq='MS')
    fd = pd.DataFrame({'ds': future_dates, 'unique_id': 'inflasi'})
    fd['Ramadhan']  = make_dummy_col(fd['ds'], config['ramadhan_months'])
    fd['Idulfitri'] = make_dummy_col(fd['ds'], config['idulfitri_months'])
    fd['Natal']     = make_dummy_col(fd['ds'], config['natal_months'])
    fd['Imlek']     = make_dummy_col(fd['ds'], config['imlek_months'])
    return fd

def set_dark_style():
    plt.rcParams.update({
        'figure.facecolor': '#0F1117', 'axes.facecolor': '#1A202C',
        'axes.edgecolor':  '#2D3748', 'axes.labelcolor': '#A0AEC0',
        'xtick.color': '#718096',     'ytick.color': '#718096',
        'grid.color':  '#2D3748',     'grid.linestyle': '--',
        'grid.alpha':  0.6,           'text.color': '#E8EAF0',
        'font.family': 'monospace',
    })

def validate_columns(df):
    required = {'ds', 'y', 'BI Rate', 'Harga Minyak Dunia', 'Kurs USD/IDR'}
    return required.issubset(set(df.columns))

def normalize_columns(raw):
    col_map = {}
    for c in raw.columns:
        cl = c.lower().strip()
        if cl in ['date', 'ds', 'tanggal']:            col_map[c] = 'ds'
        elif cl in ['inflasi umum', 'inflasi', 'y']:   col_map[c] = 'y'
        elif 'minyak' in cl:                            col_map[c] = 'Harga Minyak Dunia'
        elif 'bi rate' in cl or 'bi_rate' in cl:        col_map[c] = 'BI Rate'
        elif 'kurs' in cl or 'usd' in cl:               col_map[c] = 'Kurs USD/IDR'
    return raw.rename(columns=col_map)


# ── Session State Init ────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'upload_status' not in st.session_state:
    st.session_state.upload_status = None
if 'upload_df_scaled' not in st.session_state:
    st.session_state.upload_df_scaled = None
if 'upload_df_feat' not in st.session_state:
    st.session_state.upload_df_feat = None


# ── Sidebar Navigation ────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem'>
        <div style='font-family:Space Mono,monospace;font-size:1.1rem;
                    font-weight:700;color:#63B3ED;letter-spacing:1px;'>
            INFLASI.N-BEATSx
        </div>
        <div style='font-size:0.72rem;color:#4A5568;margin-top:2px;'>
            N-BEATSx + Bayesian Optimization
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='section-header'>Navigasi</div>",
                unsafe_allow_html=True)

    pages = [
        ('home',        '🏠', 'Home'),
        ('upload',      '📂', 'Upload Data'),
        ('visualisasi', '📊', 'Visualisasi Data'),
        ('prediksi',    '📈', 'Prediksi Inflasi'),
        ('about',       '👥', 'About Us'),
    ]
    for key, icon, label in pages:
        active = st.session_state.page == key
        btn_style = (
            "background:#2D3748;color:#63B3ED;"
            if active else
            "background:transparent;color:#718096;"
        )
        if st.sidebar.button(
            f"{icon}  {label}",
            key=f"nav_{key}",
            use_container_width=True,
        ):
            st.session_state.page = key
            st.rerun()

    st.divider()

    # Status upload
    if st.session_state.uploaded_df is not None:
        df_up = st.session_state.uploaded_df
        st.markdown(f"""
        <div class='success-box'>
            ✅ <b>Data aktif</b><br>
            {len(df_up)} baris · {len(df_up.columns)} kolom<br>
            <span style='font-size:0.75rem;color:#4A5568;'>
                {pd.to_datetime(df_up['ds'].min()).strftime('%b %Y')} –
                {pd.to_datetime(df_up['ds'].max()).strftime('%b %Y')}
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='warning-box'>
            ⚠️ Belum ada data diunggah.<br>
            <span style='font-size:0.75rem;'>
                Menggunakan data bawaan model.
            </span>
        </div>""", unsafe_allow_html=True)

    st.divider()

# ═══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════
def page_home():
    # ── Hero Section ──────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0d1117 0%,#1a1f2e 60%,#0d1117 100%);
                    border:1px solid #2D3748;border-radius:20px;
                    padding:3rem 3rem 2.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden;">
            <div style="position:absolute;top:-60px;right:-60px;width:320px;height:320px;
                        background:radial-gradient(circle,rgba(99,179,237,0.07) 0%,transparent 70%);"></div>
            <div style="position:absolute;bottom:-80px;left:10%;width:280px;height:280px;
                        background:radial-gradient(circle,rgba(104,211,145,0.05) 0%,transparent 70%);"></div>
            <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1rem;">
                <div style="font-size:2.8rem;">📈</div>
                <div>
                    <div style="font-family:Space Mono,monospace;font-size:2rem;
                                font-weight:700;color:#63B3ED;letter-spacing:-0.5px;line-height:1.1;">
                        INFLASI.N-BEATSx
                    </div>
                    <div style="font-size:0.85rem;color:#4A5568;margin-top:4px;
                                font-weight:400;letter-spacing:0.05em;">
                        SISTEM PREDIKSI INFLASI INDONESIA
                    </div>
                </div>
            </div>
            <div style="font-size:1rem;color:#A0AEC0;line-height:1.8;max-width:680px;margin-bottom:2rem;">
                Sistem prediksi inflasi berbasis model
                <b style="color:#63B3ED;">N-BEATSx</b>
                yang dioptimasi dengan
                <b style="color:#68D391;">Bayesian Optimization</b>
                dua tahap. Model mengintegrasikan variabel makroekonomi — BI Rate,
                kurs USD/IDR, dan harga minyak dunia — serta efek kalender hari besar
                keagamaan untuk menghasilkan prediksi inflasi Indonesia hingga
                <b style="color:#F6AD55;">6 bulan ke depan</b>.
            </div>
            <div style="display:flex;gap:0.6rem;flex-wrap:wrap;">
                <span style="background:#1A365D;color:#63B3ED;padding:5px 14px;
                             border-radius:20px;font-size:0.78rem;font-weight:600;">
                    N-BEATSx · Deep Learning
                </span>
                <span style="background:#1C4532;color:#68D391;padding:5px 14px;
                             border-radius:20px;font-size:0.78rem;font-weight:600;">
                    Bayesian Optimization · Optuna
                </span>
                <span style="background:#3D2800;color:#F6AD55;padding:5px 14px;
                             border-radius:20px;font-size:0.78rem;font-weight:600;">
                    Efek Kalender · Variabel Eksogen
                </span>
                <span style="background:#322659;color:#B794F4;padding:5px 14px;
                             border-radius:20px;font-size:0.78rem;font-weight:600;">
                    Jan 2010 – Sep 2025 · 189 Observasi
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── KPI Row ───────────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>Performa Model pada Data Uji</div>",
        unsafe_allow_html=True
    )
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("MAE",     "0.00601", "Mean Absolute Error",     "#63B3ED"),
        ("RMSE",    "0.00834", "Root Mean Squared Error", "#63B3ED"),
        ("SMAPE",   "41.76%",  "Symmetric MAPE",          "#F6AD55"),
        ("Horizon", "6 Bulan", "Prediksi ke depan",       "#68D391"),
        ("Obs.",    "189",     "Data bulanan",             "#B794F4"),
    ]
    for col, (label, val, sub, color) in zip([k1, k2, k3, k4, k5], kpis):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color};font-size:1.35rem;">{val}</div>
                    <div class="metric-sub">{sub}</div>
                </div>""",
                unsafe_allow_html=True
            )
    st.markdown(
        """<div style="font-size:0.75rem;color:#4A5568;margin-top:6px;margin-bottom:1.5rem;">
            * SMAPE dipengaruhi anomali deflasi Februari 2025 (−0.09%).
            Pada bulan normal SMAPE berkisar 6–25%.
        </div>""",
        unsafe_allow_html=True
    )

    # ── Perbandingan Model & Variabel ─────────────────────────────
    col_cmp, col_var = st.columns(2)

    with col_cmp:
        st.markdown(
            "<div class='section-header'>Perbandingan Model</div>",
            unsafe_allow_html=True
        )
        comp = [
            ("N-BEATSx + BO ★", "0.00601", "0.00834", "41.76%", True),
            ("Prophet",          "0.00487", "0.00592", "43.96%", False),
            ("SARIMAX",          "0.00717", "0.00905", "46.40%", False),
            ("N-BEATS",          "0.01039", "0.01223", "62.34%", False),
        ]
        rows_comp = ""
        for model, mae, rmse, smape, is_ours in comp:
            if is_ours:
                std = "color:#63B3ED;font-weight:700;"
                sbg = "background:#0d1a2e;"
            else:
                std = "color:#A0AEC0;"
                sbg = ""
            rows_comp += (
                f"<tr style=\"{sbg}\">"
                f"<td style=\"{std}\">{model}</td>"
                f"<td style=\"{std}\">{mae}</td>"
                f"<td style=\"{std}\">{rmse}</td>"
                f"<td style=\"{std}\">{smape}</td>"
                f"</tr>"
            )
        st.markdown(
            f"""
            <table class="pred-table">
                <tr><th>Model</th><th>MAE ↓</th><th>RMSE ↓</th><th>SMAPE ↓</th></tr>
                {rows_comp}
            </table>
            <div style="font-size:0.73rem;color:#4A5568;margin-top:6px;">
                ↓ = semakin kecil semakin baik &nbsp;·&nbsp; ★ = model yang dikembangkan
            </div>""",
            unsafe_allow_html=True
        )

    with col_var:
        st.markdown(
            "<div class='section-header'>Variabel yang Digunakan</div>",
            unsafe_allow_html=True
        )
        var_groups = [
            ("🎯 Target",          [("Inflasi Bulanan (y)",)], "#63B3ED", "#1A365D"),
            ("📉 Lag Inflasi",     [("lag1",),("lag3",),("lag6",),("lag12",)], "#68D391","#1C4532"),
            ("🌐 Eksogen Historis",[("BI Rate",),("Harga Minyak Dunia",),("Kurs USD/IDR",)],"#F6AD55","#3D2800"),
            ("📅 Dummy Kalender",  [("Ramadan",),("Idulfitri",),("Natal",),("Imlek",)], "#B794F4","#322659"),
        ]
        for group, items, color, bg in var_groups:
            badges = " ".join([
                f'<span style="background:{bg};color:{color};padding:3px 10px;'
                f'border-radius:12px;font-size:0.75rem;font-weight:600;'
                f'display:inline-block;margin:2px;">{item[0]}</span>'
                for item in items
            ])
            st.markdown(
                f"""<div style="margin-bottom:0.8rem;">
                    <div style="font-size:0.73rem;color:#4A5568;font-weight:600;
                                text-transform:uppercase;letter-spacing:0.08em;
                                margin-bottom:5px;">{group}</div>
                    {badges}
                </div>""",
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-header'>Uji Asumsi Residual</div>",
            unsafe_allow_html=True
        )
        residuals = [
            ("Rata-rata Residual",       "ē = −0.004927", "Tidak bias sistematis"),
            ("Normalitas (Shapiro-Wilk)","W=0.9152",      "p=0.2489 · Normal"),
            ("Autokorelasi (Ljung-Box)", "Q=5.2282",      "p=0.5149 · Tidak autokor."),
            ("Homoskedastisitas (BP)",   "LM=8.8344",     "p=0.2648 · Homoskedastis"),
        ]
        for test, stat, desc in residuals:
            st.markdown(
                f"""<div style="background:#1A202C;border:1px solid #2D3748;
                            border-radius:8px;padding:0.55rem 0.9rem;
                            margin-bottom:0.45rem;display:flex;gap:0.6rem;align-items:center;">
                    <div style="font-size:0.95rem;">✅</div>
                    <div style="flex:1;">
                        <div style="font-size:0.77rem;font-weight:600;color:#E8EAF0;">{test}</div>
                        <div style="font-family:Space Mono,monospace;font-size:0.7rem;color:#63B3ED;">
                            {stat} &nbsp;·&nbsp; {desc}
                        </div>
                    </div>
                </div>""",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Alur Penggunaan ───────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>Alur Penggunaan Sistem</div>",
        unsafe_allow_html=True
    )
    sc1, sc2, sc3 = st.columns(3)
    steps = [
        ("01", "📂", "Upload Data",
         "Unggah file CSV/Excel berisi data inflasi, BI Rate, "
         "harga minyak dunia, dan kurs USD/IDR.", "upload"),
        ("02", "📊", "Visualisasi Data",
         "Eksplorasi pola historis dan tren variabel makroekonomi "
         "melalui grafik interaktif.", "visualisasi"),
        ("03", "📈", "Prediksi Inflasi",
         "Lihat prediksi 6 bulan ke depan dalam bentuk "
         "grafik, tabel, dan metrik evaluasi model.", "prediksi"),
    ]
    for col, (num, icon, title, desc, nav_key) in zip([sc1, sc2, sc3], steps):
        with col:
            st.markdown(
                f"""<div class="feature-card" style="position:relative;">
                    <div style="position:absolute;top:1rem;right:1rem;
                                font-family:Space Mono,monospace;
                                font-size:0.7rem;color:#2D3748;font-weight:700;">{num}</div>
                    <div style="font-size:2rem;margin-bottom:0.5rem;">{icon}</div>
                    <div class="feature-title">{title}</div>
                    <div class="feature-desc">{desc}</div>
                </div>""",
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(
                f"Buka {title} →",
                key=f"home_nav_{num}",
                use_container_width=True
            ):
                st.session_state.page = nav_key
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────
    st.markdown(
        """<div style="background:#1A202C;border:1px solid #2D3748;border-radius:12px;
                    padding:1.2rem 1.5rem;display:flex;gap:1rem;align-items:flex-start;">
            <div style="font-size:1.3rem;">ℹ️</div>
            <div>
                <div style="font-size:0.85rem;font-weight:600;color:#E8EAF0;margin-bottom:4px;">
                    Cara Mulai
                </div>
                <div style="font-size:0.82rem;color:#718096;line-height:1.7;">
                    Klik <b style="color:#63B3ED;">Upload Data</b> di sidebar kiri
                    untuk mengunggah data historis terbaru, lalu buka halaman
                    <b style="color:#63B3ED;">Prediksi Inflasi</b> untuk melihat
                    proyeksi 6 bulan ke depan. Jika tidak ada data yang diunggah,
                    sistem menggunakan data bawaan model (Jan 2010 – Sep 2025) secara otomatis.
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True
    )



def decompose_forecast(nf, df_scaled, fut_df, scaler_y):
    """
    Dekomposisi N-BEATSx mengikuti metodologi kode penelitian secara tepat.

    Kode penelitian melatih dua model terpisah (trend-only, seasonality-only)
    dengan hiperparameter identik, lalu menghitung:
        exog = total - trend - season  (dalam ruang scaled)
    kemudian setiap komponen di-inverse_transform secara independen.

    Di sini, pendekatan yang sama direplikasi tanpa melatih ulang model:
    model trend-only dan seasonality-only dibuat dengan menyalin bobot
    blok yang relevan dari model utama ke dalam instance model baru,
    lalu predict dijalankan langsung.
    """
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATSx
        import torch, copy

        model_main = nf.models[0]
        model_main.eval()
        best  = {
            'input_size':  model_main.input_size,
            'hidden_size': model_main.hidden_size,
            'n_blocks':    [len(model_main.blocks) // 2],
            'max_steps':   1,
            'lr':          1e-4,
            'dropout':     0.0,
            'h':           model_main.h,
        }
        # Ambil list eksogen dari model utama
        hist_exog = getattr(model_main, 'hist_exog_list', None) or []
        futr_exog = getattr(model_main, 'futr_exog_list', None) or []

        def make_sub_model(stack_type, blocks_idx):
            """Buat model sub-stack dengan bobot dari model utama."""
            m = NBEATSx(
                h            = best['h'],
                input_size   = best['input_size'],
                stack_types  = [stack_type],
                n_blocks     = [best['n_blocks'][0]],
                mlp_units    = [[model_main.hidden_size,
                                 model_main.hidden_size]],
                learning_rate= best['lr'],
                max_steps    = best['max_steps'],
                dropout_prob_theta = best['dropout'],
                hist_exog_list = hist_exog,
                futr_exog_list = futr_exog,
                scaler_type  = None,
            )
            # Dummy fit agar parameter terinisialisasi
            # (diperlukan sebelum load_state_dict)
            m.eval()
            # Salin bobot dari blok model utama ke blok model sub
            main_blocks = list(model_main.blocks)
            sub_blocks  = [main_blocks[i] for i in blocks_idx
                           if i < len(main_blocks)]
            with torch.no_grad():
                for i, (sb, mb) in enumerate(
                    zip(m.blocks, sub_blocks)
                ):
                    sb.load_state_dict(mb.state_dict())
            return m

        # Blok 0-2 = trend stack, blok 3-5 = seasonality stack
        n_blk    = len(model_main.blocks)
        n_trend  = min(3, n_blk)
        n_season = min(3, n_blk - n_trend)

        # ── Tangkap forecast per blok model utama via hook ────────
        block_fc = []
        def _hook(module, inp, out):
            _, fc = out
            arr = fc.detach().cpu().numpy()
            arr = arr[0] if arr.ndim == 3 else arr
            block_fc.append(arr.flatten()[:best['h']].astype(float))

        hooks = [blk.register_forward_hook(_hook)
                 for blk in model_main.blocks]
        forecast_df = nf.predict(df=df_scaled, futr_df=fut_df)
        for h_ in hooks:
            h_.remove()

        if not block_fc:
            return {"success": False,
                    "error": "Tidak ada output blok yang tertangkap."}

        h_out = best['h']

        # Seragamkan panjang
        blks = []
        for fc in block_fc:
            fc = np.array(fc, dtype=float).flatten()
            fc = fc[:h_out] if len(fc) >= h_out else                  np.pad(fc, (0, h_out - len(fc)))
            blks.append(fc)

        # ── Jumlahkan per stack (scaled) ──────────────────────────
        trend_s  = np.sum([blks[i] for i in range(n_trend)],  axis=0)
        season_s = np.sum(
            [blks[i] for i in range(n_trend, n_trend + n_season)],
            axis=0)
        total_s  = forecast_df[["NBEATSx"]].values.flatten()
        exog_s   = total_s - trend_s - season_s

        # ── Inverse transform per komponen (identik kode penelitian) ──
        def inv(arr_1d):
            return scaler_y.inverse_transform(
                np.array(arr_1d, dtype=float).reshape(-1, 1)
            ).flatten()

        trend_orig  = inv(trend_s)
        season_orig = inv(season_s)
        exog_orig   = inv(exog_s)
        total_orig  = inv(total_s)

        return {
            "trend":       trend_orig,
            "seasonality": season_orig,
            "exogenous":   exog_orig,
            "total":       total_orig,
            "success":     True,
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e) + "\n" + traceback.format_exc()
        }


def render_decomp_tab(decomp, future_dates, label="Prediksi"):
    """
    Menampilkan hasil dekomposisi komponen N-BEATSx:
    grafik garis per komponen, grafik batang proporsi,
    tabel ringkasan, dan tombol unduh CSV.
    """
    if not decomp.get("success"):
        st.warning(
            "⚠️ Dekomposisi tidak dapat ditampilkan: "
            + decomp.get("error", "Terjadi kesalahan tidak diketahui.")
        )
        return

    trend_v  = decomp["trend"]
    season_v = decomp["seasonality"]
    exog_v   = decomp["exogenous"]
    total_v  = decomp["total"]
    dates    = [pd.to_datetime(d) for d in future_dates]
    labels_x = [d.strftime("%b %Y") for d in dates]

    # ── Hitung proporsi ──────────────────────────────────────────
    props = []
    for t, s, e in zip(trend_v, season_v, exog_v):
        denom = abs(t) + abs(s) + abs(e)
        denom = denom if denom > 1e-12 else 1.0
        props.append((abs(t)/denom*100,
                      abs(s)/denom*100,
                      abs(e)/denom*100))

    avg_t = np.mean([p[0] for p in props])
    avg_s = np.mean([p[1] for p in props])
    avg_e = np.mean([p[2] for p in props])

    # ── Kartu ringkasan proporsi ─────────────────────────────────
    dc1, dc2, dc3 = st.columns(3)
    for col, (lbl, val, color, desc) in zip(
        [dc1, dc2, dc3],
        [("Proporsi Trend",       f"{avg_t:.1f}%", "#68D391",
          "Stack Trend (Blok 0–2)"),
         ("Proporsi Seasonality", f"{avg_s:.1f}%", "#F6AD55",
          "Stack Seasonality (Blok 3–5)"),
         ("Proporsi Eksogen",     f"{avg_e:.1f}%", "#63B3ED",
          "BI Rate · Minyak · Kurs · Lag")]
    ):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">{lbl}</div>
                    <div class="metric-value" style="color:{color};">{val}</div>
                    <div class="metric-sub">{desc}</div>
                </div>""",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    set_dark_style()

    # ── Grafik 1: Garis per komponen (3 subplot) ─────────────────
    st.markdown(
        "<div class='section-header'>Grafik Komponen Prediksi</div>",
        unsafe_allow_html=True
    )
    fig1, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig1.suptitle(
        f"Dekomposisi Komponen N-BEATSx — {label}",
        fontsize=10, color="#E8EAF0", fontfamily="monospace", y=1.01
    )
    comp_cfg = [
        (axes[0], trend_v,  "#68D391", "Trend (Stack 0–2)"),
        (axes[1], season_v, "#F6AD55", "Seasonality (Stack 3–5)"),
        (axes[2], exog_v,   "#63B3ED",
         "Eksogen (BI Rate, Harga Minyak, Kurs USD/IDR, Lag Inflasi)"),
    ]
    for ax, vals, color, title in comp_cfg:
        ax.plot(dates, vals * 100, color=color,
                linewidth=2, marker="o", markersize=5)
        ax.fill_between(dates, vals * 100, 0,
                        alpha=0.1, color=color)
        for d, v in zip(dates, vals):
            ax.annotate(
                f"{v*100:.3f}%", xy=(d, v*100),
                xytext=(0, 8), textcoords="offset points",
                fontsize=7, color=color, ha="center",
                fontfamily="monospace"
            )
        ax.set_ylabel("Kontribusi (%)", fontsize=8)
        ax.set_title(title, fontsize=8.5, color=color,
                     pad=4, fontfamily="monospace")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="#4A5568", linewidth=0.8, linestyle="--")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.2f}%"))

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(),
             rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Grafik 2: Batang proporsi per bulan ──────────────────────
    st.markdown(
        "<div class='section-header'>Grafik Batang Proporsi Komponen</div>",
        unsafe_allow_html=True
    )
    fig2, ax2 = plt.subplots(figsize=(11, 4.5))
    x     = np.arange(len(dates))
    width = 0.25

    bar_t = ax2.bar(x - width, [p[0] for p in props],
                    width, label="Trend", color="#68D391", alpha=0.85)
    bar_s = ax2.bar(x,         [p[1] for p in props],
                    width, label="Seasonality", color="#F6AD55", alpha=0.85)
    bar_e = ax2.bar(x + width, [p[2] for p in props],
                    width, label="Eksogen", color="#63B3ED", alpha=0.85)

    for bars, color in [(bar_t, "#68D391"),
                        (bar_s, "#F6AD55"),
                        (bar_e, "#63B3ED")]:
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, color=color, fontfamily="monospace"
            )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_x, rotation=30, ha="right")
    ax2.set_ylabel("Proporsi (%)", fontsize=9)
    ax2.set_ylim(0, 110)
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.legend(fontsize=9, framealpha=0.3,
               facecolor="#1A202C", edgecolor="#2D3748")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_title(
        "Proporsi Kontribusi Komponen per Periode",
        fontsize=10, pad=10, color="#E8EAF0", fontfamily="monospace"
    )
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabel dekomposisi ────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>Tabel Dekomposisi</div>",
        unsafe_allow_html=True
    )
    rows_d = ""
    for i, d in enumerate(dates):
        pr_t, pr_s, pr_e = props[i]
        rows_d += (
            f"<tr>"
            f"<td>{d.strftime('%B %Y')}</td>"
            f"<td style='color:#63B3ED;font-weight:600;'>"
            f"  {total_v[i]*100:.4f}%</td>"
            f"<td style='color:#68D391;'>{trend_v[i]*100:.4f}%</td>"
            f"<td style='color:#F6AD55;'>{season_v[i]*100:.4f}%</td>"
            f"<td style='color:#63B3ED;'>{exog_v[i]*100:.4f}%</td>"
            f"<td style='color:#68D391;'>{pr_t:.1f}%</td>"
            f"<td style='color:#F6AD55;'>{pr_s:.1f}%</td>"
            f"<td style='color:#63B3ED;'>{pr_e:.1f}%</td>"
            f"</tr>"
        )
    st.markdown(
        f"""<table class="pred-table">
            <tr>
                <th>Periode</th><th>Total Prediksi</th>
                <th>Trend</th><th>Seasonality</th><th>Eksogen</th>
                <th>% Trend</th><th>% Season</th><th>% Eksogen</th>
            </tr>
            {rows_d}
        </table>""",
        unsafe_allow_html=True
    )

    # Rata-rata
    st.markdown(
        f"""<div style="background:#1A202C;border:1px solid #2D3748;
                    border-radius:8px;padding:0.75rem 1rem;margin-top:0.75rem;
                    font-size:0.82rem;color:#A0AEC0;line-height:1.8;">
            <b style="color:#E8EAF0;">Rata-rata Kontribusi per Komponen:</b><br>
            Trend: <b style="color:#68D391;">{np.mean(trend_v)*100:.4f}%</b>
            ({avg_t:.1f}%) &nbsp;·&nbsp;
            Seasonality: <b style="color:#F6AD55;">
            {np.mean(season_v)*100:.4f}%</b>
            ({avg_s:.1f}%) &nbsp;·&nbsp;
            Eksogen: <b style="color:#63B3ED;">
            {np.mean(exog_v)*100:.4f}%</b>
            ({avg_e:.1f}%)
        </div>""",
        unsafe_allow_html=True
    )

    # ── Tombol unduh ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    dl_decomp = pd.DataFrame({
        "Periode":              [d.strftime("%Y-%m") for d in dates],
        "Total_Prediksi_%":     [f"{v*100:.4f}" for v in total_v],
        "Trend_%":              [f"{v*100:.4f}" for v in trend_v],
        "Seasonality_%":        [f"{v*100:.4f}" for v in season_v],
        "Eksogen_%":            [f"{v*100:.4f}" for v in exog_v],
        "Proporsi_Trend_%":     [f"{p[0]:.2f}" for p in props],
        "Proporsi_Seasonality_%":[f"{p[1]:.2f}" for p in props],
        "Proporsi_Eksogen_%":   [f"{p[2]:.2f}" for p in props],
    })
    st.download_button(
        "⬇️ Unduh Tabel Dekomposisi (CSV)",
        dl_decomp.to_csv(index=False).encode("utf-8"),
        file_name="dekomposisi_nbeatsx.csv",
        mime="text/csv"
    )

def page_upload():
    st.markdown(
        """
        <div class="main-header">
            <div class="main-title">📂 Upload Data</div>
            <div class="main-subtitle">
                Unggah dataset, atur variabel eksogen, dan jalankan prediksi kustom
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── STEP 1: Upload File ───────────────────────────────────────
    st.markdown(
        "<div class='section-header'>① Unggah File Data</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="info-box">
            <b>Format:</b> CSV (.csv) atau Excel (.xlsx) &nbsp;·&nbsp;
            <b>Maks:</b> 200 MB &nbsp;·&nbsp;
            <b>Frekuensi:</b> Bulanan (min. 36 baris)
        </div>
        """,
        unsafe_allow_html=True
    )

    c_up, c_fmt = st.columns([1.2, 1])

    with c_up:
        uploaded = st.file_uploader(
            "Pilih file CSV atau Excel",
            type=["csv", "xlsx"],
            label_visibility="collapsed",
            key="file_uploader_main"
        )

        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv")                       else pd.read_excel(uploaded)
                raw = normalize_columns(raw)

                missing = {"ds", "y", "BI Rate",
                           "Harga Minyak Dunia", "Kurs USD/IDR"} - set(raw.columns)
                if missing:
                    st.error(f"❌ Kolom tidak lengkap: **{', '.join(missing)}**")
                    st.session_state.uploaded_df    = None
                    st.session_state.upload_status  = "error"
                else:
                    raw["ds"] = pd.to_datetime(raw["ds"])
                    raw = raw.sort_values("ds").reset_index(drop=True)
                    st.session_state.uploaded_df    = raw
                    st.session_state.upload_status  = "ok"
                    # ── Proses sekarang: build features + scale → simpan ke session_state
                    # Ini memastikan df_scaled selalu konsisten dengan data yang diunggah
                    try:
                        _nf, _sy, _se, _bp, _cfg, _fds, _fdr = load_artifacts()
                        _df_feat = build_features(
                            raw[["ds","y","BI Rate","Harga Minyak Dunia","Kurs USD/IDR"]].copy(),
                            _cfg)
                        _df_scaled = scale_df(_df_feat, _sy, _se, _cfg["num_cols"])
                        st.session_state.upload_df_scaled = _df_scaled
                        st.session_state.upload_df_feat   = _df_feat
                        st.session_state.upload_last_date = pd.to_datetime(_df_feat["ds"].max())
                        st.success(
                            f"✅ Data berhasil diunggah & diproses — "
                            f"{len(raw)} baris · "
                            f"{raw['ds'].min().strftime('%b %Y')} – "
                            f"{raw['ds'].max().strftime('%b %Y')}"
                        )
                    except Exception as _e:
                        st.session_state.upload_df_scaled = None
                        st.session_state.upload_df_feat   = None
                        st.session_state.upload_last_date = None
                        st.error(f"⚠️ Data tersimpan tapi gagal diproses: {_e}. Periksa format kolom.")
            except Exception as e:
                st.error(f"❌ Gagal membaca file: {e}")
                st.session_state.uploaded_df = None

        if st.session_state.uploaded_df is not None:
            df_show = st.session_state.uploaded_df
            show_cols = [c for c in
                ["ds","y","BI Rate","Harga Minyak Dunia","Kurs USD/IDR"]
                if c in df_show.columns]
            st.dataframe(
                df_show[show_cols].head (5).style.format({
                    "y": "{:.4f}", "BI Rate": "{:.4f}",
                    "Harga Minyak Dunia": "{:.2f}", "Kurs USD/IDR": "{:.0f}",
                }),
                use_container_width=True, height=220
            )
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Total Baris", len(df_show))
            with col_s2:
                st.metric("Periode Awal", df_show["ds"].min().strftime("%b %Y"))
            with col_s3:
                st.metric("Periode Akhir", df_show["ds"].max().strftime("%b %Y"))
            with col_s4:
                st.metric("Inflasi Terakhir",
                          f"{df_show['y'].iloc[-1]*100:.2f}%")
            if st.button("🗑️  Hapus & gunakan data bawaan",
                         use_container_width=True):
                st.session_state.uploaded_df      = None
                st.session_state.upload_status    = None
                st.session_state.upload_df_scaled = None
                st.session_state.upload_df_feat   = None
                st.session_state.upload_last_date = None
                st.rerun()

    with c_fmt:
        st.markdown(
            "<div class='section-header'>Format Kolom yang Diperlukan</div>",
            unsafe_allow_html=True
        )
        cols_info = [
            ("ds / Date / Tanggal",       "Waktu",   "YYYY-MM-DD"),
            ("y / Inflasi / Inflasi Umum","Target",  "Desimal (0.0265 = 2.65%)"),
            ("BI Rate",                   "Eksogen", "Desimal"),
            ("Harga Minyak Dunia",        "Eksogen", "USD/barel"),
            ("Kurs USD/IDR",              "Eksogen", "Nilai tukar"),
        ]
        badge_map = {"Waktu": "#3D2800|#F6AD55",
                     "Target": "#1A365D|#63B3ED",
                     "Eksogen": "#1C4532|#68D391"}
        rows_fmt = ""
        for name, tipe, fmt in cols_info:
            bg, fg = badge_map[tipe].split("|")
            rows_fmt += (
                f"<tr>"
                f"<td><code style='color:#63B3ED;font-size:0.78rem;'>{name}</code> ✱</td>"
                f"<td><span style='background:{bg};color:{fg};padding:2px 8px;"
                f"border-radius:10px;font-size:0.72rem;font-weight:600;'>{tipe}</span></td>"
                f"<td style='color:#718096;font-size:0.77rem;'>{fmt}</td>"
                f"</tr>"
            )
        st.markdown(
            f"""
            <table class="pred-table">
                <tr><th>Nama Kolom</th><th>Tipe</th><th>Format</th></tr>
                {rows_fmt}
            </table>
            <div style="font-size:0.72rem;color:#4A5568;margin-top:6px;">
                ✱ Wajib · Nama kolom tidak peka huruf besar/kecil
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="warning-box" style="margin-top:1rem;">
                Jika tidak ada data diunggah, sistem menggunakan data bawaan
                model (Jan 2010 – Sep 2025) secara otomatis.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # ── STEP 2: Konfigurasi Prediksi Kustom ──────────────────────
    st.markdown(
        "<div class='section-header'>② Konfigurasi Prediksi Kustom (6 Bulan ke Depan)</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="info-box">
            Atur nilai variabel eksogen makroekonomi dan dummy kalender
            untuk setiap bulan prediksi. Klik <b>Jalankan Prediksi</b>
            setelah semua nilai diisi.
        </div>
        """,
        unsafe_allow_html=True
    )

    try:
        nf, scaler_y, scaler_exog, best, config, full_data_scaled, full_data_raw = load_artifacts()
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return

    # use_data_raw = data dalam skala ASLI (untuk UI, default input, dll)
    # use_data_scaled = data dalam skala NORMALIZED (untuk input model)
    if st.session_state.uploaded_df is not None:
        use_data_raw    = st.session_state.uploaded_df   # data upload = raw
        _is_default     = False
    else:
        use_data_raw    = full_data_raw                  # data bawaan = inverse-transformed
        _is_default     = True

    last_date = pd.to_datetime(use_data_raw["ds"].max())
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1), periods=6, freq="MS"
    )

    # ── Input eksogen per bulan ───────────────────────────────────
    st.markdown("**Variabel Makroekonomi per Bulan Prediksi**")

    # Ambil nilai default dari raw data (skala asli)
    last_bi = (
        float(use_data_raw["BI Rate"].iloc[-1])
        if "BI Rate" in use_data_raw.columns
        else 5.25)

    last_oil = (
        float(use_data_raw["Harga Minyak Dunia"].iloc[-1])
        if "Harga Minyak Dunia" in use_data_raw.columns
        else 75.0
    )

    last_kurs = (
        float(use_data_raw["Kurs USD/IDR"].iloc[-1])
        if "Kurs USD/IDR" in use_data_raw.columns
        else 15500.0
    )

    # Bersihkan NaN
    if pd.isna(last_bi):   last_bi   = 5.25
    if pd.isna(last_oil):  last_oil  = 75.0
    if pd.isna(last_kurs): last_kurs = 15500.0

    # Konversi BI Rate ke persen jika masih desimal (untuk tampilan input)
    if last_bi < 1.0:
        last_bi = last_bi * 100.0

    # Pastikan masuk range number_input
    last_bi   = max(0.0, min(last_bi,   25.0))
    last_oil  = max(0.0, min(last_oil,  500.0))
    last_kurs = max(0.0, min(last_kurs, 99999.0))

    exog_inputs = {}
    header_cols = st.columns([1.6, 1, 1, 1])
    header_cols[0].markdown("**Bulan**")
    header_cols[1].markdown("**BI Rate (%)**")
    header_cols[2].markdown("**Minyak (USD)**")
    header_cols[3].markdown("**Kurs USD/IDR**")

    for i, fdate in enumerate(future_dates):
        row = st.columns([1.6, 1, 1, 1])
        row[0].markdown(
            f"<div style='padding:0.5rem 0;font-size:0.85rem;"
            f"font-weight:600;color:#E8EAF0;'>"
            f"{fdate.strftime('%B %Y')}</div>",
            unsafe_allow_html=True
        )
        bi  = row[1].number_input(
            f"bi_{i}", value=float(last_bi), min_value=0.0,
            max_value=25.0, step=0.25, format="%.2f",
            label_visibility="collapsed", key=f"bi_{i}"
        )
        oil = row[2].number_input(
            f"oil_{i}", value=last_oil, min_value=0.0,
            max_value=500.0, step=0.5, format="%.2f",
            label_visibility="collapsed", key=f"oil_{i}"
        )
        kurs = row[3].number_input(
            f"kurs_{i}", value=last_kurs, min_value=0.0,
            max_value=99999.0, step=50.0, format="%.0f",
            label_visibility="collapsed", key=f"kurs_{i}"
        )
        # Simpan BI Rate dalam %, konversi ke desimal saat dipakai model
        exog_inputs[i] = {"BI Rate": bi,
                          "Harga Minyak Dunia": oil,
                          "Kurs USD/IDR": kurs}

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Dummy Kalender (centang jika bulan tersebut ada hari besar)**")

    dummy_inputs = {}
    d_header = st.columns([1.6, 1, 1, 1, 1])
    d_header[0].markdown("**Bulan**")
    d_header[1].markdown("**Ramadan**")
    d_header[2].markdown("**Idulfitri**")
    d_header[3].markdown("**Natal**")
    d_header[4].markdown("**Imlek**")

    # Default otomatis berdasarkan bulan
    def default_dummy(dt):
        m = dt.month
        return {
            "Ramadhan":  1 if m == 3 else 0,
            "Idulfitri": 1 if m == 4 else 0,
            "Natal":     1 if m == 12 else 0,
            "Imlek":     1 if m == 1 else 0,
        }

    for i, fdate in enumerate(future_dates):
        d_def = default_dummy(fdate)
        row   = st.columns([1.6, 1, 1, 1, 1])
        row[0].markdown(
            f"<div style='padding:0.5rem 0;font-size:0.85rem;"
            f"font-weight:600;color:#E8EAF0;'>"
            f"{fdate.strftime('%B %Y')}</div>",
            unsafe_allow_html=True
        )
        ram  = row[1].checkbox("", value=bool(d_def["Ramadhan"]),
                               key=f"ram_{i}")
        idl  = row[2].checkbox("", value=bool(d_def["Idulfitri"]),
                               key=f"idl_{i}")
        nat  = row[3].checkbox("", value=bool(d_def["Natal"]),
                               key=f"nat_{i}")
        iml  = row[4].checkbox("", value=bool(d_def["Imlek"]),
                               key=f"iml_{i}")
        dummy_inputs[i] = {
            "Ramadhan":  int(ram), "Idulfitri": int(idl),
            "Natal":     int(nat), "Imlek":     int(iml),
        }

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tombol Prediksi ───────────────────────────────────────────
    col_btn, _ = st.columns([1, 3])
    run_pred = col_btn.button(
        "🚀  Jalankan Prediksi Kustom",
        use_container_width=True
    )
    
    if run_pred:
        with st.spinner("Menjalankan prediksi…"):
            try:
                # ── Build df_scaled untuk input model ──────────────
                # PENTING:
                # - Data upload (raw) → perlu build_features + scale_df
                # - Data bawaan (full_data_scaled) → sudah ter-scale, pakai langsung
                if not _is_default:
                    # Data dari user upload: raw → build features → scale
                    df_feat = build_features(
                        use_data_raw[["ds","y","BI Rate",
                                      "Harga Minyak Dunia","Kurs USD/IDR"]].copy(),
                        config
                    )
                    num_cols  = config["num_cols"]
                    df_scaled = scale_df(df_feat, scaler_y, scaler_exog, num_cols)
                else:
                    # Data bawaan: full_data_scaled sudah siap pakai model
                    df_scaled = full_data_scaled.copy()
                    df_feat   = full_data_raw.copy()  # untuk grafik hist_y

                # Build future df dari input user
                # Input BI Rate dari number_input SELALU dalam persen (%) → konversi ke desimal
                fut_rows = []

                for i, fdate in enumerate(future_dates):
                    # BI Rate dari UI selalu dalam %, konversi ke desimal
                    bi_val = exog_inputs[i]["BI Rate"] / 100.0

                    row = {
                        "unique_id": "inflasi",
                        "ds": fdate,
                        "BI Rate": bi_val,
                        "Harga Minyak Dunia": exog_inputs[i]["Harga Minyak Dunia"],
                        "Kurs USD/IDR": exog_inputs[i]["Kurs USD/IDR"],
                        "Ramadhan":  dummy_inputs[i]["Ramadhan"],
                        "Idulfitri": dummy_inputs[i]["Idulfitri"],
                        "Natal":     dummy_inputs[i]["Natal"],
                        "Imlek":     dummy_inputs[i]["Imlek"],
                    }
                    fut_rows.append(row)

                fut_df = pd.DataFrame(fut_rows)
            
                forecast  = nf.predict(df=df_scaled, futr_df=fut_df)
                pred_vals = scaler_y.inverse_transform(
                    forecast[["NBEATSx"]]).flatten()

                # Simpan ke session state
                st.session_state["custom_pred_vals"]  = pred_vals
                st.session_state["custom_pred_dates"] = future_dates
                st.session_state["custom_exog"]       = exog_inputs
                st.session_state["custom_dummy"]       = dummy_inputs
                st.session_state["custom_fut_df"] = fut_df.copy()
                pred_ok = True
            except Exception as e:
                st.error(f"❌ Prediksi gagal: {e}")
                pred_ok = False

        if pred_ok:
            pred_vals    = st.session_state["custom_pred_vals"]
            future_dates_res = st.session_state["custom_pred_dates"]

            st.divider()
            st.markdown(
                "<div class='section-header'>③ Hasil Prediksi Kustom</div>",
                unsafe_allow_html=True
            )

            # Hitung dekomposisi
            with st.spinner("Menghitung dekomposisi…"):
                decomp_custom = decompose_forecast(
                    nf, df_scaled, fut_df, scaler_y
                )
            # Simpan ke session state
            st.session_state["custom_decomp"] = decomp_custom

            # Metrik ringkas
            mc1, mc2, mc3, mc4 = st.columns(4)
            trend_dir = "↑ Naik" if pred_vals[-1] > pred_vals[0] else "↓ Turun"
            trend_col = "#68D391" if pred_vals[-1] > pred_vals[0] else "#FC8181"
            for col, (lbl, val) in zip(
                [mc1, mc2, mc3, mc4],
                [("Prediksi Bulan 1",
                  f"{pred_vals[0]*100:.2f}%"),
                 ("Rata-rata 6 Bulan",
                  f"{np.mean(pred_vals)*100:.2f}%"),
                 ("Tertinggi",
                  f"{np.max(pred_vals)*100:.2f}%"),
                 ("Terendah",
                  f"{np.min(pred_vals)*100:.2f}%")]
            ):
                col.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-label">{lbl}</div>
                        <div class="metric-value" style="font-size:1.2rem;">{val}</div>
                    </div>""",
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            res_left, res_right = st.columns([1.5, 1])

            with res_left:
                # Grafik
                import matplotlib.pyplot as plt
                import matplotlib.dates as mdates
                set_dark_style()
                # Gunakan df_feat (raw) untuk hist_y agar nilai benar di grafik
                hist_y  = df_feat["y"].values[-24:]
                hist_ds = pd.to_datetime(df_feat["ds"].values[-24:])

                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(hist_ds, hist_y * 100,
                        color="#63B3ED", linewidth=1.8,
                        marker="o", markersize=3, label="Aktual")
                ax.plot([hist_ds[-1], future_dates_res[0]],
                        [hist_y[-1]*100, pred_vals[0]*100],
                        color="#F6AD55", linewidth=1.5,
                        linestyle="--", alpha=0.5)
                ax.plot(future_dates_res, pred_vals * 100,
                        color="#F6AD55", linewidth=2,
                        marker="s", markersize=5, label="Prediksi Kustom")
                ax.fill_between(future_dates_res,
                                pred_vals * 100 * 0.85,
                                pred_vals * 100 * 1.15,
                                alpha=0.1, color="#F6AD55")
                for d, v in zip(future_dates_res, pred_vals):
                    ax.annotate(
                        f"{v*100:.2f}%", xy=(d, v*100),
                        xytext=(0, 10), textcoords="offset points",
                        fontsize=7.5, color="#F6AD55",
                        ha="center", fontfamily="monospace"
                    )
                ax.axvline(x=pd.to_datetime(last_date),
                           color="#4A5568", linewidth=1,
                           linestyle=":", alpha=0.8)
                ax.set_ylabel("Inflasi (%)")
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x:.2f}%"))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                ax.xaxis.set_major_locator(
                    mdates.MonthLocator(interval=3))
                plt.setp(ax.xaxis.get_majorticklabels(),
                         rotation=30, ha="right")
                ax.legend(fontsize=9, framealpha=0.3,
                          facecolor="#1A202C", edgecolor="#2D3748")
                ax.grid(True, alpha=0.4)
                ax.set_title(
                    "Prediksi Inflasi Kustom — N-BEATSx",
                    fontsize=10, pad=10, color="#E8EAF0",
                    fontfamily="monospace"
                )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with res_right:
                # Tabel hasil
                rows_r = ""
                for i, (d, v) in enumerate(
                    zip(future_dates_res, pred_vals)
                ):
                    pct   = v * 100
                    color = "#68D391" if pct < 3 else                             "#F6AD55" if pct < 5 else "#FC8181"
                    exog  = exog_inputs[i]
                    rows_r += (
                        f"<tr>"
                        f"<td>{pd.to_datetime(d).strftime('%b %Y')}</td>"
                        f"<td style='color:{color};font-weight:600;'>"
                        f"{pct:.4f}%</td>"
                        f"<td>{exog['BI Rate']:.2f}%</td>"
                        f"<td>{exog['Harga Minyak Dunia']:.1f}</td>"
                        f"<td>{exog['Kurs USD/IDR']:,.0f}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f"""
                    <table class="pred-table">
                        <tr>
                            <th>Bulan</th>
                            <th>Prediksi</th>
                            <th>BI Rate</th>
                            <th>Minyak</th>
                            <th>Kurs</th>
                        </tr>
                        {rows_r}
                    </table>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)

                # Download
                dl_data = {
                    "Periode": [
                        pd.to_datetime(d).strftime("%Y-%m")
                        for d in future_dates_res
                    ],
                    "Prediksi_Inflasi_%": [
                        f"{v*100:.4f}" for v in pred_vals
                    ],
                    "BI_Rate": [
                        exog_inputs[i]["BI Rate"]
                        for i in range(6)
                    ],
                    "Harga_Minyak": [
                        exog_inputs[i]["Harga Minyak Dunia"]
                        for i in range(6)
                    ],
                    "Kurs_USDIDR": [
                        exog_inputs[i]["Kurs USD/IDR"]
                        for i in range(6)
                    ],
                }
                st.download_button(
                    "⬇️ Download Hasil (CSV)",
                    pd.DataFrame(dl_data).to_csv(
                        index=False).encode("utf-8"),
                    file_name="prediksi_kustom.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # ── Tab Dekomposisi Kustom ────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                "<div class='section-header'>Dekomposisi Komponen — Prediksi Kustom</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                """<div class="info-box">
                    Dekomposisi prediksi kustom berdasarkan nilai eksogen
                    yang Anda masukkan. Komponen
                    <b style="color:#68D391;">Trend</b>,
                    <b style="color:#F6AD55;">Seasonality</b>, dan
                    <b style="color:#63B3ED;">Eksogen</b> (BI Rate,
                    harga minyak, kurs, lag inflasi).
                </div>""",
                unsafe_allow_html=True
            )
            if "custom_decomp" in st.session_state:
                render_decomp_tab(
                    st.session_state["custom_decomp"],
                    future_dates_res,
                    label="Prediksi Kustom"
                )

    elif "custom_pred_vals" in st.session_state:
        # Tampilkan hasil prediksi terakhir jika sudah pernah dijalankan
        st.info("ℹ️ Menampilkan hasil prediksi terakhir. "
                "Ubah nilai di atas dan klik 'Jalankan Prediksi' untuk memperbarui.")



# ═══════════════════════════════════════════════════════════════════
# PAGE: VISUALISASI DATA
# ═══════════════════════════════════════════════════════════════════
def page_visualisasi():
    st.markdown("""
    <div class='main-header'>
        <div class='main-title'>📊 Visualisasi Data</div>
        <div class='main-subtitle'>
            Eksplorasi pola historis inflasi dan variabel makroekonomi
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        _, scaler_y, scaler_exog, _, config, full_data_scaled, full_data_raw = load_artifacts()
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return

    use_data = st.session_state.uploaded_df \
               if st.session_state.uploaded_df is not None \
               else full_data_raw   # gunakan raw (skala asli) untuk visualisasi
    data_src = "Upload" if st.session_state.uploaded_df is not None \
               else "Data Bawaan Model"

    st.markdown(f"""
    <div class='info-box'>
        📌 Sumber data aktif: <b>{data_src}</b> ·
        {len(use_data)} observasi ·
        {pd.to_datetime(use_data['ds'].min()).strftime('%b %Y')} –
        {pd.to_datetime(use_data['ds'].max()).strftime('%b %Y')}
    </div>""", unsafe_allow_html=True)

    set_dark_style()

    df_viz = use_data.copy()
    df_viz['ds'] = pd.to_datetime(df_viz['ds'])
    df_viz = df_viz.sort_values('ds')

    tab1, tab2, tab3 = st.tabs([
        "📉 Inflasi", "🏦 Variabel Makroekonomi", "📋 Statistik Deskriptif"
    ])

    with tab1:
        st.markdown("<div class='section-header'>Pola Historis Inflasi</div>",
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df_viz['ds'], df_viz['y'] * 100,
                color='#63B3ED', linewidth=1.5,
                marker='o', markersize=2.5, label='Inflasi (%)')
        ax.axhline(y=df_viz['y'].mean() * 100, color='#F6AD55',
                   linewidth=1, linestyle='--',
                   label=f"Rata-rata: {df_viz['y'].mean()*100:.2f}%")
        ax.fill_between(df_viz['ds'], df_viz['y'] * 100, 0,
                        alpha=0.08, color='#63B3ED')
        ax.set_ylabel('Inflasi (%)')
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        ax.legend(fontsize=9, framealpha=0.3,
                  facecolor='#1A202C', edgecolor='#2D3748')
        ax.grid(True, alpha=0.4)
        ax.set_title('Inflasi Indonesia — Data Historis',
                     fontsize=11, pad=12, color='#E8EAF0',
                     fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Statistik inflasi
        s1, s2, s3, s4 = st.columns(4)
        stats = [
            ('Rata-rata', f"{df_viz['y'].mean()*100:.4f}%"),
            ('Minimum',   f"{df_viz['y'].min()*100:.4f}%"),
            ('Maksimum',  f"{df_viz['y'].max()*100:.4f}%"),
            ('Std Dev',   f"{df_viz['y'].std()*100:.4f}%"),
        ]
        for col, (lbl, val) in zip([s1, s2, s3, s4], stats):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{lbl}</div>
                    <div class='metric-value' style='font-size:1.2rem;'>{val}</div>
                </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='section-header'>Variabel Makroekonomi</div>",
                    unsafe_allow_html=True)
        exog_vars = [
            ('BI Rate',            '#68D391', 'BI Rate (%)'),
            ('Harga Minyak Dunia', '#F6AD55', 'Harga Minyak (USD/bbl)'),
            ('Kurs USD/IDR',       '#FC8181', 'Kurs USD/IDR'),
        ]
        for var, color, ylabel in exog_vars:
            if var not in df_viz.columns:
                continue
            fig2, ax2 = plt.subplots(figsize=(12, 3))
            ax2.plot(df_viz['ds'], df_viz[var],
                     color=color, linewidth=1.5, marker='o', markersize=2)
            ax2.fill_between(df_viz['ds'], df_viz[var],
                              df_viz[var].min(), alpha=0.08, color=color)
            ax2.set_ylabel(ylabel, fontsize=9)
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
            ax2.grid(True, alpha=0.4)
            ax2.set_title(var, fontsize=10, pad=8, color='#E8EAF0',
                          fontfamily='monospace')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()
            st.markdown("<br>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='section-header'>Statistik Deskriptif</div>",
                    unsafe_allow_html=True)
        num_cols_viz = ['y', 'BI Rate', 'Harga Minyak Dunia', 'Kurs USD/IDR']
        num_cols_viz = [c for c in num_cols_viz if c in df_viz.columns]
        desc = df_viz[num_cols_viz].describe().T.round(4)
        desc.index = ['Inflasi (y)', 'BI Rate',
                      'Harga Minyak', 'Kurs USD/IDR'][:len(desc)]
        st.dataframe(desc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE: ABOUT US
# ═══════════════════════════════════════════════════════════════════
def page_about():
    st.markdown("""
    <div class='main-header'>
        <div class='main-title'>👥 About Us</div>
        <div class='main-subtitle'>
            Latar belakang penelitian dan pengembangan sistem
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("<div class='section-header'>Latar Belakang Penelitian</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='about-card'>
            <div class='about-text'>
                Inflasi merupakan indikator utama stabilitas ekonomi yang
                berperan penting dalam pengambilan kebijakan moneter dan fiskal.
                Di Indonesia, pola inflasi dipengaruhi tidak hanya oleh faktor
                makroekonomi seperti suku bunga, harga minyak dunia, dan nilai
                tukar, tetapi juga oleh variasi musiman akibat hari besar
                keagamaan yang menyebabkan fluktuasi harga periodik.
                <br><br>
                Metode prediksi konvensional umumnya memiliki keterbatasan
                dalam menangkap hubungan nonlinier serta pengaruh variabel
                eksogen dan variasi kalender secara simultan. Penelitian ini
                mengembangkan model prediksi inflasi menggunakan <b>N-BEATSx</b>,
                model <i>deep learning</i> berbasis dekomposisi yang mampu
                mengintegrasikan variabel eksogen makroekonomi dan efek
                kalender dalam proses peramalan.
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Tujuan Pengembangan</div>",
                    unsafe_allow_html=True)
        tujuan = [
            ('01', 'Mengembangkan model prediksi inflasi Indonesia yang '
                   'mengintegrasikan variabel makroekonomi dan efek kalender '
                   'hari besar keagamaan.'),
            ('02', 'Mengoptimasi hiperparameter model N-BEATSx menggunakan '
                   'Bayesian Optimization dua tahap dengan Optuna.'),
            ('03', 'Membandingkan performa N-BEATSx dengan model baseline '
                   '(N-BEATS, Prophet, SARIMAX) pada data inflasi Indonesia.'),
            ('04', 'Membangun sistem prediksi berbasis web yang dapat '
                   'digunakan sebagai alat bantu analisis inflasi.'),
        ]
        for num, desc in tujuan:
            st.markdown(f"""
            <div style='display:flex;gap:1rem;margin-bottom:0.75rem;
                        align-items:flex-start;'>
                <div style='font-family:Space Mono,monospace;
                            font-size:0.8rem;font-weight:700;
                            color:#63B3ED;padding-top:2px;
                            min-width:24px;'>{num}</div>
                <div style='font-size:0.84rem;color:#A0AEC0;
                            line-height:1.65;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-header'>Spesifikasi Model</div>",
                    unsafe_allow_html=True)
        specs = [
            ('Arsitektur',      'N-BEATSx (Interpretable)'),
            ('Stack',           'Trend + Seasonality'),
            ('Optimizer',       'Adam + Bayesian Opt.'),
            ('Periode Data',    'Jan 2010 – Sep 2025'),
            ('Observasi',       '189 bulan'),
            ('Horizon',         '6 bulan ke depan'),
            ('Input Size',      '24 bulan'),
            ('Hidden Size',     '256 neuron'),
            ('N-Blocks',        '[3, 3]'),
            ('Max Steps',       '850'),
            ('Dropout',         '0.1'),
            ('Learning Rate',   '0.000432'),
        ]
        rows_s = "".join([
            f"<div class='val-row'>"
            f"<span class='val-key'>{k}</span>"
            f"<span class='val-val'>{v}</span></div>"
            for k, v in specs
        ])
        st.markdown(f"<div class='about-card'>{rows_s}</div>",
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Variabel Model</div>",
                    unsafe_allow_html=True)
        var_groups = [
            ('Target', ['Inflasi Bulanan (y)'], 'badge-blue'),
            ('Lag Inflasi', ['lag1', 'lag3', 'lag6', 'lag12'], 'badge-green'),
            ('Eksogen Hist.', ['BI Rate', 'Harga Minyak Dunia',
                               'Kurs USD/IDR'], 'badge-green'),
            ('Kalender', ['Ramadan', 'Idulfitri', 'Natal', 'Imlek'], 'badge-yellow'),
        ]
        for group, items, badge_cls in var_groups:
            badges = " ".join([
                f"<span class='badge {badge_cls}'>{item}</span>"
                for item in items
            ])
            st.markdown(f"""
            <div style='margin-bottom:0.6rem;'>
                <div style='font-size:0.72rem;color:#4A5568;
                            text-transform:uppercase;letter-spacing:0.08em;
                            margin-bottom:4px;'>{group}</div>
                {badges}
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Manfaat Sistem</div>",
                    unsafe_allow_html=True)
        manfaat = [
            ('🏛️', 'Pengambil kebijakan dapat menggunakan prediksi sebagai '
                   'referensi awal dalam perencanaan kebijakan moneter.'),
            ('📊', 'Analis ekonomi dapat mengeksplorasi pola historis dan '
                   'proyeksi inflasi secara interaktif.'),
            ('🎓', 'Kontribusi akademik dalam penerapan deep learning '
                   'untuk prediksi variabel makroekonomi Indonesia.'),
        ]
        for icon, desc in manfaat:
            st.markdown(f"""
            <div style='display:flex;gap:0.75rem;margin-bottom:0.7rem;'>
                <div style='font-size:1.1rem;'>{icon}</div>
                <div style='font-size:0.82rem;color:#A0AEC0;
                            line-height:1.65;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Teknologi yang Digunakan</div>",
                unsafe_allow_html=True)
    techs = [
        ('Python 3.10', 'badge-blue'),
        ('PyTorch', 'badge-blue'),
        ('NeuralForecast', 'badge-blue'),
        ('Optuna', 'badge-green'),
        ('Streamlit', 'badge-green'),
        ('Pandas', 'badge-yellow'),
        ('Scikit-learn', 'badge-yellow'),
        ('Matplotlib', 'badge-yellow'),
    ]
    tech_badges = " ".join([
        f"<span class='badge {cls}'>{name}</span>"
        for name, cls in techs
    ])
    st.markdown(f"<div style='margin-bottom:1rem;'>{tech_badges}</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        <b>Catatan:</b> Sistem ini dikembangkan sebagai bagian dari penelitian
        skripsi dengan fokus pada penerapan <i>deep learning</i> untuk
        prediksi inflasi Indonesia. Prediksi yang dihasilkan bersifat
        indikatif dan tidak dapat menggantikan analisis kebijakan komprehensif
        oleh otoritas terkait.
    </div>""", unsafe_allow_html=True)




def page_prediksi():
    st.markdown("""
    <div class='main-header'>
        <div class='main-title'>📈 Prediksi Inflasi</div>
        <div class='main-subtitle'>
            Hasil prediksi 6 bulan ke depan — Model N-BEATSx + Bayesian Optimization
        </div>
    </div>""", unsafe_allow_html=True)

    try:
        nf, scaler_y, scaler_exog, best, config, full_data_scaled, full_data_raw = load_artifacts()
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return

    # ── Tentukan sumber data ──────────────────────────────────────
    _default = st.session_state.uploaded_df is None
    use_raw  = full_data_raw if _default else st.session_state.uploaded_df
    src_lbl  = "Data Bawaan Model" if _default else "Data Upload"
    # CEK APAKAH ADA HASIL PREDIKSI CUSTOM
    use_custom_forecast = (
        st.session_state.get("custom_pred_vals") is not None
        and st.session_state.get("custom_pred_dates") is not None
    )
    st.markdown(f"""
    <div class='info-box'>
        📌 Sumber data: <b>{src_lbl}</b> ·
        {len(use_raw)} observasi ·
        {pd.to_datetime(use_raw["ds"].min()).strftime("%b %Y")} –
        {pd.to_datetime(use_raw["ds"].max()).strftime("%b %Y")}
    </div>""", unsafe_allow_html=True)

    # ── Ambil df_scaled & df_feat dari session_state (sudah diproses saat upload) ──
    try:
        if not _default:
            # Gunakan hasil processing yang disimpan saat upload
            df_scaled = st.session_state.upload_df_scaled
            df_feat   = st.session_state.upload_df_feat

            # Fallback: jika session_state hilang (misal setelah refresh browser),
            # rebuild dari uploaded_df
            if df_scaled is None or df_feat is None:
                df_feat = build_features(
                    use_raw[["ds","y","BI Rate",
                              "Harga Minyak Dunia","Kurs USD/IDR"]].copy(),
                    config)
                df_scaled = scale_df(df_feat, scaler_y, scaler_exog,
                                     config["num_cols"])
                # Simpan kembali ke session_state
                st.session_state.upload_df_scaled = df_scaled
                st.session_state.upload_df_feat   = df_feat
        else:
            df_scaled = full_data_scaled.copy()
            df_feat   = full_data_raw.copy()

        if use_custom_forecast:

            pred_vals = st.session_state["custom_pred_vals"]
            future_dates = st.session_state["custom_pred_dates"]

        else:
            last_date = pd.to_datetime(df_feat["ds"].max())
            fut_dummy = make_future_dummy(
                last_date,
                config["h"],
                config
            )

            forecast = nf.predict(
                df=df_scaled,
                futr_df=fut_dummy
            )

            pred_vals = scaler_y.inverse_transform(
                forecast[["NBEATSx"]]
            ).flatten()

            future_dates = forecast["ds"].values

        # hist_y untuk display (skala asli / raw)
        if not _default:
            # df_scaled berasal dari scale_df → inverse untuk display
            hist_y = scaler_y.inverse_transform(
                df_scaled[["y"]].values).flatten()
        else:
            # full_data_raw SUDAH raw
            hist_y = df_feat["y"].values.flatten()

        hist_ds  = pd.to_datetime(df_feat["ds"].values)
        data_ok  = True
        if use_custom_forecast:
            st.success(
                "Menggunakan hasil prediksi kustom terakhir."
            )
            st.write("DEBUG:", future_dates)
    except Exception as e:
        import traceback
        st.error(f"❌ Error prediksi: {e}")
        st.code(traceback.format_exc())
        data_ok = False

    if not data_ok:
        return

    # ── KPI cards ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    td = "↑" if pred_vals[-1] > pred_vals[0] else "↓"
    tc = "#68D391" if pred_vals[-1] > pred_vals[0] else "#FC8181"
    for col, (lbl, val, sub) in zip(
        [c1, c2, c3, c4],
        [("Prediksi Bulan Pertama",
          f"{pred_vals[0]*100:.2f}%",
          pd.to_datetime(future_dates[0]).strftime("%b %Y")),
         ("Rata-rata 6 Bulan",
          f"{np.mean(pred_vals)*100:.2f}%", "Mean prediksi"),
         ("Arah Tren",
          f"<span style='color:{tc};'>{td}</span>",
          f"{pred_vals[0]*100:.2f}% → {pred_vals[-1]*100:.2f}%"),
         ("Horizon Prediksi", "6", "Bulan ke depan")]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{lbl}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab_decomp, tab3, tab4 = st.tabs([
        "📊 Grafik Prediksi", "📋 Tabel Hasil",
        "🧩 Dekomposisi", "🔍 Analisis Model", "ℹ️ Panduan"
    ])

    # ════════════════════════════════════════════════════════
    # TAB 1 — GRAFIK PREDIKSI
    # ════════════════════════════════════════════════════════
    with tab1:
        set_dark_style()
        fig, ax = plt.subplots(figsize=(12, 4.5))

        # Tampilkan 24 obs historis terakhir (dari data aktif)
        n_show   = min(24, len(hist_y))
        show_ds  = hist_ds[-n_show:]
        show_y   = hist_y[-n_show:]

        ax.plot(show_ds, show_y * 100,
                color="#63B3ED", linewidth=1.8,
                marker="o", markersize=3,
                label=f"Aktual ({src_lbl})", zorder=3)
        # Garis penghubung ke prediksi
        ax.plot([show_ds[-1], pd.to_datetime(future_dates[0])],
                [show_y[-1]*100, pred_vals[0]*100],
                color="#F6AD55", linewidth=1.5, linestyle="--", alpha=0.6)
        # Plot prediksi
        ax.plot(future_dates, pred_vals * 100,
                color="#F6AD55", linewidth=2,
                marker="s", markersize=5,
                label="Prediksi N-BEATSx", zorder=4)
        ax.fill_between(future_dates,
                        pred_vals*100*0.9, pred_vals*100*1.1,
                        alpha=0.1, color="#F6AD55",
                        label="Interval ±10%")
        for d, v in zip(future_dates, pred_vals):
            ax.annotate(f"{v*100:.2f}%", xy=(d, v*100),
                        xytext=(0, 12), textcoords="offset points",
                        fontsize=7.5, color="#F6AD55",
                        ha="center", fontfamily="monospace")
        # Garis batas forecast
        ax.axvline(x=show_ds[-1], color="#4A5568",
                   linewidth=1, linestyle=":", alpha=0.8,
                   label="Mulai Forecast")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.set_ylabel("Inflasi (%)")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.2f}%"))
        ax.legend(fontsize=9, framealpha=0.3,
                  facecolor="#1A202C", edgecolor="#2D3748")
        ax.grid(True, alpha=0.4)
        ax.set_title(
            f"Prediksi Inflasi Indonesia — N-BEATSx ({src_lbl})",
            fontsize=11, pad=12, color="#E8EAF0", fontfamily="monospace")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Info sumber data
        if not _default:
            st.markdown(
                f"""<div class="success-box">
                    ✅ Prediksi dihasilkan dari <b>data upload</b> kamu
                    ({len(hist_y)} observasi,
                    {hist_ds[0].strftime("%b %Y")} –
                    {hist_ds[-1].strftime("%b %Y")}).
                    Grafik menampilkan 24 observasi terakhir sebagai konteks historis.
                </div>""",
                unsafe_allow_html=True
            )

    # ════════════════════════════════════════════════════════
    # TAB 2 — TABEL HASIL
    # ════════════════════════════════════════════════════════
    with tab2:
        if not _default:
            st.markdown(
                f"""<div class="info-box">
                    Prediksi dihasilkan dari <b>data upload</b>
                    ({hist_ds[0].strftime("%b %Y")} –
                    {hist_ds[-1].strftime("%b %Y")},
                    {len(hist_y)} observasi).
                    Model menggunakan 24 observasi terakhir sebagai window input.
                </div>""",
                unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                "<div class='section-header'>Prediksi 6 Bulan ke Depan</div>",
                unsafe_allow_html=True)
            rows_p = ""
            for i, (d, v) in enumerate(zip(future_dates, pred_vals)):
                pct  = v * 100
                clr  = "#68D391" if pct < 3 else                        "#F6AD55" if pct < 5 else "#FC8181"
                cat  = "Rendah ✓" if pct < 3 else                        "Moderat"  if pct < 5 else "Tinggi ⚠"
                rows_p += (
                    f"<tr><td>{i+1}</td>"
                    f"<td>{pd.to_datetime(d).strftime('%B %Y')}</td>"
                    f"<td style='color:{clr};font-weight:600;'>{pct:.4f}%</td>"
                    f"<td style='color:{clr};'>{cat}</td></tr>"
                )
            st.markdown(
                f"""<table class="pred-table">
                    <tr><th>#</th><th>Periode</th>
                    <th>Prediksi</th><th>Kategori</th></tr>
                    {rows_p}
                </table>""",
                unsafe_allow_html=True)

        with col_b:
            st.markdown(
                "<div class='section-header'>Data Historis Terakhir (12 Obs)</div>",
                unsafe_allow_html=True)
            rows_h = ""
            for d, v in zip(hist_ds[-12:], hist_y[-12:]):
                rows_h += (
                    f"<tr><td>{pd.to_datetime(d).strftime('%b %Y')}</td>"
                    f"<td>{v*100:.4f}%</td></tr>"
                )
            st.markdown(
                f"""<table class="pred-table">
                    <tr><th>Periode</th><th>Inflasi Aktual</th></tr>
                    {rows_h}
                </table>""",
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        dl_df = pd.DataFrame({
            "Periode": [pd.to_datetime(d).strftime("%Y-%m")
                        for d in future_dates],
            "Prediksi_Inflasi_%": [f"{v*100:.4f}" for v in pred_vals],
            "Sumber_Data": src_lbl,
        })
        st.download_button(
            "⬇️ Download Hasil Prediksi (CSV)",
            dl_df.to_csv(index=False).encode("utf-8"),
            file_name="prediksi_inflasi.csv", mime="text/csv")

    # ════════════════════════════════════════════════════════
    # TAB 3 — DEKOMPOSISI
    # ════════════════════════════════════════════════════════
    with tab_decomp:
        if _default:
            # Data bawaan → hardcoded dari kode penelitian
            st.markdown(
                """<div class="info-box">
                    Dekomposisi komponen N-BEATSx periode uji
                    Oktober 2024 – September 2025, berdasarkan hasil
                    kode penelitian menggunakan model <i>trend-only</i>
                    dan <i>seasonality-only</i> yang dilatih terpisah.
                </div>""",
                unsafe_allow_html=True)

            DECOMP_DATA = {
                "ds": pd.to_datetime([
                    "2024-10-01","2024-11-01","2024-12-01",
                    "2025-01-01","2025-02-01","2025-03-01",
                    "2025-04-01","2025-05-01","2025-06-01",
                    "2025-07-01","2025-08-01","2025-09-01"]),
                "y_orig": [
                    0.0171,0.0155,0.0157,0.0076,-0.0009,0.0103,
                    0.0195,0.0160,0.0187,0.0237,0.0231,0.0265],
                "NBEATSx_orig": [
                    0.019326,0.020110,0.019025,0.019436,0.020475,0.021181,
                    0.013474,0.017076,0.021259,0.023222,0.029173,0.028163],
                "trend_orig": [
                    0.018970,0.017318,0.016566,0.016713,0.017760,0.019706,
                    0.012251,0.020248,0.026155,0.029972,0.031698,0.031335],
                "seasonality_orig": [
                    0.018408,0.017196,0.015796,0.016647,0.019020,0.023279,
                    0.012942,0.017895,0.026751,0.027229,0.033164,0.034063],
                "exogenous_orig": [
                    0.067489,0.071137,0.072205,0.071617,0.069236,0.063737,
                    0.073822,0.064474,0.053894,0.051563,0.049852,0.048306],
            }
            decomp_df = pd.DataFrame(DECOMP_DATA)
            for col_ in ["trend_orig","seasonality_orig","exogenous_orig"]:
                tot_ = (decomp_df["trend_orig"].abs()
                        + decomp_df["seasonality_orig"].abs()
                        + decomp_df["exogenous_orig"].abs())
                decomp_df[col_.replace("_orig","_pct")] =                     decomp_df[col_].abs() / tot_ * 100

            # Metrik proporsi rata-rata
            avg_t = decomp_df["trend_pct"].mean()
            avg_s = decomp_df["seasonality_pct"].mean()
            avg_e = decomp_df["exogenous_pct"].mean()

            dm1, dm2, dm3 = st.columns(3)
            for col_, (lbl, val, clr, desc) in zip(
                [dm1, dm2, dm3],
                [("Proporsi Tren",    f"{avg_t:.2f}%", "#68D391",
                  "Stack Trend (Blok 0–2)"),
                 ("Proporsi Musiman", f"{avg_s:.2f}%", "#F6AD55",
                  "Stack Seasonality (Blok 3–5)"),
                 ("Proporsi Eksogen", f"{avg_e:.2f}%", "#63B3ED",
                  "BI Rate · Minyak · Kurs · Lag")]
            ):
                with col_:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{lbl}</div>
                        <div class="metric-value" style="color:{clr};">{val}</div>
                        <div class="metric-sub">{desc}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Grafik 4 panel
            set_dark_style()
            fig_d, axes_d = plt.subplots(4, 1, figsize=(11,14))
            fig_d.suptitle(
                "Dekomposisi Komponen N-BEATSx — Test Set (Okt 2024 – Sep 2025)",
                fontsize=11, fontweight="bold", y=1.01)
            ds_vals = decomp_df["ds"].values

            ax0d = axes_d[0]
            ax0d.plot(ds_vals, decomp_df["y_orig"]*100,
                      "o-", color="#63B3ED", lw=1.8, ms=4, label="Aktual")
            ax0d.plot(ds_vals, decomp_df["NBEATSx_orig"]*100,
                      "s--", color="#FC8181", lw=1.8, ms=5,
                      label="Prediksi N-BEATSx")
            ax0d.fill_between(ds_vals,
                              decomp_df["y_orig"]*100,
                              decomp_df["NBEATSx_orig"]*100,
                              alpha=0.1, color="#FC8181")
            ax0d.set_title("Prediksi vs Aktual — Test Set",
                           fontsize=10, pad=8)
            ax0d.set_ylabel("Inflasi (%)", fontsize=9)
            ax0d.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.3f}%"))
            ax0d.legend(fontsize=9, framealpha=.3,
                        facecolor="#1A202C", edgecolor="#2D3748")
            ax0d.grid(True, alpha=.4)

            for ax_, col_, clr_, title_ in [
                (axes_d[1], "trend_orig",       "#68D391",
                 "Komponen Trend"),
                (axes_d[2], "seasonality_orig", "#F6AD55",
                 "Komponen Seasonality (Termasuk Efek Kalender)"),
            ]:
                ax_.plot(ds_vals, decomp_df[col_],
                         "o-", color=clr_, lw=1.8, ms=4, label=col_.split("_")[0].title())
                ax_.fill_between(ds_vals, decomp_df[col_], 0,
                                 alpha=0.12, color=clr_)
                ax_.axhline(y=0, color="#4A5568", lw=0.8, ls=":")
                ax_.set_title(title_, fontsize=10, pad=8)
                ax_.set_ylabel("Kontribusi", fontsize=9)
                ax_.legend(fontsize=9, framealpha=.3,
                           facecolor="#1A202C", edgecolor="#2D3748")
                ax_.grid(True, alpha=.4)

            # Panel eksogen: bar chart
            ax3d = axes_d[3]
            xlbls_d = [pd.Timestamp(d).strftime("%b %Y") for d in ds_vals]
            xpos_d  = np.arange(len(xlbls_d))
            bars_d  = ax3d.bar(xpos_d, decomp_df["exogenous_orig"],
                               color="#63B3ED", alpha=0.85,
                               width=0.6, label="Eksogen")
            for bar, val in zip(bars_d, decomp_df["exogenous_orig"]):
                ax3d.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + decomp_df["exogenous_orig"].max()*0.02,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=7, color="#A0AEC0", fontfamily="monospace")
            ax3d.set_xticks(xpos_d)
            ax3d.set_xticklabels(xlbls_d, rotation=30, ha="right", fontsize=8)
            ax3d.set_title(
                "Komponen Eksogen (BI Rate, Harga Minyak, Kurs USD/IDR, Lag)",
                fontsize=10, pad=8)
            ax3d.set_ylabel("Kontribusi", fontsize=9)
            ax3d.set_xlabel("Tanggal", fontsize=9)
            ax3d.legend(fontsize=9, framealpha=.3,
                        facecolor="#1A202C", edgecolor="#2D3748")
            ax3d.grid(True, alpha=.4, axis="y")

            for ax_ in axes_d[:3]:
                ax_.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                ax_.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.setp(ax_.xaxis.get_majorticklabels(),
                         rotation=30, ha="right", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_d)
            plt.close()

            st.markdown("<br>", unsafe_allow_html=True)

            # Tabel dekomposisi
            st.markdown(
                "<div class='section-header'>Tabel Dekomposisi</div>",
                unsafe_allow_html=True)
            rows_d = ""
            for _, row_ in decomp_df.iterrows():
                rows_d += (
                    f"<tr>"
                    f"<td>{row_['ds'].strftime('%B %Y')}</td>"
                    f"<td style='color:#FC8181;'>{row_['y_orig']*100:.4f}%</td>"
                    f"<td style='color:#63B3ED;font-weight:600;'>{row_['NBEATSx_orig']*100:.4f}%</td>"
                    f"<td style='color:#68D391;'>{row_['trend_orig']*100:.4f}%</td>"
                    f"<td style='color:#F6AD55;'>{row_['seasonality_orig']*100:.4f}%</td>"
                    f"<td style='color:#63B3ED;'>{row_['exogenous_orig']*100:.4f}%</td>"
                    f"<td style='color:#718096;font-size:0.78rem;'>"
                    f"T:{row_['trend_pct']:.1f}% "
                    f"S:{row_['seasonality_pct']:.1f}% "
                    f"E:{row_['exogenous_pct']:.1f}%</td>"
                    f"</tr>"
                )
            st.markdown(
                f"""<table class="pred-table">
                    <tr>
                        <th>Periode</th><th>Aktual</th>
                        <th>Prediksi</th><th>Tren</th>
                        <th>Musiman</th><th>Eksogen</th>
                        <th>Proporsi</th>
                    </tr>{rows_d}
                </table>""",
                unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            dl_d = decomp_df[["ds","trend_orig","seasonality_orig",
                               "exogenous_orig","NBEATSx_orig","y_orig",
                               "trend_pct","seasonality_pct","exogenous_pct"]].copy()
            dl_d["ds"] = dl_d["ds"].dt.strftime("%Y-%m-%d")
            st.download_button(
                "⬇️ Download Tabel Dekomposisi (CSV)",
                dl_d.to_csv(index=False).encode("utf-8"),
                file_name="dekomposisi_nbeatsx.csv", mime="text/csv")

        else:
            # Data upload → hitung dekomposisi dari data user
            st.markdown(
                f"""<div class="info-box">
                    Dekomposisi komponen N-BEATSx berdasarkan
                    <b>data yang Anda unggah</b>
                    ({hist_ds.min().strftime("%b %Y")} –
                    {hist_ds.max().strftime("%b %Y")},
                    {len(df_feat)} observasi).
                    Prediksi masa depan:
                    {pd.to_datetime(future_dates[0]).strftime("%b %Y")} –
                    {pd.to_datetime(future_dates[-1]).strftime("%b %Y")}.
                </div>""",
                unsafe_allow_html=True)
            with st.spinner("Menghitung dekomposisi…"):
                dc = decompose_forecast(nf, df_scaled, fut_dummy, scaler_y)

            if dc["success"]:
                du_t, du_s, du_e, du_tot = (
                    dc["trend"], dc["seasonality"],
                    dc["exogenous"], dc["total"])
                props_u = []
                for t_, s_, e_ in zip(du_t, du_s, du_e):
                    den = abs(t_)+abs(s_)+abs(e_)
                    den = den if den > 1e-12 else 1.0
                    props_u.append((abs(t_)/den*100,
                                    abs(s_)/den*100,
                                    abs(e_)/den*100))
                avg_tu = np.mean([p[0] for p in props_u])
                avg_su = np.mean([p[1] for p in props_u])
                avg_eu = np.mean([p[2] for p in props_u])

                pu1, pu2, pu3 = st.columns(3)
                for col_, (lbl, val, clr, desc) in zip(
                    [pu1, pu2, pu3],
                    [("Proporsi Tren",    f"{avg_tu:.2f}%", "#68D391",
                      "Stack Trend (Blok 0–2)"),
                     ("Proporsi Musiman", f"{avg_su:.2f}%", "#F6AD55",
                      "Stack Seasonality (Blok 3–5)"),
                     ("Proporsi Eksogen", f"{avg_eu:.2f}%", "#63B3ED",
                      "BI Rate · Minyak · Kurs · Lag")]
                ):
                    with col_:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{lbl}</div>
                            <div class="metric-value" style="color:{clr};">{val}</div>
                            <div class="metric-sub">{desc}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                set_dark_style()
                fig_u, axes_u = plt.subplots(4, 1, figsize=(11,14))
                fig_u.suptitle(
                    f"Dekomposisi Komponen N-BEATSx — Data Upload "
                    f"({pd.to_datetime(future_dates[0]).strftime('%b %Y')} – "
                    f"{pd.to_datetime(future_dates[-1]).strftime('%b %Y')})",
                    fontsize=11, fontweight="bold", y=1.01)

                fd = [pd.to_datetime(d) for d in future_dates]

                ax0u = axes_u[0]
                ax0u.plot(hist_ds[-24:], hist_y[-24:]*100,
                          "o-", color="#63B3ED", lw=1.8, ms=4, label="Aktual")
                ax0u.plot(fd, du_tot*100,
                          "s--", color="#FC8181", lw=1.8, ms=5, label="Prediksi")
                ax0u.axvline(x=hist_ds[-1], color="#4A5568", lw=1, ls=":", alpha=0.8)
                ax0u.set_title("Aktual vs Prediksi", fontsize=10, pad=8)
                ax0u.set_ylabel("Inflasi (%)", fontsize=9)
                ax0u.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x:.2f}%"))
                ax0u.legend(fontsize=9, framealpha=.3,
                            facecolor="#1A202C", edgecolor="#2D3748")
                ax0u.grid(True, alpha=.4)

                for ax_, vals_, clr_, title_ in [
                    (axes_u[1], du_t, "#68D391", "Komponen Trend"),
                    (axes_u[2], du_s, "#F6AD55",
                     "Komponen Seasonality (Termasuk Efek Kalender)"),
                ]:
                    ax_.plot(fd, vals_, "o-", color=clr_, lw=1.8, ms=4)
                    ax_.fill_between(fd, vals_, 0, alpha=0.12, color=clr_)
                    ax_.axhline(y=0, color="#4A5568", lw=0.8, ls=":")
                    ax_.set_title(title_, fontsize=10, pad=8)
                    ax_.set_ylabel("Kontribusi", fontsize=9)
                    ax_.grid(True, alpha=.4)

                ax3u = axes_u[3]
                xlbls_u = [pd.to_datetime(d).strftime("%b %Y")
                           for d in future_dates]
                xpos_u = np.arange(len(xlbls_u))
                bars_u = ax3u.bar(xpos_u, du_e,
                                  color="#63B3ED", alpha=0.85,
                                  width=0.6, label="Eksogen")
                for bar_, val_ in zip(bars_u, du_e):
                    ax3u.text(
                        bar_.get_x()+bar_.get_width()/2,
                        bar_.get_height()+abs(du_e.max())*0.02,
                        f"{val_:.4f}", ha="center", va="bottom",
                        fontsize=7, color="#A0AEC0", fontfamily="monospace")
                ax3u.set_xticks(xpos_u)
                ax3u.set_xticklabels(xlbls_u, rotation=30, ha="right", fontsize=8)
                ax3u.set_title(
                    "Komponen Eksogen (BI Rate, Harga Minyak, Kurs, Lag)",
                    fontsize=10, pad=8)
                ax3u.set_ylabel("Kontribusi", fontsize=9)
                ax3u.set_xlabel("Periode Prediksi", fontsize=9)
                ax3u.legend(fontsize=9, framealpha=.3,
                            facecolor="#1A202C", edgecolor="#2D3748")
                ax3u.grid(True, alpha=.4, axis="y")

                for ax_ in axes_u[:3]:
                    ax_.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                    ax_.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                    plt.setp(ax_.xaxis.get_majorticklabels(),
                             rotation=30, ha="right", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig_u)
                plt.close()

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='section-header'>Tabel Dekomposisi</div>",
                    unsafe_allow_html=True)
                rows_u = ""
                for i_, d_ in enumerate(future_dates):
                    pr_t, pr_s, pr_e = props_u[i_]
                    rows_u += (
                        f"<tr>"
                        f"<td>{pd.to_datetime(d_).strftime('%B %Y')}</td>"
                        f"<td style='color:#63B3ED;font-weight:600;'>{du_tot[i_]*100:.4f}%</td>"
                        f"<td style='color:#68D391;'>{du_t[i_]:.4f}</td>"
                        f"<td style='color:#F6AD55;'>{du_s[i_]:.4f}</td>"
                        f"<td style='color:#63B3ED;'>{du_e[i_]:.4f}</td>"
                        f"<td style='color:#718096;'>"
                        f"T:{pr_t:.1f}% S:{pr_s:.1f}% E:{pr_e:.1f}%</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f"""<table class="pred-table">
                        <tr><th>Periode</th><th>Total Prediksi</th>
                        <th>Trend</th><th>Seasonality</th>
                        <th>Eksogen</th><th>Proporsi</th></tr>
                        {rows_u}
                    </table>""",
                    unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                dl_u = pd.DataFrame({
                    "Periode": [pd.to_datetime(d_).strftime("%Y-%m")
                                for d_ in future_dates],
                    "Total_%":        [f"{v*100:.4f}" for v in du_tot],
                    "Trend":          [f"{v:.6f}" for v in du_t],
                    "Seasonality":    [f"{v:.6f}" for v in du_s],
                    "Eksogen":        [f"{v:.6f}" for v in du_e],
                    "Pct_Trend":      [f"{p[0]:.2f}" for p in props_u],
                    "Pct_Seasonality":[f"{p[1]:.2f}" for p in props_u],
                    "Pct_Eksogen":    [f"{p[2]:.2f}" for p in props_u],
                })
                st.download_button(
                    "⬇️ Download Tabel Dekomposisi (CSV)",
                    dl_u.to_csv(index=False).encode("utf-8"),
                    file_name="dekomposisi_upload.csv", mime="text/csv")
            else:
                st.warning("⚠️ " + dc.get("error","Dekomposisi gagal."))

    # ════════════════════════════════════════════════════════
    # TAB 4 — ANALISIS MODEL
    # ════════════════════════════════════════════════════════
    with tab3:
        if _default:
            # Metrik tetap dari penelitian
            st.markdown(
                "<div class='section-header'>Performa Model pada Data Uji (Data Bawaan)</div>",
                unsafe_allow_html=True)
            ma1, ma2, ma3 = st.columns(3)
            for col_, (lbl, val, desc) in zip(
                [ma1, ma2, ma3],
                [("MAE",   "0.00601", "Mean Absolute Error"),
                 ("RMSE",  "0.00834", "Root Mean Squared Error"),
                 ("SMAPE", "41.76%",  "Symmetric MAPE")]
            ):
                with col_:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{lbl}</div>
                        <div class="metric-value">{val}</div>
                        <div class="metric-sub">{desc}</div>
                    </div>""", unsafe_allow_html=True)
        else:
            # Hitung metrik dari data upload
            st.markdown(
                "<div class='section-header'>Performa Prediksi pada Data Upload</div>",
                unsafe_allow_html=True)
            # Bandingkan pred_vals dengan hist_y terakhir (overlap jika ada)
            last_actuals = hist_y[-config["h"]:]
            last_dates_a = hist_ds[-config["h"]:]
            if len(last_actuals) == config["h"]:
                mae_u  = np.mean(np.abs(last_actuals - pred_vals))
                rmse_u = np.sqrt(np.mean((last_actuals - pred_vals)**2))
                denom  = (np.abs(last_actuals) + np.abs(pred_vals))
                denom  = np.where(denom < 1e-6, 1.0, denom)
                smape_u = np.mean(2*np.abs(last_actuals - pred_vals)/denom)*100
                ma1u, ma2u, ma3u = st.columns(3)
                for col_, (lbl, val, desc) in zip(
                    [ma1u, ma2u, ma3u],
                    [("MAE",   f"{mae_u:.5f}",  "Mean Absolute Error"),
                     ("RMSE",  f"{rmse_u:.5f}", "Root Mean Squared Error"),
                     ("SMAPE", f"{smape_u:.2f}%","Symmetric MAPE")]
                ):
                    with col_:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{lbl}</div>
                            <div class="metric-value">{val}</div>
                            <div class="metric-sub">{desc}</div>
                        </div>""", unsafe_allow_html=True)
                st.markdown(
                    f"""<div class="info-box" style="margin-top:.75rem;">
                        Metrik dihitung dari perbandingan prediksi masa depan
                        terhadap {config["h"]} observasi terakhir data upload
                        ({last_dates_a[0].strftime("%b %Y")} –
                        {last_dates_a[-1].strftime("%b %Y")}).
                    </div>""",
                    unsafe_allow_html=True)
            else:
                st.info("Data terlalu pendek untuk menghitung metrik evaluasi.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-header'>Perbandingan Model (Data Penelitian)</div>",
            unsafe_allow_html=True)
        comp = [
            ("N-BEATSx + BO ★","0.00601","0.00834","41.76%", True),
            ("Prophet",         "0.00487","0.00592","43.96%", False),
            ("SARIMAX",         "0.00717","0.00905","46.40%", False),
            ("N-BEATS",         "0.01039","0.01223","62.34%", False),
        ]
        rows_c = ""
        for mdl, mae, rmse, smape, is_ours in comp:
            s = "color:#63B3ED;font-weight:700;" if is_ours else "color:#A0AEC0;"
            r = "background:#0d1a2e;" if is_ours else ""
            rows_c += (
                f"<tr style='{r}'>"
                f"<td style='{s}'>{mdl}</td>"
                f"<td style='{s}'>{mae}</td>"
                f"<td style='{s}'>{rmse}</td>"
                f"<td style='{s}'>{smape}</td></tr>"
            )
        st.markdown(
            f"""<table class="pred-table">
                <tr><th>Model</th><th>MAE ↓</th>
                <th>RMSE ↓</th><th>SMAPE ↓</th></tr>
                {rows_c}
            </table>
            <div style="font-size:.73rem;color:#4A5568;margin-top:6px;">
                ★ = model yang dikembangkan · ↓ = semakin kecil semakin baik
            </div>""",
            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div class='section-header'>Uji Asumsi Residual (Data Penelitian)</div>",
            unsafe_allow_html=True)
        r1c, r2c = st.columns(2)
        for i_, (test, stat, result) in enumerate([
            ("Rata-rata Residual", "ē = −0.004927", "Mendekati nol ✅"),
            ("Shapiro-Wilk",       "W=0.9152, p=0.2489", "Normal ✅"),
            ("Ljung-Box (lag=6)",  "Q=5.2282, p=0.5149", "Tidak autokorelasi ✅"),
            ("Breusch-Pagan",      "LM=8.8344, p=0.2648", "Homoskedastis ✅"),
        ]):
            with (r1c if i_ % 2 == 0 else r2c):
                st.markdown(f"""
                <div class="metric-card" style="text-align:left;margin-bottom:.8rem;">
                    <div class="metric-label">{test}</div>
                    <div style="font-family:Space Mono,monospace;
                         font-size:.82rem;color:#A0AEC0;margin:.3rem 0;">
                         {stat}</div>
                    <div style="color:#68D391;font-size:.8rem;
                         font-weight:600;">{result}</div>
                </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # TAB 5 — PANDUAN
    # ════════════════════════════════════════════════════════
    with tab4:
        st.markdown(
            "<div class='section-header'>Panduan Penggunaan</div>",
            unsafe_allow_html=True)
        st.markdown(
            """<div class="info-box">
                <b>Format File:</b> CSV (.csv) atau Excel (.xlsx)<br><br>
                <b>Kolom yang Diperlukan:</b><br>
                • <code>Date / ds / Tanggal</code> — Tanggal (YYYY-MM-DD)<br>
                • <code>Inflasi Umum / y / Inflasi</code> — Nilai inflasi desimal<br>
                • <code>BI Rate</code> — Suku bunga kebijakan (desimal)<br>
                • <code>Harga Minyak Dunia</code> — USD per barel<br>
                • <code>Kurs USD/IDR</code> — Nilai tukar rupiah<br><br>
                <b>Frekuensi:</b> Bulanan &nbsp;·&nbsp;
                <b>Minimum:</b> 36 baris (3 tahun)
            </div>
            <div class="warning-box" style="margin-top:1rem;">
                <b>Catatan:</b> Prediksi bersifat indikatif berdasarkan pola
                historis. Model tidak dapat mengantisipasi kejadian ekstrem
                yang tidak terwakili dalam data pelatihan.
            </div>""",
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════
page = st.session_state.page

if   page == 'home':        page_home()
elif page == 'upload':      page_upload()
elif page == 'visualisasi': page_visualisasi()
elif page == 'prediksi':    page_prediksi()
elif page == 'about':       page_about()
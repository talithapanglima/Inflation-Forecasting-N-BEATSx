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
    Dekomposisi prediksi N-BEATSx menjadi komponen trend, seasonality, eksogen.
    Menggunakan forward hook pada setiap block untuk menangkap forecast per blok.
    """
    import torch

    try:
        model = nf.models[0]
        model.eval()

        # ── Tangkap output per blok via forward hook ─────────────
        block_forecasts_captured = []

        def _hook(module, inp, out):
            # out = (backcast, forecast) — forecast shape (B, h, 1)
            _, fc = out
            block_forecasts_captured.append(fc.detach().cpu())

        hooks = []
        for blk in model.blocks:
            hooks.append(blk.register_forward_hook(_hook))

        # Jalankan predict — hook akan mengisi block_forecasts_captured
        _ = nf.predict(df=df_scaled, futr_df=fut_df)

        # Lepas semua hook
        for h_ in hooks:
            h_.remove()

        if len(block_forecasts_captured) == 0:
            return {"success": False,
                    "error": "Tidak ada output blok tertangkap"}

        n_trend  = 3   # blok 0,1,2 = trend stack
        n_season = 3   # blok 3,4,5 = seasonality stack
        h_out    = model.h

        # Jumlahkan forecast per stack (shape setiap blok: (1, h, 1))
        trend_scaled  = sum(
            block_forecasts_captured[i].squeeze(-1)
            for i in range(min(n_trend, len(block_forecasts_captured)))
        )  # (1, h)

        season_scaled = sum(
            block_forecasts_captured[i].squeeze(-1)
            for i in range(n_trend,
                           min(n_trend + n_season,
                               len(block_forecasts_captured)))
        )  # (1, h)

        # Jalankan predict sekali lagi untuk dapat total
        forecast_df  = nf.predict(df=df_scaled, futr_df=fut_df)
        # Lepas hook yg terpasang ulang (tidak ada, predict sudah selesai)

        total_scaled_np = forecast_df[["NBEATSx"]].values.flatten()
        trend_np        = trend_scaled.numpy().flatten()
        season_np       = season_scaled.numpy().flatten()
        exog_np         = total_scaled_np - trend_np - season_np

        # Inverse transform — scaler adalah StandardScaler/MinMaxScaler linear
        # inv(a - b) ≠ inv(a) - inv(b) untuk scaler dengan mean/std
        # Cara benar: inverse total, lalu hitung proporsi lalu skala ulang
        def inv(arr):
            return scaler_y.inverse_transform(
                arr.reshape(-1, 1)).flatten()

        total_orig  = inv(total_scaled_np)
        trend_orig  = inv(trend_np)
        season_orig = inv(season_np)

        # Eksogen dalam skala asli:
        # total_orig = trend_orig + season_orig + exog_orig (approx, karena bias scaler)
        # Gunakan proporsi scaled untuk estimasi eksogen
        exog_orig = total_orig - trend_orig - season_orig

        # Koreksi: pastikan jumlah = total (floating point)
        return {
            "trend":       trend_orig,
            "seasonality": season_orig,
            "exogenous":   exog_orig,
            "total":       total_orig,
            "success":     True,
        }

    except Exception as e:
        import traceback
        return {"success": False, "error": str(e) + "\n" + traceback.format_exc()}


def render_decomp_tab(decomp, future_dates, label="Prediksi"):
    """Render tab dekomposisi dengan grafik dan tabel."""
    if not decomp.get("success"):
        st.warning(
            f"⚠️ Dekomposisi tidak tersedia: {decomp.get('error','unknown')}"
        )
        return

    trend_v  = decomp["trend"]
    season_v = decomp["seasonality"]
    exog_v   = decomp["exogenous"]
    total_v  = decomp["total"]
    dates    = [pd.to_datetime(d) for d in future_dates]

    # ── Proporsi ─────────────────────────────────────────────────
    props = []
    for t, s, e in zip(trend_v, season_v, exog_v):
        total_abs = abs(t) + abs(s) + abs(e)
        if total_abs > 1e-10:
            props.append((abs(t)/total_abs*100,
                          abs(s)/total_abs*100,
                          abs(e)/total_abs*100))
        else:
            props.append((33.3, 33.3, 33.3))

    avg_t = np.mean([p[0] for p in props])
    avg_s = np.mean([p[1] for p in props])
    avg_e = np.mean([p[2] for p in props])

    # ── Metrik proporsi ──────────────────────────────────────────
    dc1, dc2, dc3 = st.columns(3)
    for col, (lbl, val, color) in zip(
        [dc1, dc2, dc3],
        [("Trend",       f"{avg_t:.1f}%", "#68D391"),
         ("Seasonality", f"{avg_s:.1f}%", "#F6AD55"),
         ("Eksogen",     f"{avg_e:.1f}%", "#63B3ED")]
    ):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-label">Proporsi {lbl}</div>
                    <div class="metric-value" style="color:{color};">{val}</div>
                    <div class="metric-sub">Rata-rata 6 bulan</div>
                </div>""",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Grafik dekomposisi ────────────────────────────────────────
    set_dark_style()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(
        f"Dekomposisi Komponen N-BEATSx — {label}",
        fontsize=11, color="#E8EAF0", fontfamily="monospace", y=1.01
    )

    pairs = [
        (axes[0], trend_v,  "#68D391", "Trend"),
        (axes[1], season_v, "#F6AD55", "Seasonality"),
        (axes[2], exog_v,   "#63B3ED", "Eksogen"),
    ]
    for ax, vals, color, title in pairs:
        ax.plot(dates, vals * 100,
                color=color, linewidth=2,
                marker="o", markersize=5, label=title)
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
        ax.set_title(title, fontsize=9,
                     color=color, pad=5, fontfamily="monospace")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="#4A5568", linewidth=0.8, linestyle="--")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.2f}%"))

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(),
             rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabel dekomposisi ─────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>Tabel Dekomposisi</div>",
        unsafe_allow_html=True
    )
    rows_d = ""
    for i, d in enumerate(dates):
        t_pct = trend_v[i]  * 100
        s_pct = season_v[i] * 100
        e_pct = exog_v[i]   * 100
        total_pct = total_v[i] * 100
        pr_t, pr_s, pr_e = props[i]
        rows_d += (
            f"<tr>"
            f"<td>{d.strftime('%B %Y')}</td>"
            f"<td style='color:#63B3ED;font-weight:600;'>{total_pct:.4f}%</td>"
            f"<td style='color:#68D391;'>{t_pct:.4f}%</td>"
            f"<td style='color:#F6AD55;'>{s_pct:.4f}%</td>"
            f"<td style='color:#63B3ED;'>{e_pct:.4f}%</td>"
            f"<td style='color:#718096;font-size:0.78rem;'>"
            f"T:{pr_t:.1f}% S:{pr_s:.1f}% E:{pr_e:.1f}%</td>"
            f"</tr>"
        )
    st.markdown(
        f"""<table class="pred-table">
            <tr>
                <th>Periode</th>
                <th>Total</th>
                <th>Trend</th>
                <th>Seasonality</th>
                <th>Eksogen</th>
                <th>Proporsi</th>
            </tr>
            {rows_d}
        </table>""",
        unsafe_allow_html=True
    )

    # Rata-rata
    st.markdown(
        f"""<div style="background:#1A202C;border:1px solid #2D3748;
                    border-radius:8px;padding:0.75rem 1rem;margin-top:0.75rem;
                    font-size:0.82rem;color:#A0AEC0;">
            <b style="color:#E8EAF0;">Rata-rata Kontribusi:</b>
            &nbsp; Trend: <b style="color:#68D391;">{np.mean(trend_v)*100:.4f}%</b>
            &nbsp;·&nbsp; Seasonality: <b style="color:#F6AD55;">{np.mean(season_v)*100:.4f}%</b>
            &nbsp;·&nbsp; Eksogen: <b style="color:#63B3ED;">{np.mean(exog_v)*100:.4f}%</b>
            &nbsp;·&nbsp; Proporsi Eksogen: <b style="color:#63B3ED;">{avg_e:.1f}%</b>
        </div>""",
        unsafe_allow_html=True
    )

    # Download tabel
    dl_decomp = pd.DataFrame({
        "Periode":     [d.strftime("%Y-%m") for d in dates],
        "Total_%":     [f"{v*100:.4f}" for v in total_v],
        "Trend_%":     [f"{v*100:.4f}" for v in trend_v],
        "Seasonality_%": [f"{v*100:.4f}" for v in season_v],
        "Eksogen_%":   [f"{v*100:.4f}" for v in exog_v],
        "Proporsi_Trend_%":      [f"{p[0]:.2f}" for p in props],
        "Proporsi_Seasonality_%":[f"{p[1]:.2f}" for p in props],
        "Proporsi_Eksogen_%":    [f"{p[2]:.2f}" for p in props],
    })
    st.download_button(
        "⬇️ Download Tabel Dekomposisi (CSV)",
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
            <b>Frekuensi:</b> Bulanan (min. 13 baris)
        </div>
        """,
        unsafe_allow_html=True
    )

    c_up, c_fmt = st.columns([1.2, 1])

    with c_up:
        uploaded = st.file_uploader(
            "Pilih file CSV atau Excel",
            type=["csv", "xlsx"],
            label_visibility="collapsed"
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
                    st.success(
                        f"✅ Data berhasil diunggah — "
                        f"{len(raw)} baris · "
                        f"{raw['ds'].min().strftime('%b %Y')} – "
                        f"{raw['ds'].max().strftime('%b %Y')}"
                    )
            except Exception as e:
                st.error(f"❌ Gagal membaca file: {e}")
                st.session_state.uploaded_df = None

        if st.session_state.uploaded_df is not None:
            df_show = st.session_state.uploaded_df
            show_cols = [c for c in
                ["ds","y","BI Rate","Harga Minyak Dunia","Kurs USD/IDR"]
                if c in df_show.columns]
            st.dataframe(
                df_show[show_cols].tail(8).style.format({
                    "y": "{:.4f}", "BI Rate": "{:.4f}",
                    "Harga Minyak Dunia": "{:.2f}", "Kurs USD/IDR": "{:.0f}",
                }),
                use_container_width=True, height=260
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
                st.session_state.uploaded_df   = None
                st.session_state.upload_status = None
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
# PAGE: PREDIKSI INFLASI
# ═══════════════════════════════════════════════════════════════════
def page_prediksi():
    st.markdown("""
    <div class='main-header'>
        <div class='main-title'>📈 Prediksi Inflasi</div>
        <div class='main-subtitle'>
            Hasil prediksi 6 bulan ke depan — Model N-BEATSx + Bayesian Optimization
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        nf, scaler_y, scaler_exog, best, config, full_data_scaled, full_data_raw = load_artifacts()
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return

    _is_default_pred = st.session_state.uploaded_df is None
    use_data_raw_pred = (st.session_state.uploaded_df
                         if not _is_default_pred
                         else full_data_raw)
    data_src   = "Upload" if not _is_default_pred else "Data Bawaan Model"

    st.markdown(f"""
    <div class='info-box'>
        📌 Sumber data: <b>{data_src}</b> ·
        {len(use_data_raw_pred)} observasi
    </div>""", unsafe_allow_html=True)

    try:
        if not _is_default_pred:
            # Upload: build features dari raw lalu scale
            df_feat = build_features(
                use_data_raw_pred[['ds', 'y', 'BI Rate',
                                   'Harga Minyak Dunia', 'Kurs USD/IDR']].copy(),
                config)
            num_cols  = config['num_cols']
            df_scaled = scale_df(df_feat, scaler_y, scaler_exog, num_cols)
        else:
            # Bawaan: full_data_scaled sudah siap
            df_scaled = full_data_scaled.copy()
            df_feat   = full_data_raw.copy()

        last_date = pd.to_datetime(df_feat['ds'].max())
        fut_dummy = make_future_dummy(last_date, config['h'], config)

        forecast    = nf.predict(df=df_scaled, futr_df=fut_dummy)
        pred_vals   = scaler_y.inverse_transform(
            forecast[['NBEATSx']]).flatten()
        future_dates = forecast['ds'].values
        # hist_y dari raw data (sudah dalam skala asli)
        hist_y  = df_feat['y'].values
        hist_ds = pd.to_datetime(df_feat['ds'].values)
        data_ok = True
    except Exception as e:
        st.error(f"❌ Error prediksi: {e}")
        data_ok = False

    if not data_ok:
        return

    # Metrik ringkasan
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Prediksi Bulan Pertama</div>
            <div class='metric-value'>{pred_vals[0]*100:.2f}%</div>
            <div class='metric-sub'>
                {pd.to_datetime(future_dates[0]).strftime('%b %Y')}
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Rata-rata 6 Bulan</div>
            <div class='metric-value'>{np.mean(pred_vals)*100:.2f}%</div>
            <div class='metric-sub'>Mean prediksi</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        trend_dir = "↑" if pred_vals[-1] > pred_vals[0] else "↓"
        trend_col = "#68D391" if pred_vals[-1] > pred_vals[0] else "#FC8181"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Arah Tren</div>
            <div class='metric-value' style='color:{trend_col};'>
                {trend_dir}
            </div>
            <div class='metric-sub'>
                {pred_vals[0]*100:.2f}% → {pred_vals[-1]*100:.2f}%
            </div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Horizon Prediksi</div>
            <div class='metric-value'>6</div>
            <div class='metric-sub'>Bulan ke depan</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab_decomp, tab3, tab4 = st.tabs([
        "📊 Grafik Prediksi", "📋 Tabel Hasil",
        "🧩 Dekomposisi", "🔍 Analisis Model", "ℹ️ Panduan"
    ])

    with tab1:
        set_dark_style()
        fig, ax = plt.subplots(figsize=(12, 4.5))
        n_show    = 24
        hist_show = hist_y[-n_show:]
        ds_show   = hist_ds[-n_show:]

        ax.plot(ds_show, hist_show,
                color='#63B3ED', linewidth=1.8,
                marker='o', markersize=3, label='Aktual', zorder=3)
        ax.plot([ds_show[-1], future_dates[0]],
                [hist_show[-1], pred_vals[0]],
                color='#F6AD55', linewidth=1.5,
                linestyle='--', alpha=0.6)
        ax.plot(future_dates, pred_vals,
                color='#F6AD55', linewidth=2,
                marker='s', markersize=5,
                label='Prediksi N-BEATSx', zorder=4)
        ax.fill_between(future_dates,
                        pred_vals * 0.85, pred_vals * 1.15,
                        alpha=0.12, color='#F6AD55')
        for d, v in zip(future_dates, pred_vals):
            ax.annotate(f'{v*100:.2f}%', xy=(d, v),
                        xytext=(0, 12), textcoords='offset points',
                        fontsize=7.5, color='#F6AD55',
                        ha='center', fontfamily='monospace')
        ax.axvline(x=pd.to_datetime(last_date),
                   color='#4A5568', linewidth=1, linestyle=':', alpha=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        ax.set_ylabel('Inflasi (%)')
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
        ax.legend(fontsize=9, framealpha=0.3,
                  facecolor='#1A202C', edgecolor='#2D3748')
        ax.grid(True, alpha=0.4)
        ax.set_title('Prediksi Inflasi Indonesia — N-BEATSx',
                     fontsize=11, pad=12, color='#E8EAF0',
                     fontfamily='monospace')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if data_src == "Data Bawaan Model":
            st.markdown("""
            <div class='info-box'>
                ℹ️ Menggunakan data historis bawaan model.
                Upload file di halaman <b>Upload Data</b> untuk
                prediksi dengan data terbaru Anda.
            </div>""", unsafe_allow_html=True)

    with tab2:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("<div class='section-header'>Prediksi 6 Bulan ke Depan</div>",
                        unsafe_allow_html=True)
            rows = ""
            for i, (d, v) in enumerate(zip(future_dates, pred_vals)):
                pct   = v * 100
                color = "#68D391" if pct < 3 else \
                        "#F6AD55" if pct < 5 else "#FC8181"
                cat   = 'Rendah ✓' if pct < 3 else \
                        'Moderat' if pct < 5 else 'Tinggi ⚠'
                rows += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{pd.to_datetime(d).strftime('%B %Y')}</td>
                    <td style='color:{color};font-weight:600;'>{pct:.4f}%</td>
                    <td style='color:{color};'>{cat}</td>
                </tr>"""
            st.markdown(f"""
            <table class='pred-table'>
                <tr><th>#</th><th>Periode</th>
                    <th>Prediksi</th><th>Kategori</th></tr>
                {rows}
            </table>""", unsafe_allow_html=True)

        with col_b:
            st.markdown("<div class='section-header'>Data Historis Terakhir (12 Obs)</div>",
                        unsafe_allow_html=True)
            rows_h = ""
            for d, v in zip(hist_ds[-12:], hist_y[-12:]):
                rows_h += f"""
                <tr>
                    <td>{pd.to_datetime(d).strftime('%b %Y')}</td>
                    <td>{v*100:.4f}%</td>
                </tr>"""
            st.markdown(f"""
            <table class='pred-table'>
                <tr><th>Periode</th><th>Inflasi Aktual</th></tr>
                {rows_h}
            </table>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        dl_df = pd.DataFrame({
            'Periode': [pd.to_datetime(d).strftime('%Y-%m')
                        for d in future_dates],
            'Prediksi_Inflasi': [f"{v*100:.4f}" for v in pred_vals]
        })
        st.download_button(
            "⬇️ Download Hasil Prediksi (CSV)",
            dl_df.to_csv(index=False).encode('utf-8'),
            file_name="prediksi_inflasi.csv",
            mime="text/csv"
        )

    with tab_decomp:
        st.markdown(
            "<div class='section-header'>Dekomposisi Komponen Prediksi</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            """<div class="info-box">
                Dekomposisi membagi prediksi N-BEATSx menjadi tiga komponen:
                <b style="color:#68D391;">Trend</b> (blok 0–2),
                <b style="color:#F6AD55;">Seasonality</b> (blok 3–5), dan
                <b style="color:#63B3ED;">Eksogen</b> (selisih total minus trend minus seasonality).
                Komponen eksogen mencerminkan kontribusi variabel BI Rate,
                harga minyak dunia, kurs USD/IDR, dan lag inflasi.
            </div>""",
            unsafe_allow_html=True
        )
        with st.spinner("Menghitung dekomposisi komponen…"):
            decomp_result = decompose_forecast(
                nf, df_scaled, fut_dummy, scaler_y
            )
        render_decomp_tab(
            decomp_result, future_dates,
            label=f"6 Bulan ke Depan ({data_src})"
        )

    with tab3:
        st.markdown("<div class='section-header'>Performa Model pada Data Uji</div>",
                    unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        metrics = [
            ("MAE",   "0.00601", "Mean Absolute Error"),
            ("RMSE",  "0.00834", "Root Mean Squared Error"),
            ("SMAPE", "41.76%",  "Symmetric MAPE"),
        ]
        for col, (label, val, desc) in zip([m1, m2, m3], metrics):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-value'>{val}</div>
                    <div class='metric-sub'>{desc}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Perbandingan Model</div>",
                    unsafe_allow_html=True)
        comp_data = {
            'Model':  ['N-BEATSx + BO ★',
                       'Prophet', 'SARIMAX', 'N-BEATS'],
            'MAE':    ['0.00601', '0.00487', '0.00717', '0.01039'],
            'RMSE':   ['0.00834', '0.00592', '0.00905', '0.01223'],
            'SMAPE':  ['41.76%',  '43.96%',  '46.40%',  '62.34%'],
        }
        rows_c = ""
        for i in range(len(comp_data['Model'])):
            bold = "font-weight:700;color:#63B3ED;" \
                   if '★' in comp_data['Model'][i] else ""
            rows_c += f"""
            <tr>
                <td style='{bold}'>{comp_data['Model'][i]}</td>
                <td style='{bold}'>{comp_data['MAE'][i]}</td>
                <td style='{bold}'>{comp_data['RMSE'][i]}</td>
                <td style='{bold}'>{comp_data['SMAPE'][i]}</td>
            </tr>"""
        st.markdown(f"""
        <table class='pred-table'>
            <tr><th>Model</th><th>MAE</th><th>RMSE</th><th>SMAPE</th></tr>
            {rows_c}
        </table>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='warning-box'>
            ⚠️ SMAPE yang relatif tinggi pada seluruh model dipengaruhi oleh
            anomali deflasi Februari 2025 (inflasi = −0.09%).
            Pada kondisi inflasi normal, SMAPE model berkisar 6–25%.
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Hasil Uji Asumsi Residual</div>",
                    unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        residual_tests = [
            ("Rata-rata Residual", "ē = −0.004927", "Mendekati nol", True),
            ("Shapiro-Wilk",       "W=0.9152, p=0.2489", "Normal", True),
            ("Ljung-Box (lag=6)",  "Q=5.2282, p=0.5149", "Tidak autokorelasi", True),
            ("Breusch-Pagan",      "LM=8.8344, p=0.2648", "Homoskedastis", True),
        ]
        for i, (test, stat, result, passed) in enumerate(residual_tests):
            col = r1 if i % 2 == 0 else r2
            icon  = "✅" if passed else "❌"
            color = "#68D391" if passed else "#FC8181"
            with col:
                st.markdown(f"""
                <div class='metric-card' style='text-align:left;
                     margin-bottom:0.8rem;'>
                    <div class='metric-label'>{test}</div>
                    <div style='font-family:Space Mono,monospace;
                         font-size:0.82rem;color:#A0AEC0;
                         margin:0.3rem 0;'>{stat}</div>
                    <div style='color:{color};font-size:0.8rem;
                         font-weight:600;'>{icon} {result}</div>
                </div>""", unsafe_allow_html=True)

    with tab4:
        st.markdown("<div class='section-header'>Panduan Penggunaan</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <b>Format File yang Diterima:</b> CSV (.csv) atau Excel (.xlsx)<br><br>
            <b>Kolom yang Diperlukan:</b><br>
            • <code>Date / ds / Tanggal</code> — Tanggal (YYYY-MM-DD)<br>
            • <code>Inflasi Umum / y / Inflasi</code> — Nilai inflasi desimal<br>
            • <code>BI Rate</code> — Suku bunga kebijakan (desimal)<br>
            • <code>Harga Minyak Dunia</code> — USD per barel<br>
            • <code>Kurs USD/IDR</code> — Nilai tukar rupiah<br><br>
            <b>Frekuensi:</b> Bulanan · <b>Minimum:</b> 13 baris
        </div>
        <div class='warning-box' style='margin-top:1rem;'>
            <b>Catatan:</b> Prediksi bersifat indikatif berdasarkan pola
            historis. Model tidak dapat mengantisipasi kejadian ekstrem
            yang tidak terwakili dalam data pelatihan.
        </div>""", unsafe_allow_html=True)


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


# ═══════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════
page = st.session_state.page

if   page == 'home':        page_home()
elif page == 'upload':      page_upload()
elif page == 'visualisasi': page_visualisasi()
elif page == 'prediksi':    page_prediksi()
elif page == 'about':       page_about()
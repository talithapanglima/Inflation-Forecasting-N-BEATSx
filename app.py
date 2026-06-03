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
    page_title="INFLASI.AI — Prediksi Inflasi Indonesia",
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
    nf          = NeuralForecast.load('saved_model/nf_final')
    with open('saved_model/scaler_y.pkl',    'rb') as f: scaler_y    = pickle.load(f)
    with open('saved_model/scaler_exog.pkl', 'rb') as f: scaler_exog = pickle.load(f)
    with open('saved_model/best_params.pkl', 'rb') as f: best        = pickle.load(f)
    with open('saved_model/config.json',     'r')  as f: config      = json.load(f)
    full_data   = pd.read_parquet('saved_model/full_data.parquet')
    return nf, scaler_y, scaler_exog, best, config, full_data


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
            INFLASI.AI
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

    # Model info
    try:
        _, _, _, best, config, _ = load_artifacts()
        st.markdown("<div class='section-header'>Model Info</div>",
                    unsafe_allow_html=True)
        params = [
            ('input_size',  best['input_size']),
            ('hidden_size', best['hidden_size']),
            ('n_blocks',    str(best['n_blocks'])),
            ('max_steps',   best['max_steps']),
            ('dropout',     best['dropout']),
            ('lr',          f"{best['lr']:.5f}"),
        ]
        rows_p = "".join([
            f"<div class='val-row'>"
            f"<span class='val-key'>{k}</span>"
            f"<span class='val-val'>{v}</span></div>"
            for k, v in params
        ])
        st.markdown(f"<div class='about-card'>{rows_p}</div>",
                    unsafe_allow_html=True)
    except:
        st.warning("Model belum tersedia")


# ═══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════
def page_home():
    st.markdown("""
    <div class='main-header'>
        <div class='main-title'>📈 INFLASI.AI</div>
        <div class='main-subtitle'>
            Sistem Prediksi Inflasi Indonesia berbasis N-BEATSx
            dengan Bayesian Optimization
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        Selamat datang di <b>INFLASI.AI</b> — sistem prediksi inflasi Indonesia
        yang dikembangkan menggunakan model <i>deep learning</i> N-BEATSx
        yang dioptimasi dengan Bayesian Optimization dua tahap.
        Sistem ini mampu mengintegrasikan variabel makroekonomi dan efek
        kalender hari besar keagamaan dalam menghasilkan prediksi inflasi
        hingga 6 bulan ke depan.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Fitur Utama</div>",
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    features = [
        ('📂', 'Upload Data',
         'Unggah data historis inflasi dan variabel makroekonomi '
         'dalam format CSV atau Excel untuk memulai prediksi.'),
        ('📊', 'Visualisasi Data',
         'Eksplorasi pola historis inflasi dan variabel pendukung '
         'melalui grafik interaktif sebelum melakukan prediksi.'),
        ('📈', 'Prediksi Inflasi',
         'Hasil prediksi 6 bulan ke depan disajikan dalam bentuk '
         'grafik dan tabel, dilengkapi interval kepercayaan ±15%.'),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], features):
        with col:
            st.markdown(f"""
            <div class='feature-card'>
                <div class='feature-icon'>{icon}</div>
                <div class='feature-title'>{title}</div>
                <div class='feature-desc'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c4, c5 = st.columns(2)
    features2 = [
        ('🔬', 'Analisis Model',
         'Lihat metrik evaluasi model (MAE, RMSE, SMAPE), '
         'perbandingan dengan model baseline, dan hasil uji '
         'asumsi residual secara lengkap.'),
        ('👥', 'About Us',
         'Informasi latar belakang penelitian, tujuan pengembangan '
         'sistem, dan manfaat penerapan model prediksi inflasi '
         'dalam mendukung pengambilan kebijakan.'),
    ]
    for col, (icon, title, desc) in zip([c4, c5], features2):
        with col:
            st.markdown(f"""
            <div class='feature-card'>
                <div class='feature-icon'>{icon}</div>
                <div class='feature-title'>{title}</div>
                <div class='feature-desc'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Performa Model</div>",
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    kpi = [
        ('MAE', '0.00601', 'Data Uji'),
        ('RMSE', '0.00834', 'Data Uji'),
        ('SMAPE', '41.76%', 'Data Uji'),
        ('Horizon', '6 Bulan', 'Prediksi ke depan'),
    ]
    for col, (label, val, sub) in zip([m1, m2, m3, m4], kpi):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{val}</div>
                <div class='metric-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='warning-box'>
        ℹ️ <b>Cara Mulai:</b> Klik <b>Upload Data</b> di menu navigasi kiri
        untuk mengunggah data, kemudian buka halaman <b>Prediksi Inflasi</b>
        untuk melihat hasil prediksi. Jika tidak ada data yang diunggah,
        sistem akan menggunakan data historis bawaan model secara otomatis.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PAGE: UPLOAD DATA
# ═══════════════════════════════════════════════════════════════════
def page_upload():
    st.markdown("""
    <div class='main-header'>
        <div class='main-title'>📂 Upload Data</div>
        <div class='main-subtitle'>
            Unggah dataset inflasi dan variabel eksogen untuk prediksi
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("<div class='section-header'>Unggah File</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <b>Format yang didukung:</b> CSV (.csv) dan Excel (.xlsx)<br>
            <b>Ukuran maksimal:</b> 200 MB per file<br>
            <b>Frekuensi data:</b> Bulanan (minimum 13 baris untuk lag-12)
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Pilih file CSV atau Excel",
            type=['csv', 'xlsx'],
            label_visibility='collapsed'
        )

        if uploaded is not None:
            try:
                if uploaded.name.endswith('.csv'):
                    raw = pd.read_csv(uploaded)
                else:
                    raw = pd.read_excel(uploaded)

                raw = normalize_columns(raw)

                if not validate_columns(raw):
                    missing = {'ds', 'y', 'BI Rate',
                               'Harga Minyak Dunia',
                               'Kurs USD/IDR'} - set(raw.columns)
                    st.markdown(f"""
                    <div class='error-box'>
                        ❌ <b>Kolom tidak lengkap.</b><br>
                        Kolom yang belum ditemukan: {', '.join(missing)}<br>
                        Periksa nama kolom dan coba lagi.
                    </div>""", unsafe_allow_html=True)
                    st.session_state.uploaded_df = None
                    st.session_state.upload_status = 'error'
                else:
                    raw['ds'] = pd.to_datetime(raw['ds'])
                    raw = raw.sort_values('ds').reset_index(drop=True)
                    st.session_state.uploaded_df = raw
                    st.session_state.upload_status = 'ok'
                    st.markdown(f"""
                    <div class='success-box'>
                        ✅ <b>Data berhasil diunggah!</b><br>
                        {len(raw)} baris ·
                        {pd.to_datetime(raw['ds'].min()).strftime('%b %Y')} –
                        {pd.to_datetime(raw['ds'].max()).strftime('%b %Y')}
                    </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div class='error-box'>
                    ❌ Gagal membaca file: {e}
                </div>""", unsafe_allow_html=True)
                st.session_state.uploaded_df = None

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Format Kolom yang Diperlukan</div>",
                    unsafe_allow_html=True)

        cols_info = [
            ('ds / Date / Tanggal', 'Tanggal', 'YYYY-MM-DD atau DD/MM/YYYY', True),
            ('y / Inflasi / Inflasi Umum', 'Target', 'Desimal (contoh: 0.0265 = 2.65%)', True),
            ('BI Rate', 'Eksogen', 'Suku bunga kebijakan (desimal)', True),
            ('Harga Minyak Dunia', 'Eksogen', 'Harga dalam USD per barel', True),
            ('Kurs USD/IDR', 'Eksogen', 'Nilai tukar Rupiah terhadap USD', True),
        ]
        rows_col = ""
        for name, tipe, fmt, req in cols_info:
            badge_col = "badge-blue" if tipe == 'Target' else \
                        "badge-green" if tipe == 'Eksogen' else "badge-yellow"
            req_mark = "✱" if req else ""
            rows_col += f"""
            <tr>
                <td><code style='color:#63B3ED;font-size:0.78rem;'>{name}</code>
                    {req_mark}</td>
                <td><span class='badge {badge_col}'>{tipe}</span></td>
                <td style='color:#718096;font-size:0.78rem;'>{fmt}</td>
            </tr>"""
        st.markdown(f"""
        <table class='pred-table'>
            <tr><th>Nama Kolom</th><th>Tipe</th><th>Format</th></tr>
            {rows_col}
        </table>
        <div style='font-size:0.72rem;color:#4A5568;margin-top:6px;'>
            ✱ Wajib ada · Nama kolom tidak peka huruf besar/kecil
        </div>""", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='section-header'>Preview Data</div>",
                    unsafe_allow_html=True)

        if st.session_state.uploaded_df is not None:
            df_show = st.session_state.uploaded_df
            show_cols = [c for c in
                ['ds', 'y', 'BI Rate', 'Harga Minyak Dunia', 'Kurs USD/IDR']
                if c in df_show.columns]
            st.dataframe(
                df_show[show_cols].tail(10).style.format({
                    'y': '{:.4f}',
                    'BI Rate': '{:.4f}',
                    'Harga Minyak Dunia': '{:.2f}',
                    'Kurs USD/IDR': '{:.0f}',
                }),
                use_container_width=True, height=320
            )
            st.markdown(f"""
            <div class='val-row' style='margin-top:8px;'>
                <span class='val-key'>Total observasi</span>
                <span class='val-val'>{len(df_show)}</span>
            </div>
            <div class='val-row'>
                <span class='val-key'>Periode awal</span>
                <span class='val-val'>
                    {pd.to_datetime(df_show['ds'].min()).strftime('%B %Y')}
                </span>
            </div>
            <div class='val-row'>
                <span class='val-key'>Periode akhir</span>
                <span class='val-val'>
                    {pd.to_datetime(df_show['ds'].max()).strftime('%B %Y')}
                </span>
            </div>
            <div class='val-row'>
                <span class='val-key'>Nilai inflasi terbaru</span>
                <span class='val-val'>
                    {df_show['y'].iloc[-1]*100:.4f}%
                </span>
            </div>""", unsafe_allow_html=True)

            if st.button("🗑️ Hapus data & gunakan data bawaan"):
                st.session_state.uploaded_df = None
                st.session_state.upload_status = None
                st.rerun()
        else:
            st.markdown("""
            <div style='background:#1A202C;border:1px dashed #2D3748;
                        border-radius:12px;padding:3rem;text-align:center;'>
                <div style='font-size:2.5rem;margin-bottom:0.5rem;'>📋</div>
                <div style='color:#4A5568;font-size:0.85rem;'>
                    Belum ada data diunggah.<br>
                    Upload file di sebelah kiri.
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='warning-box'>
            <b>Catatan:</b> Jika tidak ada data yang diunggah, sistem akan
            menggunakan data historis bawaan model (Jan 2010 – Sep 2025)
            secara otomatis untuk menghasilkan prediksi.
        </div>""", unsafe_allow_html=True)


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
        _, scaler_y, scaler_exog, _, config, full_data = load_artifacts()
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return

    use_data = st.session_state.uploaded_df \
               if st.session_state.uploaded_df is not None \
               else full_data
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
        nf, scaler_y, scaler_exog, best, config, full_data = load_artifacts()
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return

    use_data   = st.session_state.uploaded_df \
                 if st.session_state.uploaded_df is not None \
                 else full_data
    data_src   = "Upload" if st.session_state.uploaded_df is not None \
                 else "Data Bawaan Model"

    st.markdown(f"""
    <div class='info-box'>
        📌 Sumber data: <b>{data_src}</b> ·
        {len(use_data)} observasi
    </div>""", unsafe_allow_html=True)

    try:
        df_feat = build_features(
            use_data[['ds', 'y', 'BI Rate',
                      'Harga Minyak Dunia', 'Kurs USD/IDR']].copy(),
            config)
        num_cols  = config['num_cols']
        df_scaled = scale_df(df_feat, scaler_y, scaler_exog, num_cols)
        last_date = df_feat['ds'].max()
        fut_dummy = make_future_dummy(last_date, config['h'], config)

        forecast    = nf.predict(df=df_scaled, futr_df=fut_dummy)
        pred_vals   = scaler_y.inverse_transform(
            forecast[['NBEATSx']]).flatten()
        future_dates = forecast['ds'].values
        hist_y  = scaler_y.inverse_transform(df_scaled[['y']]).flatten()
        hist_ds = df_feat['ds'].values
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

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Grafik Prediksi", "📋 Tabel Hasil",
        "🔍 Analisis Model", "ℹ️ Panduan"
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
            'Model':  ['N-BEATSx + BO ★', 'N-BEATSx Default',
                       'Prophet', 'SARIMAX', 'N-BEATS'],
            'MAE':    ['0.00601', '0.00530', '0.00487', '0.00717', '0.01039'],
            'RMSE':   ['0.00834', '0.00726', '0.00592', '0.00905', '0.01223'],
            'SMAPE':  ['41.76%',  '40.72%',  '43.96%',  '46.40%',  '62.34%'],
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
        ('Python 3.11', 'badge-blue'),
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
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Inflasi Indonesia",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: #0F1117;
        color: #E8EAF0;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
        border: 1px solid #2D3748;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .main-title {
        font-family: 'Space Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #63B3ED;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        color: #718096;
        font-size: 0.9rem;
        margin-top: 0.3rem;
        font-weight: 400;
    }

    /* Metric Cards */
    .metric-card {
        background: #1A202C;
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #63B3ED; }
    .metric-label {
        font-size: 0.75rem;
        color: #718096;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 1.6rem;
        font-weight: 700;
        color: #63B3ED;
    }
    .metric-sub {
        font-size: 0.72rem;
        color: #4A5568;
        margin-top: 0.2rem;
    }

    /* Section Headers */
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        color: #63B3ED;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border-bottom: 1px solid #2D3748;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Info Box */
    .info-box {
        background: #1A202C;
        border-left: 3px solid #63B3ED;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #A0AEC0;
        margin: 0.5rem 0;
    }

    /* Warning Box */
    .warning-box {
        background: #1A202C;
        border-left: 3px solid #F6AD55;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #A0AEC0;
        margin: 0.5rem 0;
    }

    /* Prediction Table */
    .pred-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    .pred-table th {
        background: #2D3748;
        color: #A0AEC0;
        font-weight: 600;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 0.6rem 0.8rem;
        text-align: left;
    }
    .pred-table td {
        padding: 0.55rem 0.8rem;
        border-bottom: 1px solid #1A202C;
        color: #E8EAF0;
        font-family: 'Space Mono', monospace;
        font-size: 0.82rem;
    }
    .pred-table tr:nth-child(even) td { background: #1A202C; }
    .pred-table tr:hover td { background: #2D3748; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0D1117;
        border-right: 1px solid #2D3748;
    }

    /* Buttons */
    .stButton > button {
        background: #2B6CB0;
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 600;
        font-size: 0.875rem;
        padding: 0.5rem 1.5rem;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #3182CE; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #1A202C;
        border: 1px dashed #2D3748;
        border-radius: 12px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1A202C;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #718096;
        border-radius: 7px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #2D3748 !important;
        color: #63B3ED !important;
    }

    /* Divider */
    hr { border-color: #2D3748; }

    /* Success/error messages */
    .stSuccess { background: #1C3A2A; border-color: #48BB78; }
    .stError   { background: #3A1C1C; border-color: #FC8181; }
</style>
""", unsafe_allow_html=True)

# ── Load Model & Config ──────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    from neuralforecast import NeuralForecast
    nf         = NeuralForecast.load('saved_model/nf_final')
    with open('saved_model/scaler_y.pkl',    'rb') as f: scaler_y    = pickle.load(f)
    with open('saved_model/scaler_exog.pkl', 'rb') as f: scaler_exog = pickle.load(f)
    with open('saved_model/best_params.pkl', 'rb') as f: best        = pickle.load(f)
    with open('saved_model/config.json',     'r')  as f: config      = json.load(f)
    full_data  = pd.read_parquet('saved_model/full_data.parquet')
    return nf, scaler_y, scaler_exog, best, config, full_data

# ── Helper Functions ─────────────────────────────────────────────
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
    df['y']        = scaler_y.transform(df[['y']])
    df[num_cols]   = scaler_exog.transform(df[num_cols])
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

def smape_fn(a, p):
    denom = (np.abs(a) + np.abs(p)) / 2
    return np.mean(np.where(denom < 1e-8, 0, np.abs(a-p)/denom)) * 100

def set_dark_style():
    plt.rcParams.update({
        'figure.facecolor'  : '#0F1117',
        'axes.facecolor'    : '#1A202C',
        'axes.edgecolor'    : '#2D3748',
        'axes.labelcolor'   : '#A0AEC0',
        'xtick.color'       : '#718096',
        'ytick.color'       : '#718096',
        'grid.color'        : '#2D3748',
        'grid.linestyle'    : '--',
        'grid.alpha'        : 0.6,
        'text.color'        : '#E8EAF0',
        'font.family'       : 'monospace',
    })

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem'>
        <div style='font-family:Space Mono,monospace;font-size:1rem;
                    font-weight:700;color:#63B3ED;'>INFLASI.AI</div>
        <div style='font-size:0.75rem;color:#4A5568;margin-top:2px;'>
            N-BEATSx + Bayesian Optimization
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='section-header'>Input Data</div>",
                unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload CSV / Excel",
        type=['csv','xlsx'],
        help="File harus memiliki kolom: Date/ds, Inflasi Umum/y, "
             "BI Rate, Harga Minyak Dunia, Kurs USD/IDR"
    )

    st.divider()
    st.markdown("<div class='section-header'>Model Info</div>",
                unsafe_allow_html=True)

    try:
        _, _, _, best, config, _ = load_artifacts()
        st.markdown(f"""
        <div class='info-box'>
            <b>Best Params:</b><br>
            input_size &nbsp;= {best['input_size']}<br>
            hidden_size = {best['hidden_size']}<br>
            n_blocks &nbsp;&nbsp;= {best['n_blocks']}<br>
            max_steps &nbsp;= {best['max_steps']}<br>
            dropout &nbsp;&nbsp;&nbsp;= {best['dropout']}<br>
            lr &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= {best['lr']:.5f}
        </div>
        """, unsafe_allow_html=True)
    except:
        st.warning("Model belum tersedia")

    st.divider()
    st.markdown("""
    <div style='font-size:0.72rem;color:#4A5568;line-height:1.6;'>
        Model: N-BEATSx + Bayesian Optimization<br>
        Data: Jan 2010 – Sep 2025 (189 obs)<br>
        Horizon: 6 bulan ke depan<br>
        Variabel: BI Rate, Harga Minyak, Kurs,<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dummy Kalender
    </div>
    """, unsafe_allow_html=True)

# ── Main Header ──────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <div class='main-title'>📈 Prediksi Inflasi Indonesia</div>
    <div class='main-subtitle'>
        Model N-BEATSx dengan Bayesian Optimization — 
        Prediksi 6 Bulan ke Depan
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load Artifacts ───────────────────────────────────────────────
try:
    nf, scaler_y, scaler_exog, best, config, full_data = load_artifacts()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ── Process Data ─────────────────────────────────────────────────
if uploaded is not None:
    try:
        if uploaded.name.endswith('.csv'):
            raw = pd.read_csv(uploaded)
        else:
            raw = pd.read_excel(uploaded)

        # Normalisasi nama kolom
        col_map = {}
        for c in raw.columns:
            cl = c.lower().strip()
            if cl in ['date','ds','tanggal']:         col_map[c] = 'ds'
            elif cl in ['inflasi umum','inflasi','y']: col_map[c] = 'y'
            elif 'minyak' in cl:                       col_map[c] = 'Harga Minyak Dunia'
            elif 'bi rate' in cl or 'bi_rate' in cl:  col_map[c] = 'BI Rate'
            elif 'kurs' in cl or 'usd' in cl:          col_map[c] = 'Kurs USD/IDR'
        raw = raw.rename(columns=col_map)
        use_data = raw
        data_source = "upload"
    except Exception as e:
        st.error(f"❌ Error membaca file: {e}")
        use_data = full_data
        data_source = "default"
else:
    use_data = full_data
    data_source = "default"

# ── Build Features & Scale ───────────────────────────────────────
try:
    df_feat = build_features(use_data[['ds','y',
        'BI Rate','Harga Minyak Dunia','Kurs USD/IDR']].copy(),
        config)

    num_cols = config['num_cols']
    df_scaled = scale_df(df_feat, scaler_y, scaler_exog, num_cols)

    last_date    = df_feat['ds'].max()
    future_dummy = make_future_dummy(last_date, config['h'], config)

    # Prediksi
    forecast     = nf.predict(df=df_scaled, futr_df=future_dummy)
    pred_vals    = scaler_y.inverse_transform(
        forecast[['NBEATSx']]).flatten()
    future_dates = forecast['ds'].values

    # Historis (skala asli)
    hist_y  = scaler_y.inverse_transform(
        df_scaled[['y']]).flatten()
    hist_ds = df_feat['ds'].values

    data_ok = True
except Exception as e:
    st.error(f"❌ Error prediksi: {e}")
    data_ok = False

# ── Tabs ─────────────────────────────────────────────────────────
if data_ok:
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Prediksi", "📋 Tabel Hasil",
        "🔍 Analisis Model", "ℹ️ Panduan"
    ])

    # ── TAB 1: Prediksi ──────────────────────────────────────────
    with tab1:
        # Metrik ringkasan
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Prediksi Bulan Pertama</div>
                <div class='metric-value'>{pred_vals[0]*100:.2f}%</div>
                <div class='metric-sub'>{pd.to_datetime(future_dates[0]).strftime('%b %Y')}</div>
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
                <div class='metric-value' style='color:{trend_col}'>{trend_dir}</div>
                <div class='metric-sub'>
                    {pred_vals[0]*100:.2f}% → {pred_vals[-1]*100:.2f}%
                </div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Horizon Prediksi</div>
                <div class='metric-value'>6</div>
                <div class='metric-sub'>Bulan ke depan</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Plot utama
        set_dark_style()
        fig, ax = plt.subplots(figsize=(12, 4.5))

        # Historis — 24 bulan terakhir
        n_show    = 24
        hist_show = hist_y[-n_show:]
        ds_show   = hist_ds[-n_show:]

        ax.plot(ds_show, hist_show,
                color='#63B3ED', linewidth=1.8,
                marker='o', markersize=3, label='Aktual', zorder=3)

        # Garis transisi
        ax.plot([ds_show[-1], future_dates[0]],
                [hist_show[-1], pred_vals[0]],
                color='#F6AD55', linewidth=1.5,
                linestyle='--', alpha=0.6)

        # Prediksi
        ax.plot(future_dates, pred_vals,
                color='#F6AD55', linewidth=2,
                marker='s', markersize=5,
                label='Prediksi N-BEATSx', zorder=4)

        # Area prediksi
        ax.fill_between(future_dates, pred_vals * 0.85, pred_vals * 1.15,
                         alpha=0.12, color='#F6AD55')

        # Label prediksi
        for i, (d, v) in enumerate(zip(future_dates, pred_vals)):
            ax.annotate(f'{v*100:.2f}%',
                        xy=(d, v),
                        xytext=(0, 12), textcoords='offset points',
                        fontsize=7.5, color='#F6AD55',
                        ha='center', fontfamily='monospace')

        # Garis pemisah historis/prediksi
        ax.axvline(x=pd.to_datetime(last_date),
                   color='#4A5568', linewidth=1,
                   linestyle=':', alpha=0.8)
        ax.text(pd.to_datetime(last_date), ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.04,
                ' Data\n Historis', fontsize=7, color='#4A5568',
                ha='left', va='top')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
        ax.set_ylabel('Inflasi (%)', fontsize=9)
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

        if data_source == "default":
            st.markdown("""
            <div class='info-box'>
                ℹ️ Menggunakan data historis bawaan model. 
                Upload file CSV/Excel di sidebar untuk prediksi 
                dengan data Anda sendiri.
            </div>""", unsafe_allow_html=True)

    # ── TAB 2: Tabel ────────────────────────────────────────────
    with tab2:
        col_a, col_b = st.columns([1, 1])

        with col_a:
            st.markdown("<div class='section-header'>Prediksi 6 Bulan ke Depan</div>",
                        unsafe_allow_html=True)

            rows = ""
            for i, (d, v) in enumerate(zip(future_dates, pred_vals)):
                pct = v * 100
                color = "#68D391" if pct < 3 else \
                        "#F6AD55" if pct < 5 else "#FC8181"
                rows += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{pd.to_datetime(d).strftime('%B %Y')}</td>
                    <td style='color:{color};font-weight:600;'>
                        {pct:.4f}%
                    </td>
                    <td style='color:{color};'>
                        {'Rendah ✓' if pct < 3 else
                         'Moderat' if pct < 5 else 'Tinggi ⚠'}
                    </td>
                </tr>"""

            st.markdown(f"""
            <table class='pred-table'>
                <tr>
                    <th>#</th><th>Periode</th>
                    <th>Prediksi</th><th>Kategori</th>
                </tr>
                {rows}
            </table>""", unsafe_allow_html=True)

        with col_b:
            st.markdown("<div class='section-header'>Data Historis Terakhir (12 Obs)</div>",
                        unsafe_allow_html=True)

            rows_h = ""
            for d, v in zip(hist_ds[-12:], hist_y[-12:]):
                pct = v * 100
                rows_h += f"""
                <tr>
                    <td>{pd.to_datetime(d).strftime('%b %Y')}</td>
                    <td>{pct:.4f}%</td>
                </tr>"""

            st.markdown(f"""
            <table class='pred-table'>
                <tr><th>Periode</th><th>Inflasi Aktual</th></tr>
                {rows_h}
            </table>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # Download
        dl_df = pd.DataFrame({
            'Periode'         : [pd.to_datetime(d).strftime('%Y-%m') 
                                 for d in future_dates],
            'Prediksi_Inflasi': [f"{v*100:.4f}" for v in pred_vals]
        })
        st.download_button(
            "⬇️ Download Hasil Prediksi (CSV)",
            dl_df.to_csv(index=False).encode('utf-8'),
            file_name="prediksi_inflasi.csv",
            mime="text/csv"
        )

    # ── TAB 3: Analisis ─────────────────────────────────────────
    with tab3:
        st.markdown("<div class='section-header'>Performa Model pada Data Uji</div>",
                    unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        metrics = [
            ("MAE", "0.005687", "Mean Absolute Error"),
            ("RMSE", "0.008195", "Root Mean Squared Error"),
            ("SMAPE", "40.67%", "Symmetric MAPE"),
        ]
        for col, (label, val, desc) in zip([m1,m2,m3], metrics):
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
            'Model'    : ['N-BEATSx + BO ★', 'Prophet',
                          'SARIMAX', 'N-BEATS'],
            'MAE'      : ['0.00569', '0.00487', '0.00717', '0.01039'],
            'RMSE'     : ['0.00820', '0.00592', '0.00905', '0.01223'],
            'SMAPE'    : ['40.67%', '43.96%', '46.40%', '62.34%'],
        }
        comp_df = pd.DataFrame(comp_data)

        rows_c = ""
        for _, row in comp_df.iterrows():
            bold = "font-weight:700;color:#63B3ED;" \
                   if '★' in row['Model'] else ""
            rows_c += f"""
            <tr>
                <td style='{bold}'>{row['Model']}</td>
                <td style='{bold}'>{row['MAE']}</td>
                <td style='{bold}'>{row['RMSE']}</td>
                <td style='{bold}'>{row['SMAPE']}</td>
            </tr>"""

        st.markdown(f"""
        <table class='pred-table'>
            <tr><th>Model</th><th>MAE</th>
                <th>RMSE</th><th>SMAPE</th></tr>
            {rows_c}
        </table>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='warning-box'>
            ⚠️ SMAPE yang tinggi pada seluruh model dipengaruhi oleh 
            anomali deflasi Februari 2025 (inflasi = −0.09%). 
            Pada kondisi inflasi normal, SMAPE model berkisar 12–16%.
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Hasil Uji Asumsi Residual</div>",
                    unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        residual_tests = [
            ("Rata-rata Residual",  "0.003463", "Mendekati nol",   True),
            ("Shapiro-Wilk",        "W=0.9215, p=0.2987", "Normal", True),
            ("Ljung-Box (lag=6)",   "Q=10.30, p=0.1125",
             "Tidak autokorelasi",  True),
            ("Breusch-Pagan",       "LM=10.41, p=0.1665",
             "Homoskedastis",       True),
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

    # ── TAB 4: Panduan ──────────────────────────────────────────
    with tab4:
        st.markdown("<div class='section-header'>Panduan Penggunaan</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
            <b>Format File yang Diterima:</b> CSV atau Excel (.xlsx)<br><br>
            <b>Kolom yang Diperlukan:</b><br>
            • <code>Date</code> / <code>ds</code> / <code>Tanggal</code> — 
              Tanggal dalam format YYYY-MM-DD atau DD/MM/YYYY<br>
            • <code>Inflasi Umum</code> / <code>y</code> / 
              <code>Inflasi</code> — Nilai inflasi (desimal, contoh: 0.0265)<br>
            • <code>BI Rate</code> — Suku bunga kebijakan (desimal)<br>
            • <code>Harga Minyak Dunia</code> — USD per barel<br>
            • <code>Kurs USD/IDR</code> — Nilai tukar rupiah<br><br>
            <b>Frekuensi Data:</b> Bulanan (minimum 13 baris untuk lag-12)<br>
            <b>Format Nilai:</b> Desimal (0.0265 = 2.65%)
        </div>

        <div class='info-box' style='margin-top:1rem;'>
            <b>Tentang Model:</b><br>
            Model N-BEATSx dengan Bayesian Optimization dilatih 
            menggunakan data inflasi Indonesia Januari 2010 – 
            September 2025 (189 observasi). Model menggunakan 
            konfigurasi <i>interpretable</i> dengan trend stack 
            dan seasonality stack untuk menangkap komponen tren 
            jangka panjang dan pola musiman secara terpisah.
        </div>

        <div class='warning-box' style='margin-top:1rem;'>
            <b>Catatan:</b> Prediksi yang dihasilkan bersifat 
            indikatif berdasarkan pola historis dan variabel 
            eksogen yang tersedia. Model tidak dapat mengantisipasi 
            kejadian ekstrem (shock) yang tidak terwakili dalam 
            data pelatihan.
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class='warning-box'>
        ⚠️ Tidak dapat memproses data. 
        Pastikan file memiliki kolom yang sesuai.
    </div>""", unsafe_allow_html=True)
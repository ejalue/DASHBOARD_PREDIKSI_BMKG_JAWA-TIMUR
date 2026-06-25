"""
Dashboard Prediksi Hujan Jawa Timur
EAS — Pembelajaran Mesin
Institut Teknologi Sepuluh Nopember

Jalankan: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Hujan Jawa Timur",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS CUSTOM
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Font & base */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #f8f9fb;
    border: 1px solid #e8eaed;
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
[data-testid="metric-container"] label { font-size: 12px !important; color: #6b7280 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 600 !important; }

/* Section header */
.section-header {
    display: flex; align-items: center; gap: 10px;
    padding: 0.75rem 1rem;
    background: linear-gradient(90deg, #1e3a5f08, transparent);
    border-left: 3px solid #1e3a5f;
    border-radius: 0 8px 8px 0;
    margin-bottom: 1rem;
}
.section-header h3 { margin: 0; font-size: 15px; font-weight: 600; color: #1e3a5f; }

/* Info box */
.info-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 0.875rem 1rem;
    font-size: 13px;
    color: #1e40af;
    margin-bottom: 1rem;
}

/* Sidebar nav button style */
div[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    font-size: 13px;
    transition: background .15s;
}
div[data-testid="stSidebar"] .stButton > button:hover { background: #f1f5f9; }

/* Divider */
hr { border: none; border-top: 1px solid #e5e7eb; margin: 1.5rem 0; }

/* Chart container */
.chart-card {
    background: white;
    border: 1px solid #e8eaed;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA & MODEL (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Memuat data...")
def load_data():
    df = pd.read_csv("climate_data_clean.csv", parse_dates=["date"])
    df = df[df["province_name"] == "Jawa Timur"].copy()
    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)
    df["rain"] = (df["RR"] > 0).astype(int)
    df["year"]  = df["date"].dt.year
    df["month_name"] = df["date"].dt.strftime("%b")
    df["month_order"] = df["date"].dt.month
    return df

@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    try:
        model    = joblib.load("model_xgboost_jatim.pkl")
        features = joblib.load("features_list.pkl")
        info     = joblib.load("model_info.pkl")
        return model, features, info
    except FileNotFoundError:
        return None, None, None

df = load_data()
model, features, model_info = load_model()

# Konstantan bulan
MONTHS_ID = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"Mei",6:"Jun",
             7:"Jul",8:"Agu",9:"Sep",10:"Okt",11:"Nov",12:"Des"}
BULAN_LIST = [MONTHS_ID[i] for i in range(1,13)]

# Warna tema
C_BLUE   = "#1e3a5f"
C_RAIN   = "#3b82f6"
C_DRY    = "#f59e0b"
C_GREEN  = "#10b981"
C_RED    = "#ef4444"

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <span style='font-size:36px;'>🌧️</span>
        <h2 style='margin:4px 0 0; font-size:16px; color:#1e3a5f; font-weight:600;'>
            Prediksi Hujan
        </h2>
        <p style='font-size:11px; color:#6b7280; margin:2px 0 0;'>Jawa Timur — Data BMKG</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    halaman = st.radio(
        "Navigasi",
        options=[
            "🏠  Beranda",
            "📊  Eksplorasi Data (EDA)",
            "🤖  Prediksi Hujan",
            "📈  Evaluasi Model",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Filter global (dipakai halaman EDA)
    st.markdown("**⚙️ Filter Global**")
    tahun_list = sorted(df["year"].unique())
    tahun_pilihan = st.multiselect(
        "Tahun",
        options=tahun_list,
        default=tahun_list[-5:],
        key="filter_tahun",
    )
    stasiun_list = sorted(df["station_name"].unique())
    stasiun_pilihan = st.multiselect(
        "Stasiun / Kota",
        options=stasiun_list,
        default=stasiun_list[:5],
        key="filter_stasiun",
    )

    st.markdown("---")
    st.markdown(
        "<p style='font-size:10px; color:#9ca3af; text-align:center;'>"
        "EAS Pembelajaran Mesin · ITS Vokasi<br>Data: BMKG 2010–2022</p>",
        unsafe_allow_html=True,
    )

# Filter dataframe sesuai pilihan sidebar
df_filtered = df.copy()
if tahun_pilihan:
    df_filtered = df_filtered[df_filtered["year"].isin(tahun_pilihan)]
if stasiun_pilihan:
    df_filtered = df_filtered[df_filtered["station_name"].isin(stasiun_pilihan)]


# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 1 — BERANDA
# ══════════════════════════════════════════════════════════════════════════════
if halaman == "🏠  Beranda":

    # Hero
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, {C_BLUE} 0%, #2563eb 100%);
        border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
        color: white;
    '>
        <div style='display:flex; align-items:center; gap:16px;'>
            <span style='font-size:52px;'>🌧️</span>
            <div>
                <h1 style='margin:0; font-size:24px; font-weight:700; color:white;'>
                    Dashboard Prediksi Hujan Jawa Timur
                </h1>
                <p style='margin:6px 0 0; font-size:14px; opacity:.85;'>
                    Klasifikasi hujan/tidak hujan berbasis data iklim BMKG
                    menggunakan XGBoost · EAS Pembelajaran Mesin ITS Vokasi
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric cards ──
    total_baris  = len(df)
    jml_stasiun  = df["station_id"].nunique()
    pct_hujan    = df["rain"].mean() * 100
    rentang_thn  = f"{df['year'].min()}–{df['year'].max()}"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📍 Stasiun BMKG",   f"{jml_stasiun}",          "Jawa Timur")
    c2.metric("📋 Total Observasi", f"{total_baris:,}",        "baris data")
    c3.metric("🗓️ Rentang Data",    rentang_thn,               "2010–2022")
    c4.metric("🌧️ Hari Hujan",     f"{pct_hujan:.1f}%",        "dari total hari")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Peta stasiun ──
    st.markdown("""
    <div class='section-header'>
        <span>🗺️</span><h3>Peta Sebaran Stasiun BMKG di Jawa Timur</h3>
    </div>
    """, unsafe_allow_html=True)

    df_stasiun = df.groupby(
        ["station_id", "station_name", "region_name", "latitude", "longitude"]
    ).agg(
        pct_hujan=("rain", lambda x: round(x.mean()*100, 1)),
        n_obs=("rain", "count"),
    ).reset_index()

    fig_map = px.scatter_mapbox(
        df_stasiun,
        lat="latitude", lon="longitude",
        hover_name="station_name",
        hover_data={"region_name": True, "pct_hujan": True,
                    "n_obs": True, "latitude": False, "longitude": False},
        color="pct_hujan",
        color_continuous_scale="Blues",
        size="n_obs",
        size_max=18,
        zoom=7,
        center={"lat": -7.5, "lon": 112.5},
        mapbox_style="open-street-map",
        labels={"pct_hujan": "% Hari Hujan", "n_obs": "Jumlah Obs."},
        height=420,
    )
    fig_map.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        coloraxis_colorbar=dict(title="% Hari Hujan", thickness=12, len=0.7),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("""
    <div class='info-box'>
        💡 <b>Cara baca peta:</b> Ukuran titik = jumlah observasi; warna lebih gelap = persentase hari hujan
        lebih tinggi. Hover untuk detail tiap stasiun.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Distribusi target & ringkasan per stasiun ──
    col_l, col_r = st.columns([1, 1.6])

    with col_l:
        st.markdown("""
        <div class='section-header'>
            <span>🎯</span><h3>Distribusi Target</h3>
        </div>
        """, unsafe_allow_html=True)

        counts = df["rain"].value_counts().reset_index()
        counts.columns = ["rain", "count"]
        counts["label"] = counts["rain"].map({0: "Tidak Hujan", 1: "Hujan"})
        counts["pct"] = (counts["count"] / counts["count"].sum() * 100).round(1)

        fig_pie = go.Figure(go.Pie(
            labels=counts["label"],
            values=counts["count"],
            hole=0.55,
            marker_colors=[C_DRY, C_RAIN],
            textinfo="percent",
            textfont_size=13,
            hovertemplate="<b>%{label}</b><br>%{value:,} hari (%{percent})<extra></extra>",
        ))
        fig_pie.add_annotation(
            text=f"<b>{counts.loc[counts['rain']==1,'count'].values[0]:,}</b><br><span style='font-size:11px'>hari hujan</span>",
            x=0.5, y=0.5, showarrow=False, font_size=14, align="center",
        )
        fig_pie.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=260,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.markdown("""
        <div class='section-header'>
            <span>📋</span><h3>Ringkasan per Stasiun (Top 10)</h3>
        </div>
        """, unsafe_allow_html=True)

        df_sum = df.groupby("station_name").agg(
            Kabupaten=("region_name", "first"),
            Obs=("rain", "count"),
            Pct_Hujan=("rain", lambda x: round(x.mean()*100, 1)),
            RR_rata=("RR", lambda x: round(x.mean(), 1)),
            Tavg_rata=("Tavg", lambda x: round(x.mean(), 1)),
        ).reset_index().sort_values("Pct_Hujan", ascending=False).head(10)

        df_sum.columns = ["Stasiun", "Kabupaten", "Obs", "% Hujan", "RR rata² (mm)", "Tavg (°C)"]
        st.dataframe(
            df_sum,
            use_container_width=True,
            hide_index=True,
            height=265,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Statistik deskriptif ──
    st.markdown("""
    <div class='section-header'>
        <span>📐</span><h3>Statistik Deskriptif Variabel Meteorologi</h3>
    </div>
    """, unsafe_allow_html=True)

    cols_desc = ["Tn", "Tx", "Tavg", "RH_avg", "RR", "ss", "ff_avg"]
    labels_desc = {
        "Tn":"Suhu Min (°C)", "Tx":"Suhu Max (°C)", "Tavg":"Suhu Rata² (°C)",
        "RH_avg":"Kelembapan (%)", "RR":"Curah Hujan (mm)",
        "ss":"Sinar Matahari (jam)", "ff_avg":"Kec. Angin (m/s)",
    }
    desc = df[cols_desc].describe().T.round(2).reset_index()
    desc["index"] = desc["index"].map(labels_desc)
    desc.columns = ["Variabel", "N", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max"]
    st.dataframe(desc, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif halaman == "📊  Eksplorasi Data (EDA)":

    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
        border-radius: 12px; padding: 1.25rem 2rem; margin-bottom: 1.5rem; color: white;
    '>
        <h2 style='margin:0; font-size:20px; font-weight:700; color:white;'>
            📊 Eksplorasi Data (EDA)
        </h2>
        <p style='margin:4px 0 0; font-size:13px; opacity:.85;'>
            Analisis pola cuaca berdasarkan data yang dipilih di sidebar
        </p>
    </div>
    """, unsafe_allow_html=True)

    if df_filtered.empty:
        st.warning("⚠️ Tidak ada data untuk filter yang dipilih. Ubah filter di sidebar.")
        st.stop()

    # Info filter aktif
    st.markdown(
        f"<div class='info-box'>📌 Menampilkan <b>{len(df_filtered):,} baris</b> dari "
        f"<b>{df_filtered['station_name'].nunique()} stasiun</b> · "
        f"Tahun: <b>{', '.join(map(str, sorted(tahun_pilihan))) if tahun_pilihan else 'Semua'}</b></div>",
        unsafe_allow_html=True,
    )

    # ── TAB EDA ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈  Tren Curah Hujan",
        "📦  Distribusi per Bulan",
        "🌡️  Probabilitas Hujan",
        "🔗  Korelasi Fitur",
    ])

    # ─────────────────────────────────────────
    # TAB 1 — Tren Curah Hujan
    # ─────────────────────────────────────────
    with tab1:
        st.markdown("#### Tren Curah Hujan Bulanan")

        col_opt1, col_opt2 = st.columns([2, 1])
        with col_opt1:
            mode_agregasi = st.radio(
                "Tampilkan sebagai",
                ["Rata-rata harian", "Total bulanan"],
                horizontal=True,
                key="tren_mode",
            )
        with col_opt2:
            overlay_hujan = st.checkbox("Tampilkan % hari hujan", value=True, key="overlay_hujan")

        # Agregasi bulanan
        df_tren = df_filtered.copy()
        df_tren["year_month"] = df_tren["date"].dt.to_period("M").astype(str)

        if mode_agregasi == "Rata-rata harian":
            df_tren_agg = df_tren.groupby("year_month").agg(
                RR=("RR", "mean"), rain_pct=("rain", "mean")
            ).reset_index()
            y_label = "Curah Hujan Rata-rata (mm/hari)"
        else:
            df_tren_agg = df_tren.groupby("year_month").agg(
                RR=("RR", "sum"), rain_pct=("rain", "mean")
            ).reset_index()
            y_label = "Total Curah Hujan (mm)"

        df_tren_agg["RR"] = df_tren_agg["RR"].round(2)
        df_tren_agg["rain_pct_pct"] = (df_tren_agg["rain_pct"] * 100).round(1)

        if overlay_hujan:
            fig_tren = make_subplots(specs=[[{"secondary_y": True}]])
            fig_tren.add_trace(
                go.Bar(
                    x=df_tren_agg["year_month"], y=df_tren_agg["RR"],
                    name="Curah Hujan", marker_color=C_RAIN, opacity=0.75,
                    hovertemplate="%{x}<br>RR: %{y:.1f}<extra></extra>",
                ),
                secondary_y=False,
            )
            fig_tren.add_trace(
                go.Scatter(
                    x=df_tren_agg["year_month"], y=df_tren_agg["rain_pct_pct"],
                    name="% Hari Hujan", line=dict(color=C_RED, width=2),
                    hovertemplate="%{x}<br>% Hujan: %{y:.1f}%<extra></extra>",
                ),
                secondary_y=True,
            )
            fig_tren.update_yaxes(title_text=y_label, secondary_y=False)
            fig_tren.update_yaxes(title_text="% Hari Hujan", secondary_y=True, range=[0, 100])
        else:
            fig_tren = px.bar(
                df_tren_agg, x="year_month", y="RR",
                labels={"year_month": "Bulan", "RR": y_label},
                color_discrete_sequence=[C_RAIN],
            )

        fig_tren.update_layout(
            height=380, margin=dict(t=20, b=20, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        fig_tren.update_xaxes(
            tickangle=45,
            tickmode="array",
            tickvals=df_tren_agg["year_month"][::6].tolist(),
        )
        st.plotly_chart(fig_tren, use_container_width=True)

        # ── Rolling 12 bulan ──
        df_tren_agg["RR_roll12"] = df_tren_agg["RR"].rolling(12, min_periods=6).mean()

        fig_roll = px.line(
            df_tren_agg, x="year_month", y=["RR", "RR_roll12"],
            labels={"value": y_label, "year_month": "Bulan", "variable": ""},
            color_discrete_map={"RR": "#93c5fd", "RR_roll12": C_BLUE},
            height=300,
        )
        fig_roll.update_traces(
            selector={"name": "RR"}, opacity=0.4,
        )
        fig_roll.update_traces(
            selector={"name": "RR_roll12"}, line_width=2.5,
        )
        fig_roll.update_layout(
            margin=dict(t=30, b=20, l=10, r=10),
            title=dict(text="Rolling Average 12 Bulan", font_size=13),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig_roll.update_xaxes(tickangle=45, tickmode="array",
                               tickvals=df_tren_agg["year_month"][::12].tolist())
        st.plotly_chart(fig_roll, use_container_width=True)

    # ─────────────────────────────────────────
    # TAB 2 — Distribusi per Bulan
    # ─────────────────────────────────────────
    with tab2:
        st.markdown("#### Distribusi Variabel Meteorologi per Bulan")

        var_opts = {
            "Curah Hujan — RR (mm)": "RR",
            "Suhu Rata² — Tavg (°C)": "Tavg",
            "Kelembapan — RH_avg (%)": "RH_avg",
            "Kec. Angin — ff_avg (m/s)": "ff_avg",
            "Sinar Matahari — ss (jam)": "ss",
        }
        var_pilihan = st.selectbox("Pilih variabel", list(var_opts.keys()), key="var_box")
        col_var = var_opts[var_pilihan]

        df_box = df_filtered.copy()
        df_box["bulan_nama"] = df_box["month_order"].map(MONTHS_ID)
        df_box["bulan_nama"] = pd.Categorical(
            df_box["bulan_nama"], categories=BULAN_LIST, ordered=True
        )

        fig_box = px.box(
            df_box.sort_values("bulan_nama"),
            x="bulan_nama", y=col_var,
            color="bulan_nama",
            color_discrete_sequence=px.colors.sequential.Blues_r[::-1][:12],
            labels={"bulan_nama": "Bulan", col_var: var_pilihan},
            height=400,
        )
        fig_box.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=10, r=10),
            hovermode="x",
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Violin split hujan/tidak
        st.markdown("#### Perbandingan Distribusi: Hari Hujan vs Tidak Hujan")
        df_box["Status"] = df_box["rain"].map({0: "Tidak Hujan", 1: "Hujan"})

        fig_vio = px.violin(
            df_box, x="bulan_nama", y=col_var,
            color="Status",
            color_discrete_map={"Hujan": C_RAIN, "Tidak Hujan": C_DRY},
            box=True, points=False,
            labels={"bulan_nama": "Bulan", col_var: var_pilihan},
            height=380,
        )
        fig_vio.update_layout(
            margin=dict(t=20, b=20, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            violingap=0.2, violinmode="overlay",
        )
        st.plotly_chart(fig_vio, use_container_width=True)

    # ─────────────────────────────────────────
    # TAB 3 — Probabilitas Hujan
    # ─────────────────────────────────────────
    with tab3:
        st.markdown("#### Probabilitas Hujan per Bulan")

        prob_bulan = df_filtered.groupby("month_order").agg(
            prob=("rain", "mean"),
            rr_mean=("RR", "mean"),
            n=("rain", "count"),
        ).reset_index()
        prob_bulan["bulan"] = prob_bulan["month_order"].map(MONTHS_ID)
        prob_bulan["prob_pct"] = (prob_bulan["prob"] * 100).round(1)
        prob_bulan["rr_mean"] = prob_bulan["rr_mean"].round(1)
        prob_bulan["musim"] = prob_bulan["month_order"].apply(
            lambda m: "Musim Hujan" if m in [12,1,2] else
                      "Musim Kemarau" if m in [6,7,8] else "Peralihan"
        )

        color_musim = {
            "Musim Hujan": C_RAIN,
            "Peralihan": C_GREEN,
            "Musim Kemarau": C_DRY,
        }

        fig_prob = go.Figure()
        for musim, grp in prob_bulan.groupby("musim"):
            fig_prob.add_trace(go.Bar(
                x=grp["bulan"], y=grp["prob_pct"],
                name=musim,
                marker_color=color_musim[musim],
                hovertemplate="<b>%{x}</b><br>P(Hujan): %{y:.1f}%<extra></extra>",
                text=grp["prob_pct"].apply(lambda v: f"{v:.0f}%"),
                textposition="outside",
            ))

        fig_prob.update_layout(
            barmode="group",
            xaxis=dict(categoryorder="array", categoryarray=BULAN_LIST),
            yaxis=dict(title="Probabilitas Hujan (%)", range=[0, 110]),
            height=380, margin=dict(t=20, b=20, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # Heatmap prob per tahun-bulan
        st.markdown("#### Heatmap Probabilitas Hujan per Tahun dan Bulan")

        pivot_prob = df_filtered.groupby(["year", "month_order"])["rain"].mean().unstack()
        pivot_prob.columns = [MONTHS_ID[c] for c in pivot_prob.columns]
        pivot_prob = (pivot_prob * 100).round(1)

        fig_heat = px.imshow(
            pivot_prob,
            color_continuous_scale="Blues",
            labels={"color": "P(Hujan) %"},
            aspect="auto",
            height=max(250, len(pivot_prob) * 28),
            text_auto=".0f",
        )
        fig_heat.update_layout(
            margin=dict(t=20, b=20, l=10, r=10),
            xaxis_title="Bulan",
            yaxis_title="Tahun",
            coloraxis_colorbar=dict(title="P(Hujan) %", thickness=12),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("""
        <div class='info-box'>
            💡 <b>Interpretasi:</b> Warna lebih gelap = probabilitas hujan lebih tinggi.
            Pola musiman terlihat jelas dengan bulan Jan–Mar cenderung lebih basah
            dan Jun–Agu lebih kering di Jawa Timur.
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # TAB 4 — Korelasi
    # ─────────────────────────────────────────
    with tab4:
        st.markdown("#### Matriks Korelasi Variabel Meteorologi")

        num_cols = ["Tn", "Tx", "Tavg", "RH_avg", "RR", "ss", "ff_x", "ff_avg", "rain"]
        label_map = {
            "Tn":"Suhu Min", "Tx":"Suhu Max", "Tavg":"Suhu Rata²",
            "RH_avg":"Kelembapan", "RR":"Curah Hujan",
            "ss":"Sinar Matahari", "ff_x":"Angin Max",
            "ff_avg":"Angin Rata²", "rain":"Hujan (0/1)",
        }
        corr_df = df_filtered[num_cols].corr().round(3)
        corr_df.index   = [label_map[c] for c in corr_df.index]
        corr_df.columns = [label_map[c] for c in corr_df.columns]

        fig_corr = px.imshow(
            corr_df,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            text_auto=".2f",
            aspect="auto",
            height=480,
        )
        fig_corr.update_layout(
            margin=dict(t=20, b=20, l=10, r=10),
            coloraxis_colorbar=dict(title="Korelasi", thickness=12),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Bar chart korelasi terhadap target
        st.markdown("#### Korelasi Setiap Fitur terhadap Target (Hujan/Tidak)")

        target_corr = df_filtered[num_cols].corr()["rain"].drop("rain").sort_values()
        target_corr.index = [label_map.get(c, c) for c in target_corr.index]
        colors_bar = [C_RAIN if v > 0 else C_DRY for v in target_corr.values]

        fig_bar_corr = go.Figure(go.Bar(
            x=target_corr.values.round(3),
            y=target_corr.index,
            orientation="h",
            marker_color=colors_bar,
            hovertemplate="<b>%{y}</b><br>Korelasi: %{x:.3f}<extra></extra>",
            text=target_corr.values.round(3),
            textposition="outside",
        ))
        fig_bar_corr.add_vline(x=0, line_color="gray", line_width=1, line_dash="dash")
        fig_bar_corr.update_layout(
            height=380, margin=dict(t=20, b=20, l=10, r=10),
            xaxis_title="Korelasi Pearson",
            xaxis_range=[-0.55, 0.55],
        )
        st.plotly_chart(fig_bar_corr, use_container_width=True)

        st.markdown("""
        <div class='info-box'>
            🔵 <b>Biru</b> = korelasi positif dengan hujan (nilai tinggi → cenderung hujan) &nbsp;|&nbsp;
            🟡 <b>Kuning</b> = korelasi negatif (nilai tinggi → cenderung tidak hujan).
            Kelembapan (RH_avg) dan curah hujan sebelumnya (RR) adalah prediktor terkuat.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 3 — PREDIKSI HUJAN
# ══════════════════════════════════════════════════════════════════════════════
elif halaman == "🤖  Prediksi Hujan":

    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #1e1b4b 0%, #3730a3 100%);
        border-radius: 12px; padding: 1.25rem 2rem; margin-bottom: 1.5rem; color: white;
    '>
        <h2 style='margin:0; font-size:20px; font-weight:700; color:white;'>
            🤖 Prediksi Hujan — XGBoost Classifier
        </h2>
        <p style='margin:4px 0 0; font-size:13px; opacity:.85;'>
            Masukkan kondisi cuaca hari ini untuk memprediksi apakah besok akan hujan
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Cek model tersedia
    if model is None:
        st.error(
            "⚠️ **Model belum ditemukan.** Pastikan file berikut ada di folder yang sama dengan `app.py`:\n"
            "- `model_xgboost_jatim.pkl`\n- `features_list.pkl`\n- `model_info.pkl`\n\n"
            "Jalankan notebook EAS terlebih dahulu untuk menghasilkan file model."
        )
        st.stop()

    # ── Helper: buat satu baris input sesuai FEATURES ──
    def build_input_row(inputs: dict) -> pd.DataFrame:
        """Konversi dict input user ke DataFrame 1 baris sesuai urutan FEATURES."""
        row = {f: inputs.get(f, 0.0) for f in features}
        return pd.DataFrame([row])

    # ── Default nilai historis berdasar median stasiun yg dipilih ──
    @st.cache_data(show_spinner=False)
    def get_station_defaults(station_name: str) -> dict:
        sub = df[df["station_name"] == station_name]
        if sub.empty:
            sub = df
        return {
            "Tn":     float(sub["Tn"].median()),
            "Tx":     float(sub["Tx"].median()),
            "Tavg":   float(sub["Tavg"].median()),
            "RH_avg": float(sub["RH_avg"].median()),
            "ss":     float(sub["ss"].median()),
            "ff_x":   float(sub["ff_x"].median()),
            "ff_avg": float(sub["ff_avg"].median()),
            "ddd_x":  float(sub["ddd_x"].median()),
            "RR":     float(sub["RR"].median()),
        }

    # ── Layout: kiri = form, kanan = output ──
    col_form, col_out = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown("### 📝 Input Kondisi Cuaca")

        # Pilih stasiun → auto-fill default
        stasiun_pred = st.selectbox(
            "Pilih Stasiun / Kota",
            options=sorted(df["station_name"].unique()),
            key="pred_stasiun",
            help="Default nilai diisi dari median historis stasiun yang dipilih."
        )
        defaults = get_station_defaults(stasiun_pred)

        st.markdown("**🌡️ Suhu Udara**")
        c1, c2, c3 = st.columns(3)
        with c1:
            Tn = st.number_input("Min — Tn (°C)", 15.0, 35.0,
                                  float(round(defaults["Tn"], 1)), 0.1, key="Tn")
        with c2:
            Tx = st.number_input("Max — Tx (°C)", 20.0, 42.0,
                                  float(round(defaults["Tx"], 1)), 0.1, key="Tx")
        with c3:
            Tavg = st.number_input("Rata² — Tavg (°C)", 18.0, 38.0,
                                    float(round(defaults["Tavg"], 1)), 0.1, key="Tavg")

        st.markdown("**💧 Kelembapan & Hujan Sebelumnya**")
        c4, c5 = st.columns(2)
        with c4:
            RH_avg = st.slider("Kelembapan RH_avg (%)", 40, 100,
                                int(round(defaults["RH_avg"])), key="RH_avg")
        with c5:
            ss = st.slider("Sinar Matahari ss (jam)", 0.0, 12.0,
                            float(round(defaults["ss"], 1)), 0.1, key="ss")

        st.markdown("**💨 Angin**")
        c6, c7, c8 = st.columns(3)
        with c6:
            ff_avg = st.number_input("Kec. Rata² ff_avg (m/s)", 0.0, 20.0,
                                      float(round(defaults["ff_avg"], 1)), 0.1, key="ff_avg")
        with c7:
            ff_x = st.number_input("Kec. Max ff_x (m/s)", 0.0, 30.0,
                                    float(round(defaults["ff_x"], 1)), 0.1, key="ff_x")
        with c8:
            ddd_x = st.number_input("Arah Angin ddd_x (°)", 0.0, 360.0,
                                     float(round(defaults["ddd_x"])), 1.0, key="ddd_x")

        st.markdown("**🌧️ Curah Hujan Hari-Hari Sebelumnya**")
        c9, c10, c11 = st.columns(3)
        with c9:
            RR_lag1 = st.number_input("RR kemarin (mm)", 0.0, 300.0, 0.0, 0.5, key="RR_lag1")
        with c10:
            RR_lag2 = st.number_input("RR 2 hari lalu (mm)", 0.0, 300.0, 0.0, 0.5, key="RR_lag2")
        with c11:
            RR_lag3 = st.number_input("RR 3 hari lalu (mm)", 0.0, 300.0, 0.0, 0.5, key="RR_lag3")

        st.markdown("**💦 Kelembapan Hari Sebelumnya**")
        c12, c13, c14 = st.columns(3)
        with c12:
            RH_lag1 = st.number_input("RH kemarin (%)", 40.0, 100.0,
                                       float(round(defaults["RH_avg"])), 0.5, key="RH_lag1")
        with c13:
            RH_lag2 = st.number_input("RH 2 hari lalu (%)", 40.0, 100.0,
                                       float(round(defaults["RH_avg"])), 0.5, key="RH_lag2")
        with c14:
            RH_lag3 = st.number_input("RH 3 hari lalu (%)", 40.0, 100.0,
                                       float(round(defaults["RH_avg"])), 0.5, key="RH_lag3")

        st.markdown("**📅 Temporal**")
        c15, c16 = st.columns(2)
        with c15:
            month_in = st.selectbox("Bulan", options=list(MONTHS_ID.items()),
                                     format_func=lambda x: x[1], key="month_in")
            month_val = month_in[0]
        with c16:
            doy = st.number_input("Hari ke- (1–366)", 1, 366,
                                   int(pd.Timestamp.today().timetuple().tm_yday), key="doy")

        predict_btn = st.button("🔍  Prediksi Sekarang", use_container_width=True, type="primary")

    # ── Output prediksi ──
    with col_out:
        st.markdown("### 📊 Hasil Prediksi")

        if predict_btn:
            # Hitung fitur turunan
            wind_deg = ddd_x
            wind_sin_v = float(np.sin(np.radians(wind_deg)))
            wind_cos_v = float(np.cos(np.radians(wind_deg)))

            RR_roll3_v  = float(np.mean([RR_lag1, RR_lag2, RR_lag3]))
            RR_roll7_v  = RR_roll3_v
            RH_roll3_v  = float(np.mean([RH_lag1, RH_lag2, RH_lag3]))
            RH_roll7_v  = RH_roll3_v
            Tavg_roll3_v = Tavg
            Tavg_roll7_v = Tavg
            ff_roll3_v  = ff_avg
            ff_roll7_v  = ff_avg

            season_v = 1 if month_val in [12,1,2] else 2 if month_val in [6,7,8] else 3
            week_v   = int(pd.Timestamp(year=2024, month=month_val, day=15).isocalendar().week)
            rain_streak_v = int(RR_lag1 > 0) + int(RR_lag2 > 0) + int(RR_lag3 > 0)

            user_inputs = {
                "Tn": Tn, "Tx": Tx, "Tavg": Tavg, "RH_avg": RH_avg,
                "ss": ss, "ff_x": ff_x, "ff_avg": ff_avg, "ddd_x": ddd_x,
                "month": month_val, "day_of_year": doy, "week_of_year": week_v,
                "season": season_v,
                "wind_sin": wind_sin_v, "wind_cos": wind_cos_v,
                "RR_lag1": RR_lag1, "RR_lag2": RR_lag2, "RR_lag3": RR_lag3,
                "RH_lag1": RH_lag1, "RH_lag2": RH_lag2, "RH_lag3": RH_lag3,
                "RR_roll3": RR_roll3_v, "RR_roll7": RR_roll7_v,
                "RH_avg_roll3": RH_roll3_v, "RH_avg_roll7": RH_roll7_v,
                "Tavg_roll3": Tavg_roll3_v, "Tavg_roll7": Tavg_roll7_v,
                "ff_avg_roll3": ff_roll3_v, "ff_avg_roll7": ff_roll7_v,
                "rain_streak": rain_streak_v,
            }

            X_input = build_input_row(user_inputs)
            prob_hujan = float(model.predict_proba(X_input)[0][1])
            pred_label = 1 if prob_hujan >= 0.5 else 0
            prob_pct   = round(prob_hujan * 100, 1)

            # ── Card hasil ──
            if pred_label == 1:
                bg_col, icon, label_text, txt_col = "#eff6ff", "🌧️", "HUJAN", "#1d4ed8"
            else:
                bg_col, icon, label_text, txt_col = "#fffbeb", "☀️", "TIDAK HUJAN", "#b45309"

            st.markdown(f"""
            <div style='
                background:{bg_col}; border-radius:16px; padding:1.75rem;
                text-align:center; margin-bottom:1rem;
                border: 2px solid {"#bfdbfe" if pred_label==1 else "#fde68a"};
            '>
                <div style='font-size:60px; line-height:1;'>{icon}</div>
                <h2 style='margin:0.5rem 0 0; color:{txt_col}; font-size:22px; font-weight:700;'>
                    Prediksi: {label_text}
                </h2>
                <p style='margin:4px 0 0; color:{txt_col}; font-size:13px; opacity:.8;'>
                    Probabilitas hujan: <b>{prob_pct}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                number={"suffix": "%", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": C_RAIN if pred_label == 1 else C_DRY, "thickness": 0.25},
                    "steps": [
                        {"range": [0, 30],  "color": "#fef3c7"},
                        {"range": [30, 60], "color": "#dbeafe"},
                        {"range": [60, 100],"color": "#bfdbfe"},
                    ],
                    "threshold": {
                        "line": {"color": "#dc2626", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
                title={"text": "Probabilitas Hujan", "font": {"size": 13}},
            ))
            fig_gauge.update_layout(
                height=240, margin=dict(t=30, b=10, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Ringkasan input ──
            st.markdown("**📋 Ringkasan Input**")
            summary_data = {
                "Variabel": ["Tavg", "RH_avg", "ss", "ff_avg",
                              "RR lag-1", "RR lag-2", "RR lag-3",
                              "Rain streak", "Bulan", "Musim"],
                "Nilai": [
                    f"{Tavg:.1f} °C", f"{RH_avg} %", f"{ss:.1f} jam",
                    f"{ff_avg:.1f} m/s",
                    f"{RR_lag1:.1f} mm", f"{RR_lag2:.1f} mm", f"{RR_lag3:.1f} mm",
                    f"{rain_streak_v} hari",
                    MONTHS_ID[month_val],
                    ["—","Hujan","Kemarau","Peralihan"][season_v],
                ],
            }
            st.dataframe(
                pd.DataFrame(summary_data),
                use_container_width=True,
                hide_index=True,
                height=350,
            )

        else:
            # Tampilan awal sebelum prediksi
            st.markdown("""
            <div style='
                background:#f8fafc; border:2px dashed #cbd5e1;
                border-radius:16px; padding:3rem 2rem; text-align:center;
            '>
                <span style='font-size:48px;'>⬅️</span>
                <p style='color:#64748b; margin:0.75rem 0 0; font-size:14px;'>
                    Isi kondisi cuaca di sebelah kiri<br>lalu klik <b>Prediksi Sekarang</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class='info-box'>
                💡 <b>Tips pengisian:</b><br>
                • Pilih stasiun dulu — nilai default otomatis terisi dari median historis stasiun tersebut<br>
                • Kolom lag (RR kemarin, 2 hari lalu, dst.) sangat berpengaruh pada prediksi<br>
                • RH_avg > 80% + RR_lag1 > 0 cenderung menghasilkan prediksi hujan
            </div>
            """, unsafe_allow_html=True)

    # ── Catatan metodologi ──
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("📖 Catatan Metodologi Prediksi"):
        st.markdown(f"""
        **Model:** XGBoost Classifier  
        **Data training:** Jawa Timur, 2010–2022 (setelah SMOTE)  
        **Fitur yang digunakan:** {len(features)} fitur (meteorologi + lag + rolling + temporal)  
        **Threshold klasifikasi:** 0.5 (P ≥ 0.5 → Hujan)

        **Definisi target:**  
        - `Hujan (1)` = curah hujan RR > 0 mm  
        - `Tidak Hujan (0)` = curah hujan RR = 0 mm  

        **Catatan penting:** Prediksi didasarkan pada pola historis stasiun di Jawa Timur.
        Akurasi dapat bervariasi antar stasiun dan musim.
        """)
        if model_info:
            st.markdown(f"""
            **Performa model (test set):**  
            - ROC-AUC: `{model_info.get('roc_auc', 0):.4f}`  
            - CV ROC-AUC: `{model_info.get('cv_roc_auc_mean', 0):.4f} ± {model_info.get('cv_roc_auc_std', 0):.4f}`
            """)


# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN 4 — EVALUASI MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif halaman == "📈  Evaluasi Model":

    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #14532d 0%, #16a34a 100%);
        border-radius: 12px; padding: 1.25rem 2rem; margin-bottom: 1.5rem; color: white;
    '>
        <h2 style='margin:0; font-size:20px; font-weight:700; color:white;'>
            📈 Evaluasi Model & Interpretasi
        </h2>
        <p style='margin:4px 0 0; font-size:13px; opacity:.85;'>
            Performa XGBoost Classifier · Data Jawa Timur 2010–2022
        </p>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ Model belum ditemukan. Jalankan notebook training terlebih dahulu.")
        st.stop()

    # ── Metric cards performa ──
    st.markdown("""
    <div class='section-header'>
        <span>🏆</span><h3>Ringkasan Performa Model</h3>
    </div>
    """, unsafe_allow_html=True)

    roc_auc  = model_info.get("roc_auc", 0)
    cv_mean  = model_info.get("cv_roc_auc_mean", 0)
    cv_std   = model_info.get("cv_roc_auc_std", 0)
    n_train  = model_info.get("n_train", 0)
    n_test   = model_info.get("n_test", 0)
    n_feat   = len(features) if features else model_info.get("features", [])

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("ROC-AUC (test)", f"{roc_auc:.4f}",    "XGBoost")
    m2.metric("CV ROC-AUC",     f"{cv_mean:.4f}",    f"± {cv_std:.4f}")
    m3.metric("Training samples", f"{n_train:,}",    "setelah SMOTE")
    m4.metric("Test samples",    f"{n_test:,}",       "20% hold-out")
    m5.metric("Fitur dipakai",   f"{len(features)}",  "features")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Hitung prediksi ulang di test set (subset data) ──
    @st.cache_data(show_spinner="Menghitung prediksi test set...")
    def compute_test_predictions():
        """Buat test set dari data aktual untuk evaluasi visual."""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            classification_report, confusion_matrix,
            roc_auc_score, roc_curve, precision_recall_curve,
        )

        df_eval = df.copy()

        # Feature engineering ulang (sama persis dengan notebook)
        df_eval["day_of_year"] = df_eval["date"].dt.dayofyear
        df_eval["week_of_year"] = df_eval["date"].dt.isocalendar().week.astype(int)
        df_eval["season"] = df_eval["month_order"].apply(
            lambda m: 1 if m in [12,1,2] else 2 if m in [6,7,8] else 3
        )
        wind_dir_map = {
            "N":0,"NNE":22.5,"NE":45,"ENE":67.5,"E":90,"ESE":112.5,
            "SE":135,"SSE":157.5,"S":180,"SSW":202.5,"SW":225,"WSW":247.5,
            "W":270,"WNW":292.5,"NW":315,"NNW":337.5,"C":0,
        }
        df_eval["ddd_car_deg"] = df_eval["ddd_car"].map(wind_dir_map).fillna(0)
        df_eval["wind_sin"] = np.sin(np.radians(df_eval["ddd_car_deg"]))
        df_eval["wind_cos"] = np.cos(np.radians(df_eval["ddd_car_deg"]))

        for lag in [1, 2, 3]:
            df_eval[f"RR_lag{lag}"]  = df_eval.groupby("station_id")["RR"].shift(lag)
            df_eval[f"RH_lag{lag}"]  = df_eval.groupby("station_id")["RH_avg"].shift(lag)

        for col in ["RR", "RH_avg", "Tavg", "ff_avg"]:
            df_eval[f"{col}_roll3"] = df_eval.groupby("station_id")[col].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean()
            )
            df_eval[f"{col}_roll7"] = df_eval.groupby("station_id")[col].transform(
                lambda x: x.shift(1).rolling(7, min_periods=1).mean()
            )

        def rain_streak(s):
            streak = np.zeros(len(s), dtype=int)
            for i in range(1, len(s)):
                streak[i] = streak[i-1] + 1 if s.iloc[i-1] > 0 else 0
            return pd.Series(streak, index=s.index)

        df_eval["rain_streak"] = df_eval.groupby("station_id")["RR"].apply(
            rain_streak
        ).values
        df_eval = df_eval.dropna().reset_index(drop=True)

        feat_avail = [f for f in features if f in df_eval.columns]
        X = df_eval[feat_avail]
        y = df_eval["rain"]

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        y_pred      = model.predict(X_test)
        y_prob      = model.predict_proba(X_test)[:, 1]
        cm          = confusion_matrix(y_test, y_pred)
        report      = classification_report(y_test, y_pred, output_dict=True)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        auc_val     = roc_auc_score(y_test, y_prob)

        return cm, report, fpr.tolist(), tpr.tolist(), prec.tolist(), rec.tolist(), auc_val

    cm, report, fpr, tpr, prec_arr, rec_arr, auc_val = compute_test_predictions()

    # ── Tab evaluasi ──
    tab_e1, tab_e2, tab_e3 = st.tabs([
        "🎯  Confusion Matrix & Metrik",
        "📉  ROC & Precision-Recall Curve",
        "🔍  Feature Importance & SHAP",
    ])

    # ─────────────────────────────────────────
    # TAB E1 — Confusion Matrix & Klasifikasi
    # ─────────────────────────────────────────
    with tab_e1:
        col_cm, col_rep = st.columns([1, 1.1], gap="large")

        with col_cm:
            st.markdown("#### Confusion Matrix")

            labels = ["Tidak Hujan", "Hujan"]
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                x=labels, y=labels,
                color_continuous_scale="Blues",
                text_auto=True,
                aspect="equal",
                height=320,
            )
            fig_cm.update_traces(
                textfont_size=18,
                hovertemplate="Aktual: %{y}<br>Prediksi: %{x}<br>Jumlah: %{z}<extra></extra>",
            )
            fig_cm.update_layout(
                margin=dict(t=20, b=20, l=10, r=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            tn, fp, fn, tp = cm.ravel()
            st.markdown(f"""
            <div class='info-box'>
                ✅ <b>TP</b> (prediksi hujan, benar): <b>{tp:,}</b><br>
                ✅ <b>TN</b> (prediksi tidak hujan, benar): <b>{tn:,}</b><br>
                ⚠️ <b>FP</b> (prediksi hujan, salah): <b>{fp:,}</b><br>
                ⚠️ <b>FN</b> (prediksi tidak hujan, salah): <b>{fn:,}</b>
            </div>
            """, unsafe_allow_html=True)

        with col_rep:
            st.markdown("#### Classification Report")

            classes = ["Tidak Hujan (0)", "Hujan (1)"]
            metrics_name = ["precision", "recall", "f1-score"]
            vals = [
                [report["0"][m] for m in metrics_name],
                [report["1"][m] for m in metrics_name],
            ]
            report_df = pd.DataFrame(
                vals,
                columns=["Precision", "Recall", "F1-Score"],
                index=classes,
            ).round(4).reset_index()
            report_df.columns = ["Kelas", "Precision", "Recall", "F1-Score"]

            st.dataframe(report_df, use_container_width=True, hide_index=True)

            # Radar chart per kelas
            categories = ["Precision", "Recall", "F1-Score"]
            fig_radar = go.Figure()
            colors_radar = [C_DRY, C_RAIN]
            for i, (kelas, row) in enumerate(zip(classes, vals)):
                vals_radar = row + [row[0]]
                cats_radar = categories + [categories[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_radar, theta=cats_radar,
                    fill="toself", name=kelas,
                    line_color=colors_radar[i],
                    fillcolor=colors_radar[i],
                    opacity=0.25,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=280,
                margin=dict(t=30, b=20, l=30, r=30),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Accuracy & macro avg
            acc = report.get("accuracy", 0)
            mac = report.get("macro avg", {})
            st.markdown(f"""
            | Metrik | Nilai |
            |--------|-------|
            | **Accuracy** | `{acc:.4f}` |
            | **Macro Precision** | `{mac.get('precision',0):.4f}` |
            | **Macro Recall** | `{mac.get('recall',0):.4f}` |
            | **Macro F1-Score** | `{mac.get('f1-score',0):.4f}` |
            | **ROC-AUC** | `{auc_val:.4f}` |
            """)

    # ─────────────────────────────────────────
    # TAB E2 — ROC & PR Curve
    # ─────────────────────────────────────────
    with tab_e2:
        col_roc, col_pr = st.columns(2, gap="medium")

        with col_roc:
            st.markdown("#### ROC Curve")
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"XGBoost (AUC = {auc_val:.4f})",
                line=dict(color=C_RAIN, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.1)",
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0,1], y=[0,1],
                mode="lines",
                name="Random classifier",
                line=dict(color="#9ca3af", width=1.5, dash="dash"),
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=350,
                margin=dict(t=20, b=20, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode="x unified",
            )
            st.plotly_chart(fig_roc, use_container_width=True)
            st.markdown(f"""
            <div class='info-box'>
                AUC = <b>{auc_val:.4f}</b> → Model sangat baik membedakan hari hujan vs tidak hujan.
                Nilai AUC > 0.9 tergolong <b>excellent</b>.
            </div>
            """, unsafe_allow_html=True)

        with col_pr:
            st.markdown("#### Precision-Recall Curve")
            baseline_pr = report["1"]["support"] / (
                report["0"]["support"] + report["1"]["support"]
            )
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=rec_arr, y=prec_arr,
                mode="lines",
                name="XGBoost",
                line=dict(color=C_GREEN, width=2.5),
                fill="tozeroy",
                fillcolor="rgba(16,185,129,0.1)",
            ))
            fig_pr.add_hline(
                y=baseline_pr,
                line_color="#9ca3af", line_width=1.5, line_dash="dash",
                annotation_text=f"Baseline ({baseline_pr:.2f})",
                annotation_position="top right",
            )
            fig_pr.update_layout(
                xaxis_title="Recall",
                yaxis_title="Precision",
                height=350,
                margin=dict(t=20, b=20, l=10, r=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                hovermode="x unified",
            )
            st.plotly_chart(fig_pr, use_container_width=True)
            st.markdown("""
            <div class='info-box'>
                Precision-Recall Curve berguna untuk dataset imbalanced.
                Kurva yang jauh di atas baseline menunjukkan model efektif mendeteksi hari hujan.
            </div>
            """, unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # TAB E3 — Feature Importance & SHAP
    # ─────────────────────────────────────────
    with tab_e3:
        st.markdown("#### Feature Importance — XGBoost (Gain)")

        importance_vals = model.feature_importances_
        feat_imp = pd.DataFrame({
            "Fitur": features,
            "Importance": importance_vals,
        }).sort_values("Importance", ascending=True).tail(20)

        # Label lebih readable
        feat_label_map = {
            "RR_lag1":"RR lag-1 (kemarin)", "RR_lag2":"RR lag-2",
            "RR_lag3":"RR lag-3", "RH_lag1":"RH lag-1",
            "RH_lag2":"RH lag-2", "RH_lag3":"RH lag-3",
            "RR_roll3":"RR roll 3 hari", "RR_roll7":"RR roll 7 hari",
            "RH_avg_roll3":"RH roll 3 hari", "RH_avg_roll7":"RH roll 7 hari",
            "Tavg_roll3":"Tavg roll 3 hari", "Tavg_roll7":"Tavg roll 7 hari",
            "ff_avg_roll3":"Angin roll 3 hari", "ff_avg_roll7":"Angin roll 7 hari",
            "rain_streak":"Rain streak", "RH_avg":"Kelembapan (RH_avg)",
            "Tavg":"Suhu rata² (Tavg)", "ss":"Sinar matahari (ss)",
            "ff_avg":"Kec. angin (ff_avg)", "ff_x":"Kec. angin max",
            "Tn":"Suhu min (Tn)", "Tx":"Suhu max (Tx)",
            "ddd_x":"Arah angin (ddd_x)", "month":"Bulan",
            "day_of_year":"Hari ke-", "week_of_year":"Minggu ke-",
            "season":"Musim", "wind_sin":"Arah angin (sin)",
            "wind_cos":"Arah angin (cos)",
        }
        feat_imp["Label"] = feat_imp["Fitur"].map(
            lambda x: feat_label_map.get(x, x)
        )

        # Warna gradient
        n = len(feat_imp)
        colors_imp = px.colors.sample_colorscale(
            "Blues", [i / (n - 1) for i in range(n)]
        )

        fig_imp = go.Figure(go.Bar(
            x=feat_imp["Importance"],
            y=feat_imp["Label"],
            orientation="h",
            marker_color=colors_imp,
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            text=feat_imp["Importance"].round(4),
            textposition="outside",
        ))
        fig_imp.update_layout(
            height=560,
            margin=dict(t=20, b=20, l=10, r=60),
            xaxis_title="Importance Score (Gain)",
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("""
        <div class='info-box'>
            💡 <b>Interpretasi Feature Importance (Gain):</b><br>
            Gain mengukur seberapa banyak fitur ini mengurangi impuritas (loss) saat digunakan
            sebagai split point di pohon XGBoost. Semakin tinggi nilai gain, semakin penting
            fitur tersebut dalam membuat keputusan klasifikasi.
        </div>
        """, unsafe_allow_html=True)

        # ── SHAP (jika tersedia) ──
        st.markdown("#### SHAP Feature Importance")
        st.markdown("""
        <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                    padding:1rem;font-size:13px;color:#475569;'>
            SHAP (SHapley Additive exPlanations) menjelaskan <b>kontribusi tiap fitur</b>
            terhadap setiap prediksi secara individual — berbeda dari feature importance
            biasa yang bersifat global.<br><br>
            Untuk menampilkan SHAP plot interaktif, jalankan kode berikut di notebook:
            <pre style='background:#1e293b;color:#e2e8f0;padding:0.75rem;border-radius:8px;
                        margin-top:0.5rem;font-size:12px;'>
import shap, joblib
model = joblib.load('model_xgboost_jatim.pkl')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_sample)
shap.summary_plot(shap_values, X_test_sample, max_display=15)
            </pre>
            Plot SHAP yang sudah disimpan sebagai PNG dapat ditampilkan di sini dengan:
            <code>st.image('shap_beeswarm.png')</code>
        </div>
        """, unsafe_allow_html=True)

        # Tampilkan SHAP PNG jika sudah ada
        import os
        if os.path.exists("shap_beeswarm.png"):
            st.image("shap_beeswarm.png", caption="SHAP Beeswarm Plot", use_column_width=True)
        elif os.path.exists("shap_importance.png"):
            st.image("shap_importance.png", caption="SHAP Bar Chart", use_column_width=True)
        else:
            st.info("📌 Jalankan notebook training dan simpan `shap_beeswarm.png` ke folder yang sama dengan `app.py` untuk menampilkan plot ini.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Route Optimizer",
    page_icon="🗺️",
    layout="wide"
)

# ═══════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════
@st.cache_data
def load_train_data():
    from databricks.sdk import WorkspaceClient
    import io
    w = WorkspaceClient()
    r = w.files.download("/Volumes/hackathon/default/train_running_history/all_trains_history.csv")
    df = pd.read_csv(io.BytesIO(r.contents.read()))
    # ── local fallback ──
    # df = pd.read_csv("all_trains_history.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Train_Number"] = df["Train_Number"].astype(str)
    return df

@st.cache_data
def load_bus_data():
    from databricks.sdk import WorkspaceClient
    import io
    w = WorkspaceClient()
    r = w.files.download("/Volumes/hackathon/default/train_running_history/delhi_traffic.csv")
    df = pd.read_csv(io.BytesIO(r.contents.read()))
    # ── local fallback ──
    # df = pd.read_csv("bus_trips.csv")
    df["actual_time_min"]   = (df["distance_km"] / df["average_speed_kmph"]) * 60
    df["expected_time_min"] = (df["distance_km"] / 60.0) * 60
    df["delay_min"]         = df["actual_time_min"] - df["expected_time_min"]
    return df

train_df = load_train_data()
bus_df   = load_bus_data()

# ═══════════════════════════════════════════════
# 2. ML HELPERS
# ═══════════════════════════════════════════════

@st.cache_data
def get_train_distribution(train_num, station):
    df = train_df.copy()
    df_f = df[(df["Train_Number"] == train_num) & (df["Station"] == station)]\
             .sort_values("Date").reset_index(drop=True)
    if len(df_f) < 5:
        return None

    le = LabelEncoder()
    df_f["Station_Encoded"] = le.fit_transform(df_f["Station"])
    df_f["DayOfWeek"] = df_f["Date"].dt.dayofweek
    df_f["DayNum"]    = (df_f["Date"] - df_f["Date"].min()).dt.days + 1
    df_f["Rolling3"]  = df_f["Delay_Minutes"].shift(1)\
                          .rolling(3, min_periods=1).mean()\
                          .fillna(df_f["Delay_Minutes"].mean())

    X = df_f[["Station_Encoded", "DayOfWeek", "DayNum", "Rolling3"]].values
    y = df_f["Delay_Minutes"].values

    quantiles = {"q10": 0.10, "q25": 0.25, "q50": 0.50, "q75": 0.75, "q90": 0.90}
    today     = df_f["Date"].max() + pd.Timedelta(days=1)
    today_row = np.array([[le.transform([station])[0], today.dayofweek,
                           (today - df_f["Date"].min()).days + 1,
                           df_f["Rolling3"].iloc[-1]]])
    preds = {}
    for name, q in quantiles.items():
        m = GradientBoostingRegressor(loss="quantile", alpha=q,
                                      n_estimators=50, max_depth=2,
                                      learning_rate=0.15, subsample=0.8,
                                      random_state=42)
        m.fit(X, y)
        preds[name] = float(m.predict(today_row)[0])

    hist    = df_f["Delay_Minutes"].values
    dates   = df_f["Date"].dt.date.values
    p50     = preds["q50"]
    shift   = p50 - np.median(hist)
    shifted = hist + shift
    return {"preds": preds, "hist": hist, "dates": dates, "shifted": shifted,
        "today": today, "n_days": len(df_f)}


@st.cache_data
def get_bus_distribution_v2(start_area, end_area, time_of_day, day_of_week, origin_coords=None, dest_coords=None):
    df = bus_df.copy()

    cat_cols = ["start_area", "end_area", "time_of_day", "day_of_week",
                "weather_condition", "traffic_density_level", "road_type"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    FEATURES = ["distance_km", "start_area_enc", "end_area_enc",
                "time_of_day_enc", "day_of_week_enc",
                "weather_condition_enc", "traffic_density_level_enc", "road_type_enc"]
    X = df[FEATURES].values
    y = df["actual_time_min"].values

    quantiles = {"q10": 0.10, "q25": 0.25, "q50": 0.50, "q75": 0.75, "q90": 0.90}
    models = {}
    for name, q in quantiles.items():
        m = GradientBoostingRegressor(loss="quantile", alpha=q,
                                      n_estimators=100, max_depth=3,
                                      learning_rate=0.1, subsample=0.8,
                                      random_state=42)
        m.fit(X, y)
        models[name] = m

    route_h = df[(df["start_area"] == start_area) & (df["end_area"] == end_area)]
    if len(route_h) == 0:
        route_h = df[(df["start_area"] == end_area) & (df["end_area"] == start_area)]
        
    if len(route_h) == 0:
        if origin_coords and dest_coords:
            import metro_routing
            avg_dist = metro_routing.haversine(origin_coords[0], origin_coords[1], dest_coords[0], dest_coords[1])
        else:
            raise ValueError(f"[DEBUG] No history found, and coords missing: origin={origin_coords}, dest={dest_coords}")
            
        cond = df[df["time_of_day"] == time_of_day]
        if len(cond) == 0:
            cond = df
            
        typ_weather = cond["weather_condition"].mode()[0]
        typ_traffic = cond["traffic_density_level"].mode()[0]
        typ_road    = cond["road_type"].mode()[0]
        kde_data = df["actual_time_min"].values
        n_trips = 0
    else:
        avg_dist    = float(route_h["distance_km"].median())
        cond        = route_h[route_h["time_of_day"] == time_of_day]
        if len(cond) == 0:
            cond = route_h
        typ_weather = cond["weather_condition"].mode()[0]
        typ_traffic = cond["traffic_density_level"].mode()[0]
        typ_road    = cond["road_type"].mode()[0]
        kde_data = cond["actual_time_min"].values
        if len(kde_data) < 3:
            kde_data = route_h["actual_time_min"].values
        n_trips = len(route_h)

    def safe_enc(col, val):
        le = encoders[col]
        v  = val if val in le.classes_ else df[col].mode()[0]
        return int(le.transform([v])[0])

    pred_row = np.array([[avg_dist,
                          safe_enc("start_area",              start_area),
                          safe_enc("end_area",                end_area),
                          safe_enc("time_of_day",             time_of_day),
                          safe_enc("day_of_week",             day_of_week),
                          safe_enc("weather_condition",       typ_weather),
                          safe_enc("traffic_density_level",   typ_traffic),
                          safe_enc("road_type",               typ_road)]])

    preds       = {name: float(models[name].predict(pred_row)[0]) for name in models}
    expected_ff = (avg_dist / 60.0) * 60

    kde_data = cond["actual_time_min"].values
    if len(kde_data) < 3:
        kde_data = route_h["actual_time_min"].values

    p50    = preds["q50"]
    scale  = p50 / np.median(kde_data) if np.median(kde_data) > 0 else 1.0
    scaled = kde_data * scale
    return {"preds": preds, "expected_ff": expected_ff, "scaled": scaled,
        "avg_dist": avg_dist, "typ_traffic": typ_traffic,
        "typ_weather": typ_weather, "n_trips": n_trips}


def build_kde(samples, bw=0.12):
    kde = gaussian_kde(samples, bw_method=bw)
    xr  = np.linspace(max(0, samples.min() - 3), samples.max() + 6, 500)
    pdf = kde(xr)
    return kde, xr, pdf
# ═══════════════════════════════════════════════
# 3. COMBINED PROBABILITY
#    Sample from both KDEs, add delays, check
#    P(total_extra_delay <= buffer)
# ═══════════════════════════════════════════════
def compute_combined(train_res, bus_res, buffer_min):
    # Assume bus variance is 0, bus expected time is absolute truth
    # We catch the train strictly when: Train Delay >= -buffer_min
    # To map to the thresholding chart that highlights values <= buffer_min
    total_extra  = -train_res["shifted"]
    prob_on_time = float(np.mean(total_extra <= buffer_min))
    return total_extra, prob_on_time


# ═══════════════════════════════════════════════
# 4. PLOT
# ═══════════════════════════════════════════════
def make_figure(train_res, bus_res, total_delay, prob_on_time,
                buffer_min, train_num, station, start_area, end_area):

    # ── Build KDEs fresh (not cached — kde can't be pickled) ──
    train_kde, train_xr, train_pdf = build_kde(train_res["shifted"])
    bus_kde,   bus_xr,   bus_pdf   = build_kde(bus_res["scaled"])

    fig = plt.figure(figsize=(18, 20))          # <-- taller to fit 3 rows
    fig.patch.set_facecolor("#0f1117")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.32)

    def style_ax(ax):
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333333")
        ax.grid(color="#222222", linestyle="--", alpha=0.5)

    def single_kde_plot(ax, xr, pdf, preds, title, xlabel, accent):
        pdf_n = pdf / pdf.max()
        p10, p25, p50, p75, p90 = (preds["q10"], preds["q25"], preds["q50"],
                                    preds["q75"], preds["q90"])
        ax.fill_between(xr, pdf_n, where=(xr <= p25),
                        color="#00c896", alpha=0.2, label="Low (<P25)")
        ax.fill_between(xr, pdf_n, where=((xr >= p25) & (xr <= p75)),
                        color=accent,    alpha=0.2, label="Typical (P25–P75)")
        ax.fill_between(xr, pdf_n, where=(xr >= p75),
                        color="#e74c3c", alpha=0.18, label="High (>P75)")
        ax.plot(xr, pdf_n, color="white", linewidth=2, zorder=5)

        for xv, col, ls, lw in [(p10, "#00c896", ":", 1.2),
                                  (p25, "#00c896", "--", 1.4),
                                  (p50, accent,    "-", 2.2),
                                  (p75, accent,    "--", 1.4),
                                  (p90, "#e74c3c", ":", 1.2)]:
            ax.axvline(xv, color=col, linestyle=ls, linewidth=lw, alpha=0.9)

        for xv, lbl, col, side in [(p10, f"P10  {p10:.0f}", "#00c896", "left"),
                                    (p50, f"P50  {p50:.0f}", accent,    "right"),
                                    (p90, f"P90  {p90:.0f}", "#e74c3c", "right")]:
            idx  = np.argmin(np.abs(xr - xv))
            yv   = float(pdf_n[idx])
            ypos = min(yv + 0.12, 0.92)
            dx   = -0.5 if side == "left" else 0.4
            ax.annotate(lbl, xy=(xv, yv), xytext=(xv + dx, ypos),
                        fontsize=8.5, color=col, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.7),
                        ha="right" if side == "left" else "left")

        ax.set_xlabel(xlabel,                color="#aaaaaa", fontsize=10)
        ax.set_ylabel("Relative Probability", color="#aaaaaa", fontsize=10)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], color="#aaaaaa")
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=10)
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    # ── Plot 1: Train delay ──
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1)
    single_kde_plot(ax1, train_xr, train_pdf,
                    train_res["preds"],
                    f"🚆 Train {train_num} @ {station}",
                    "Delay (minutes)", "#f5a623")

    # ── Plot 2: Bus extra time ──
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2)
    single_kde_plot(ax2, bus_xr, bus_pdf,          # <-- use bus_xr directly (absolute minutes)
                bus_res["preds"],               # <-- use raw preds (absolute minutes)
                f"🚌 Bus: {start_area} → {end_area}",
                "Travel time (minutes)", "#4a9eff") 

    # ── Plot 3: Combined (full width bottom) ──
    ax3 = fig.add_subplot(gs[1, :])
    style_ax(ax3)

    ckde   = gaussian_kde(total_delay, bw_method=0.12)
    xc     = np.linspace(total_delay.min() - 2, total_delay.max() + 5, 600)
    pdfc   = ckde(xc)
    pdfc_n = pdfc / pdfc.max()

    ax3.fill_between(xc, pdfc_n, where=(xc <= buffer_min),
                     color="#00c896", alpha=0.25,
                     label=f"✅ On time  ({prob_on_time*100:.1f}%)")
    ax3.fill_between(xc, pdfc_n, where=(xc > buffer_min),
                     color="#e74c3c", alpha=0.2,
                     label=f"❌ Late  ({(1-prob_on_time)*100:.1f}%)")
    ax3.plot(xc, pdfc_n, color="white", linewidth=2.5, zorder=5)

    # Buffer marker
    ymax_at_buf = float(pdfc_n[np.argmin(np.abs(xc - buffer_min))])
    ax3.axvline(buffer_min, color="#f5a623", linewidth=2.5,
                linestyle="--", alpha=0.95, zorder=6)
    ax3.annotate(f"Buffer\n{buffer_min} min",
                 xy=(buffer_min, ymax_at_buf),
                 xytext=(buffer_min + 0.5, min(ymax_at_buf + 0.15, 1.1)),
                 color="#f5a623", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color="#f5a623", lw=1))

    # P10 / P50 / P90 of combined
    for q, col, ls in [(10, "#00c896", ":"),
                        (50, "white",   "-"),
                        (90, "#e74c3c", ":")]:
        val = float(np.percentile(total_delay, q))
        ax3.axvline(val, color=col, linestyle=ls, linewidth=1.2, alpha=0.7)
        ax3.text(val + 0.2, 0.03, f"P{q}: {val:.0f}m",
                 color=col, fontsize=8,
                 transform=ax3.get_xaxis_transform())

    risk_col = ("#00c896" if prob_on_time >= 0.8 else
                "#f5a623" if prob_on_time >= 0.5 else "#e74c3c")

    ax3.set_xlabel(
        "Margin Requirement Threshold (-Train Delay) (minutes)",
        color="#aaaaaa", fontsize=11
    )
    ax3.set_ylabel("Relative Probability", color="#aaaaaa", fontsize=11)
    ax3.set_ylim(0, 1.25)
    ax3.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax3.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], color="#aaaaaa")
    ax3.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)
    ax3.set_title(
        f"Journey Confidence  ·  P(on time) = {prob_on_time*100:.1f}%  "
        f"with {buffer_min} min buffer",
        color=risk_col, fontsize=13, fontweight="bold", pad=12
    )

    plt.suptitle("🗺️  Route Optimizer — Journey Risk Analysis",
                 color="white", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    # ── Plot 4: Buffer vs Probability curve ──
    ax4 = fig.add_subplot(gs[2, :])          # <-- new, full width bottom row
    style_ax(ax4)

    buffer_range = np.arange(0, 91, 5)       # 0 to 90 min in steps of 5
    probs = [float(np.mean(total_delay <= b)) for b in buffer_range]

    ax4.plot(buffer_range, [p * 100 for p in probs],
             color="#4a9eff", linewidth=2.5, zorder=5)
    ax4.fill_between(buffer_range, [p * 100 for p in probs],
                     alpha=0.15, color="#4a9eff")

    # Colour bands: red < 50%, yellow 50-80%, green > 80%
    ax4.axhspan(0,  50, color="#e74c3c", alpha=0.07)
    ax4.axhspan(50, 80, color="#f5a623", alpha=0.07)
    ax4.axhspan(80, 100, color="#00c896", alpha=0.07)

    # Threshold lines
    for threshold, col, label in [(50, "#e74c3c", "50%"),
                                   (80, "#00c896", "80%")]:
        ax4.axhline(threshold, color=col, linestyle="--",
                    linewidth=1.2, alpha=0.7)
        ax4.text(91.5, threshold, label, color=col,
                 fontsize=9, va="center", fontweight="bold")

    # Mark the selected buffer
    current_prob = float(np.mean(total_delay <= buffer_min)) * 100
    ax4.axvline(buffer_min, color="#f5a623", linewidth=2,
                linestyle="--", alpha=0.95, zorder=6)
    ax4.scatter([buffer_min], [current_prob],
                color="#f5a623", s=80, zorder=7)
    ax4.annotate(f"  Your buffer\n  {buffer_min} min → {current_prob:.1f}%",
                 xy=(buffer_min, current_prob),
                 xytext=(buffer_min + 3, current_prob - 10),
                 color="#f5a623", fontsize=10, fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color="#f5a623", lw=1))

    # Mark minimum buffer for 80% confidence
    idx_80 = next((i for i, p in enumerate(probs) if p >= 0.8), None)
    if idx_80 is not None:
        b80 = buffer_range[idx_80]
        ax4.axvline(b80, color="#00c896", linewidth=1.5,
                    linestyle=":", alpha=0.8, zorder=6)
        ax4.annotate(f"  Min for 80%\n  {b80} min",
                     xy=(b80, 80),
                     xytext=(b80 + 3, 65),
                     color="#00c896", fontsize=9,
                     arrowprops=dict(arrowstyle="-", color="#00c896", lw=0.8))

    ax4.set_xlabel("Buffer time (minutes)", color="#aaaaaa", fontsize=11)
    ax4.set_ylabel("P(catch train)  %",     color="#aaaaaa", fontsize=11)
    ax4.set_xlim(0, 90)
    ax4.set_ylim(0, 105)
    ax4.set_xticks(buffer_range)
    ax4.set_yticks([0, 25, 50, 75, 80, 90, 100])
    ax4.set_yticklabels(["0%","25%","50%","75%","80%","90%","100%"],
                        color="#aaaaaa")
    ax4.set_title("Buffer Time vs Probability of Catching the Train",
                  color="white", fontsize=13, fontweight="bold", pad=12)
    ax4.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)
    return fig

# ═══════════════════════════════════════════════
# 5. STREAMLIT UI
# ═══════════════════════════════════════════════
st.markdown("## 🗺️ Route Optimizer")
st.markdown("Plan your journey — train delay + bus travel risk + combined on-time probability.")
st.divider()

with st.sidebar:
    st.markdown("### 🚆 Train")
    search_by = st.radio("Search by", ["Train Number", "Station"], horizontal=True)

    all_trains   = sorted(train_df["Train_Number"].unique().tolist())
    all_stations = sorted(train_df["Station"].unique().tolist())

    if search_by == "Train Number":
        train_sel   = st.selectbox("Train Number", all_trains)
        avail_st    = sorted(train_df[train_df["Train_Number"] == train_sel]
                             ["Station"].unique().tolist())
        station_sel = st.selectbox("Station", avail_st)
    else:
        station_sel = st.selectbox("Station", all_stations)
        avail_tr    = sorted(train_df[train_df["Station"] == station_sel]
                             ["Train_Number"].unique().tolist())
        train_sel   = st.selectbox("Train Number", avail_tr)

    st.divider()
    st.markdown("### 🚌 Bus / Cab")
    all_starts = sorted(bus_df["start_area"].unique().tolist())
    all_ends   = sorted(bus_df["end_area"].unique().tolist())

    src_mode = st.radio("Start Location", ["Select Area", "Type Address"], horizontal=True)
    if src_mode == "Select Area":
        start_sel = st.selectbox("Starting Area", all_starts)
    else:
        start_sel = st.text_input("Enter Starting Address", "India Gate, Delhi")

    dst_mode = st.radio("End Location", ["Select Area", "Type Address"], horizontal=True)
    if dst_mode == "Select Area":
        default_end_idx = all_ends.index("Chandni Chowk") if "Chandni Chowk" in all_ends else 0
        end_sel = st.selectbox("Ending Area", all_ends, index=default_end_idx)
    else:
        end_sel = st.text_input("Enter Ending Address", "Rajiv Chowk")

    # Hardcode defaults for Cab ML Model
    time_sel = "Morning Peak"
    day_sel  = "Monday"

    st.divider()
    st.markdown("### ⏱️ Time Remaining")
    st.caption("Minutes remaining until the train's scheduled departure.")
    time_remaining = st.number_input("Time Remaining (minutes)", min_value=-360, max_value=360, value=120, step=5)

    analyse = st.button("🔍 Analyse Journey", type="primary",
                        use_container_width=True)

# ── Main panel ──
if analyse:
    with st.spinner("Training models, fetching Metro routes and computing probabilities..."):
        import metro_routing

        origin_coords = metro_routing.geocode_location(start_sel)
        dest_coords = metro_routing.geocode_location(end_sel)
        
        if not origin_coords:
            st.error(f"Could not locate starting address: {start_sel}")
            st.stop()
        if not dest_coords:
            st.error(f"Could not locate ending address: {end_sel}")
            st.stop()
            
        def get_nearest_metro(coords):
            stations_dict = metro_routing.get_station_coords_cache()
            nearest = None
            min_d = float('inf')
            for st_name, c in stations_dict.items():
                d = metro_routing.haversine(coords[0], coords[1], c[0], c[1])
                if d < min_d:
                    min_d = d
                    nearest = st_name.title()
            return nearest

        actual_start = start_sel
        actual_end = end_sel

        if src_mode == "Type Address":
            nearest_start = get_nearest_metro(origin_coords)
            st.info(f"📍 Mapped given starting address to nearest Metro Station: **{nearest_start}**")
            actual_start = nearest_start

        if dst_mode == "Type Address":
            nearest_end = get_nearest_metro(dest_coords)
            st.info(f"📍 Mapped given ending address to nearest Metro Station: **{nearest_end}**")
            actual_end = nearest_end

        train_res = get_train_distribution(train_sel, station_sel)
        bus_res   = get_bus_distribution_v2(actual_start, actual_end, time_sel, day_sel, origin_coords, dest_coords)
        
        if origin_coords and dest_coords:
            metro_res = metro_routing.get_metro_travel_time_and_path(origin_coords, dest_coords)
        else:
            metro_res = None

    if train_res is None:
        st.error(f"Not enough train data for Train {train_sel} @ {station_sel}.")
        st.stop()
    if bus_res is None:
        st.error(f"No bus history found for {actual_start} → {actual_end}.")
        st.stop()

    cab_expected = bus_res['preds']['q50']
    cab_req_slack = time_remaining - cab_expected
    total_delay_cab, prob_cab = compute_combined(train_res, bus_res, cab_req_slack)
    
    if metro_res:
        metro_time = metro_res['time_mins']
        metro_req_slack = time_remaining - metro_time
        prob_metro = float(np.mean((train_res['shifted'] + metro_req_slack) >= 0))
    else:
        prob_metro = 0.0

    st.markdown("### ⏱️ Travel Options Comparison")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### 🚌 Cab")
        st.metric("Estimated Time", f"{cab_expected:.0f} min")
        st.metric("P(Catch Train)", f"{prob_cab*100:.1f} %")
    with c2:
        st.markdown(f"#### 🚇 Metro")
        if metro_res:
            st.metric("Estimated Time", f"{float(metro_res['time_mins']):.0f} min")
            st.metric("P(Catch Train)", f"{prob_metro*100:.1f} %")
            with st.expander("View Metro Path"):
                st.write(f"**Origin:** {metro_res['origin_station']}")
                st.write(f"**Dest:** {metro_res['dest_station']}")
                for ins in metro_res['instructions']:
                    if ins['type'] == 'board':
                        st.markdown(f"- 🟢 **Board** at {ins['station']} ({ins['line']})")
                    elif ins['type'] == 'travel':
                        st.markdown(f"- 🚇 **Travel** to {ins['to']}")
                    elif ins['type'] == 'transfer':
                        st.markdown(f"- 🔄 **Transfer** at {ins['station']} to {ins['to_line']}")
                    elif ins['type'] == 'alight':
                        st.markdown(f"- 🔴 **Alight** at {ins['station']}")
        else:
            st.warning("Could not find a Metro route for these locations.")

    # ── Figure ──
    fig = make_figure(train_res, bus_res, total_delay_cab, prob_cab,
                      cab_req_slack, train_sel, station_sel, actual_start, actual_end)
    st.pyplot(fig)

    # ── Detail breakdown ──
    st.divider()
    st.markdown("##### 📊 Train Delay Analytics")
    c3, c4 = st.columns(2)
    avg_delay = float(np.mean(train_res['hist']))
    curr_exp  = train_res['preds']['q50']
    c3.metric("Current Expected Delay", f"{curr_exp:.0f} min")
    c4.metric("Average Delay (Historical)", f"{avg_delay:.1f} min")
    
    # 30-day plot
    df_plot = pd.DataFrame({
        "Date": train_res['dates'],
        "Delay (min)": train_res['hist']
    }).sort_values("Date").tail(30)
    
    st.markdown("**Past 30 Days Train Delay**")
    st.line_chart(df_plot.set_index("Date"))
else:
    st.info("👈 Fill in your journey details in the sidebar and click **Analyse Journey**.")
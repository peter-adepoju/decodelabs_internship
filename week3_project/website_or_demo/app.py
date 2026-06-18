"""
website_or_demo/app.py
───────────────────────
Interactive Streamlit dashboard for the E-Commerce Customer Intelligence project.

Run with:
    streamlit run website_or_demo/app.py

Requires the processed master.parquet and rfm_scores.parquet files to exist.
Run notebooks 00-07 first to generate them.

Pages
─────
1. 📊 Overview       — KPI cards and revenue trend
2. 🗺  Geography     — Revenue and late delivery by state
3. 🛍  Products      — Top categories by revenue and review
4. 🚚 Delivery       — Delay distribution and late-rate analysis
5. 👥 Customers      — RFM segments and scatter plot
6. 🔮 Predict        — Interactive review score predictor (uses saved RF model)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Path setup ───────────────────────────────────────────────────────────────
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Customer Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = px.colors.qualitative.Safe


# ─── Data loading (cached) ────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading master dataset...")
def load_master() -> pd.DataFrame:
    p = PROCESSED_DIR / "master.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner="Loading RFM scores...")
def load_rfm() -> pd.DataFrame:
    p = PROCESSED_DIR / "rfm_scores.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_resource(show_spinner="Loading prediction model...")
def load_model():
    import joblib
    p = MODELS_DIR / "rf_review_model.joblib"
    if not p.exists():
        return None
    return joblib.load(p)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def sidebar(master: pd.DataFrame):
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3144/3144456.png", width=80)
    st.sidebar.title("E-Commerce Intel")
    st.sidebar.markdown("---")

    pages = {
        "📊 Overview":    "overview",
        "🗺  Geography":  "geography",
        "🛍  Products":   "products",
        "🚚 Delivery":    "delivery",
        "👥 Customers":   "customers",
        "🔮 Predict":     "predict",
    }
    selected = st.sidebar.radio("Navigate", list(pages.keys()))

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filters**")

    if master.empty:
        return pages[selected], None, None

    # Date range
    min_date = master["order_purchase_timestamp"].min().date()
    max_date = master["order_purchase_timestamp"].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # State filter
    states = sorted(master["customer_state"].dropna().unique().tolist())
    selected_states = st.sidebar.multiselect(
        "Customer state (blank = all)", states, default=[]
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Dataset: Olist Brazilian E-Commerce\n"
        "[Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | CC BY-NC-SA 4.0"
    )

    return pages[selected], date_range, selected_states


def apply_filters(master: pd.DataFrame, date_range, selected_states) -> pd.DataFrame:
    if master.empty:
        return master
    df = master.copy()
    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        df = df[(df["order_purchase_timestamp"] >= start) &
                (df["order_purchase_timestamp"] <= end)]
    if selected_states:
        df = df[df["customer_state"].isin(selected_states)]
    return df


# ─── Pages ────────────────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame):
    st.title("📊 Overview")
    if df.empty:
        st.warning("No data loaded. Run notebooks 00-07 first to generate master.parquet.")
        return

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Orders",    f"{len(df):,}")
    c2.metric("Total Revenue",   f"R$ {df['order_value'].sum()/1e6:.2f}M")
    c3.metric("Avg Order Value", f"R$ {df['order_value'].mean():.2f}")
    c4.metric("Avg Review",      f"{df['review_score'].mean():.2f} ★")
    late_pct = df["is_late"].mean() * 100 if "is_late" in df.columns else 0
    c5.metric("Late Delivery %", f"{late_pct:.1f}%")

    st.markdown("---")

    # Monthly revenue
    df["month"] = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")["order_value"].sum().reset_index()
    fig = px.line(monthly, x="month", y="order_value",
                  title="Monthly Revenue Trend",
                  labels={"order_value": "Revenue (BRL)", "month": "Month"},
                  color_discrete_sequence=[PALETTE[0]])
    fig.update_traces(mode="lines+markers")
    st.plotly_chart(fig, use_container_width=True)

    # Review distribution
    col1, col2 = st.columns(2)
    with col1:
        rev_counts = df["review_score"].value_counts().sort_index().reset_index()
        rev_counts.columns = ["Score", "Count"]
        fig2 = px.bar(rev_counts, x="Score", y="Count",
                      title="Review Score Distribution",
                      color="Score", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        if "payment_type" in df.columns:
            pt = df["payment_type"].value_counts().reset_index()
            pt.columns = ["Payment Type", "Count"]
            fig3 = px.pie(pt, names="Payment Type", values="Count",
                          title="Payment Type Mix",
                          color_discrete_sequence=PALETTE)
            st.plotly_chart(fig3, use_container_width=True)


def page_geography(df: pd.DataFrame):
    st.title("🗺  Geography")
    if df.empty:
        st.warning("No data loaded.")
        return

    col1, col2 = st.columns(2)
    with col1:
        rev_state = (
            df.groupby("customer_state")["order_value"]
            .sum()
            .reset_index()
            .sort_values("order_value", ascending=True)
            .tail(15)
        )
        fig = px.bar(rev_state, x="order_value", y="customer_state",
                     orientation="h",
                     title="Revenue by Customer State (Top 15)",
                     labels={"order_value": "Total Revenue (BRL)", "customer_state": "State"},
                     color="order_value", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "is_late" in df.columns:
            late_state = (
                df.groupby("customer_state")["is_late"]
                .mean()
                .mul(100)
                .reset_index()
                .sort_values("is_late", ascending=True)
                .tail(15)
            )
            late_state.columns = ["State", "Late Rate %"]
            fig2 = px.bar(late_state, x="Late Rate %", y="State",
                          orientation="h",
                          title="Late Delivery Rate by State (Top 15)",
                          color="Late Rate %", color_continuous_scale="Reds")
            st.plotly_chart(fig2, use_container_width=True)

    # Orders per state table
    st.subheader("State-level Summary Table")
    state_summary = df.groupby("customer_state").agg(
        orders=("order_id", "count"),
        revenue=("order_value", "sum"),
        avg_review=("review_score", "mean"),
    ).round(2).reset_index().sort_values("revenue", ascending=False)
    st.dataframe(state_summary, use_container_width=True)


def page_products(df: pd.DataFrame):
    st.title("🛍  Products")
    if df.empty:
        st.warning("No data loaded.")
        return

    if "product_category_english" not in df.columns:
        st.info("Product category column not found. Run notebook 05 first.")
        return

    cat = (
        df.groupby("product_category_english")
        .agg(
            revenue=("order_value",   "sum"),
            orders =("order_id",      "count"),
            avg_review=("review_score","mean"),
        )
        .round(2)
        .reset_index()
    )

    top_n = st.slider("Number of categories to display", 5, 30, 15)
    top_rev = cat.nlargest(top_n, "revenue")

    fig = px.bar(top_rev.sort_values("revenue"), x="revenue", y="product_category_english",
                 orientation="h",
                 title=f"Top {top_n} Categories by Revenue",
                 color="avg_review", color_continuous_scale="RdYlGn",
                 labels={"revenue": "Revenue (BRL)",
                         "product_category_english": "Category",
                         "avg_review": "Avg Review"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Bubble chart: Revenue vs. Review Score")
    fig2 = px.scatter(cat, x="avg_review", y="revenue",
                      size="orders", hover_name="product_category_english",
                      color="avg_review", color_continuous_scale="RdYlGn",
                      title="Category Revenue vs. Avg Review (bubble size = order count)",
                      labels={"avg_review": "Avg Review Score",
                              "revenue": "Total Revenue (BRL)"})
    st.plotly_chart(fig2, use_container_width=True)


def page_delivery(df: pd.DataFrame):
    st.title("🚚 Delivery Performance")
    if df.empty:
        st.warning("No data loaded.")
        return

    if "delivery_delay_days" not in df.columns:
        st.info("Delivery columns not found. Run notebook 05 first.")
        return

    delays = df["delivery_delay_days"].dropna()

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Delay (days)",  f"{delays.mean():.2f}")
    col2.metric("Median Delay",       f"{delays.median():.2f}")
    late_pct = (delays > 0).mean() * 100
    col3.metric("% Late",             f"{late_pct:.1f}%")

    st.markdown("---")
    fig = px.histogram(delays.clip(-20, 60), nbins=60,
                       title="Delivery Delay Distribution",
                       labels={"value": "Delay (days)", "count": "Orders"},
                       color_discrete_sequence=[PALETTE[2]])
    fig.add_vline(x=0, line_dash="dash", line_color="red",
                  annotation_text="On time", annotation_position="top right")
    st.plotly_chart(fig, use_container_width=True)

    # Delay vs review
    st.subheader("Delivery Delay vs. Review Score")
    delay_bins = pd.cut(
        df["delivery_delay_days"].clip(-15, 30),
        bins=[-15, -7, 0, 7, 14, 30],
        labels=["Very early", "Early", "On time", "Late 1-7d", "Very late"],
    )
    df2 = df.copy()
    df2["delay_bin"] = delay_bins
    delay_review = df2.groupby("delay_bin")["review_score"].mean().reset_index()
    fig2 = px.bar(delay_review, x="delay_bin", y="review_score",
                  title="Avg Review Score by Delivery Delay Bucket",
                  color="review_score", color_continuous_scale="RdYlGn",
                  labels={"delay_bin": "Delay Bucket", "review_score": "Avg Review Score"})
    fig2.update_layout(yaxis_range=[1, 5])
    st.plotly_chart(fig2, use_container_width=True)


def page_customers(df: pd.DataFrame):
    st.title("👥 Customer Segments")
    rfm = load_rfm()

    if rfm.empty:
        st.warning("RFM data not found. Run notebook 07 first.")
        return

    # Segment pie
    col1, col2 = st.columns(2)
    with col1:
        seg_counts = rfm["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Customers"]
        fig = px.pie(seg_counts, names="Segment", values="Customers",
                     title="Customer Segments",
                     color_discrete_sequence=PALETTE)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Segment monetary
        seg_mon = rfm.groupby("segment")["monetary"].mean().reset_index().sort_values("monetary")
        fig2 = px.bar(seg_mon, x="monetary", y="segment", orientation="h",
                      title="Avg Spend per Segment",
                      color="monetary", color_continuous_scale="Blues",
                      labels={"monetary": "Avg Monetary (BRL)", "segment": "Segment"})
        st.plotly_chart(fig2, use_container_width=True)

    # RFM scatter
    st.subheader("RFM Scatter: Recency vs Monetary")
    fig3 = px.scatter(rfm, x="recency", y=np.log1p(rfm["monetary"]),
                      color="segment", opacity=0.5,
                      title="RFM Scatter (colour = segment)",
                      labels={"recency": "Recency (days)", "y": "log(1 + Monetary)"},
                      color_discrete_sequence=PALETTE)
    st.plotly_chart(fig3, use_container_width=True)

    # Summary table
    st.subheader("Segment Summary Table")
    seg_table = rfm.groupby("segment").agg(
        customers=("customer_unique_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
    ).round(2).reset_index().sort_values("customers", ascending=False)
    st.dataframe(seg_table, use_container_width=True)


def page_predict(df: pd.DataFrame):
    st.title("🔮 Review Score Predictor")
    st.markdown(
        "Enter order details below to get a predicted review score from the trained "
        "Random Forest model.  This is an *illustrative* predictor — results are "
        "model estimates, not guarantees."
    )

    model = load_model()
    if model is None:
        st.warning(
            "Prediction model not found.  Run notebook 10 first to train and "
            "save the Random Forest model."
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        delivery_delay = st.slider("Delivery delay (days)", -20, 60, 0)
        order_value    = st.number_input("Order value (BRL)", min_value=1.0,
                                         max_value=5000.0, value=150.0, step=10.0)
        freight_value  = st.number_input("Freight value (BRL)", min_value=0.0,
                                          max_value=500.0, value=15.0, step=1.0)
        n_items        = st.number_input("Number of items", min_value=1,
                                          max_value=20, value=1)
    with col2:
        product_weight = st.number_input("Product weight (g)", min_value=50,
                                          max_value=30000, value=500, step=100)
        purchase_month = st.selectbox("Purchase month", list(range(1, 13)), index=5)
        purchase_dow   = st.selectbox("Day of week (0=Mon, 6=Sun)",
                                       list(range(7)), index=1)

    if st.button("Predict review score", type="primary"):
        freight_ratio = freight_value / order_value if order_value > 0 else 0
        features = np.array([[
            delivery_delay,
            order_value,
            freight_value,
            freight_ratio,
            n_items,
            product_weight,
            3,           # product_photos_qty (median placeholder)
            purchase_month,
            purchase_dow,
            0,           # customer_state_encoded (placeholder)
            0,           # product_category_encoded (placeholder)
            0,           # payment_type_encoded (placeholder)
        ]])

        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        st.markdown(f"### Predicted review score: **{pred} ★**")

        prob_df = pd.DataFrame({
            "Score":       [1, 2, 3, 4, 5],
            "Probability": proba,
        })
        fig = px.bar(prob_df, x="Score", y="Probability",
                     color="Score", color_continuous_scale="RdYlGn",
                     title="Predicted probability per score",
                     labels={"Probability": "P(score)"})
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        if delivery_delay > 7:
            st.info(
                "💡 Delivery delay > 7 days is one of the strongest predictors of "
                "low review scores in this dataset.  Consider proactive customer "
                "communication for delayed orders."
            )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    master = load_master()
    page, date_range, selected_states = sidebar(master)
    df = apply_filters(master, date_range, selected_states)

    if page == "overview":
        page_overview(df)
    elif page == "geography":
        page_geography(df)
    elif page == "products":
        page_products(df)
    elif page == "delivery":
        page_delivery(df)
    elif page == "customers":
        page_customers(df)
    elif page == "predict":
        page_predict(df)


if __name__ == "__main__":
    main()

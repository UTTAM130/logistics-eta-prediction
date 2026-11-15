import json
import pandas as pd
import streamlit as st
from pathlib import Path
from src.config import RAW_DIR, PROCESSED_DIR, MODEL_DIR
from src.model import predict_one

st.set_page_config(page_title="Logistics Control Tower", page_icon="🚚", layout="wide")
st.title("🚚 Logistics ETA & Delay Intelligence")
st.caption("Synthetic hackathon demonstration | Point-in-time ETA and delay-risk prediction")

ship_path = RAW_DIR / "shipment.csv"
if not ship_path.exists():
    st.warning("Data not found. Run `python -m src.run_pipeline --shipments 5000` first.")
    st.stop()

shipments = pd.read_csv(ship_path)
events = pd.read_csv(RAW_DIR / "tracking_event.csv")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total shipments", f"{len(shipments):,}")
col2.metric("Delayed shipments", f"{int(shipments.is_delayed.sum()):,}")
col3.metric("On-time rate", f"{(1-shipments.is_delayed.mean())*100:.1f}%")
col4.metric("Tracking events", f"{len(events):,}")

st.subheader("Shipment status")
status_counts = shipments.current_status.value_counts()
st.bar_chart(status_counts)

st.subheader("Delay distribution")
st.bar_chart(shipments.is_delayed.value_counts().rename(index={0:"On time",1:"Delayed"}))

st.subheader("Search shipment")
sid = st.text_input("Shipment ID", value=shipments.shipment_id.iloc[0])
match = shipments[shipments.shipment_id == sid]
if not match.empty:
    s = match.iloc[0]

    delay_hours = round(
        float(s.actual_transit_hours) - float(s.planned_transit_hours), 2
    )

    st.write({
        "shipment_id": s.shipment_id,
        "carrier_id": s.carrier_id,
        "current_status": s.current_status,
        "planned_delivery_at": s.planned_delivery_at,
        "actual_delivery_at": s.actual_delivery_at,
        "delay_hours": delay_hours,
        "ground_truth_delay": int(s.is_delayed)
    })

    ev = events[events.shipment_id == sid].copy()
    ev["event_time"] = pd.to_datetime(ev["event_time"], utc=True)
    ev = ev.sort_values("event_time")

else:
    st.info("No shipment found for that ID.")

st.subheader("What-if ETA prediction")
with st.form("prediction"):
    planned = st.number_input("Planned remaining hours", min_value=0.0, max_value=500.0, value=12.0)
    queue = st.number_input("Facility queue pressure", min_value=0.0, max_value=20.0, value=0.8)
    weather = st.selectbox("Weather severity", [0,1,2,3], format_func=lambda x: ["None","Low","Moderate","Severe"][x])
    carrier = st.slider("Carrier reliability", 0.0, 1.0, 0.90)
    submitted = st.form_submit_button("Predict ETA and delay risk")
if submitted:
    try:
        result = predict_one({
            "planned_remaining_hours": planned, "queue_pressure": queue,
            "weather_severity": weather, "carrier_reliability": carrier
        })
        a,b,c = st.columns(3)
        a.metric("Predicted remaining hours", result["predicted_remaining_hours"])
        b.metric("Delay probability", f'{result["delay_probability"]*100:.1f}%')
        c.metric("Delay risk", result["delay_risk"])
    except FileNotFoundError:
        st.error("Model files missing. Run the pipeline first.")

metrics_path = PROCESSED_DIR / "metrics.json"
if metrics_path.exists():
    st.subheader("Model evaluation")
    st.json(json.loads(metrics_path.read_text()))

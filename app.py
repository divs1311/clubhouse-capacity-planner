"""Streamlit front-end for the Clubhouse Capacity & Staffing Planner."""
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from planner import (Params, allen_cunneen_wait, capacity_table, hourly_demand, norms,
                     plan_roster, simulate_day, stage_means, standard_times, steady_state,
                     to_shifts)

st.set_page_config(page_title="Clubhouse Capacity Planner", layout="wide")
st.title("Clubhouse Capacity & Staffing Planner")
st.caption("Discrete-event simulation of a quick-commerce dark store: pick, pack, inbound "
           "unloading and putaway. Inputs are assumptions; all outputs are simulated.")

with st.sidebar:
    st.header("Demand")
    daily = st.slider("Orders per day", 600, 3000, 1600, step=100)
    peak_mult = st.slider("Peak-day multiplier", 1.0, 2.0, 1.0, step=0.1)
    lines = st.slider("Mean lines per order", 2.0, 8.0, 5.0, step=0.5)
    st.header("Floor")
    walk = st.slider("Walk per line (m)", 4.0, 15.0, 9.0, step=0.5,
                     help="Velocity-based slotting typically shortens this.")
    cv = st.slider("Task-time variability (CV)", 0.1, 0.6, 0.35, step=0.05)
    sla = st.slider("In-store SLA, p90 (seconds)", 150, 360, 240, step=10)

p = Params(daily_orders=daily, lines_mean=lines, walk_m_per_line=walk, cv=cv, sla_s=sla)


@st.cache_data(show_spinner="Finding each crew's capacity (bisection over simulations)...")
def cap_tbl(lines, walk, cv, sla):
    return capacity_table(Params(lines_mean=lines, walk_m_per_line=walk, cv=cv, sla_s=sla), 12, 5)


@st.cache_data(show_spinner="Simulating the full day with roster changes and trucks...")
def day(daily, lines, walk, cv, sla, scale, roster_json):
    pp = Params(daily_orders=daily, lines_mean=lines, walk_m_per_line=walk, cv=cv, sla_s=sla)
    ros = pd.read_json(StringIO(roster_json))
    runs = [simulate_day(pp, ros, scale, s) for s in (5, 6, 7)]
    return pd.concat([r[0] for r in runs]), pd.concat([r[1] for r in runs])


n = norms(p)
cap = cap_tbl(lines, walk, cv, sla)
dem = hourly_demand(p, peak_mult)
ros = plan_roster(p, cap, dem)
heads = ros.pickers + ros.packers
sh_pk = to_shifts(ros.set_index("hour").pickers)
sh_pa = to_shifts(ros.set_index("hour").packers)
shift_hours = int(sh_pk.hours.sum() + sh_pa.hours.sum())
flat = int(heads.max() * len(ros))
orders, trucks = day(daily, lines, walk, cv, sla, peak_mult, ros.to_json())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Standard pick time", f"{n['pick_std_s']:.0f} s", help="incl. 15% PF&D allowance")
c2.metric("Orders / picker-hour", f"{n['orders_per_picker_hr']:.1f}")
c3.metric("Labour hours (8h shifts)", f"{shift_hours}", f"{shift_hours / flat - 1:+.0%} vs flat",
          delta_color="inverse")
c4.metric("Orders ready within SLA", f"{(orders.total <= sla).mean():.1%}")
c5.metric("Worst dock-to-stock", f"{trucks.dock_to_stock_min.max():.0f} min")

t1, t2, t3, t4, t5 = st.tabs(["Time-motion standards", "Crew capacity", "Hourly roster",
                              "Full-day simulation", "Queueing check"])
with t1:
    st.dataframe(standard_times(p), use_container_width=True, hide_index=True)
    st.caption("Normal time per element x lines per order; standard time adds a 15% "
               "personal, fatigue & delay allowance.")
with t2:
    piv = cap.pivot(index="pickers", columns="packers", values="max_orders_hr")
    fig = px.imshow(piv, text_auto=".0f", aspect="auto", color_continuous_scale="Greens",
                    labels=dict(color="Max orders/hr at SLA"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Highest arrival rate each crew sustains with p90 order-to-ready inside the SLA.")
with t3:
    fig = go.Figure()
    fig.add_bar(x=ros.hour, y=ros.demand, name="Orders forecast")
    fig.add_scatter(x=ros.hour, y=ros.capacity, name="Crew capacity", mode="lines+markers")
    fig.add_scatter(x=ros.hour, y=heads * 10, name="Heads x10", mode="lines",
                    line=dict(dash="dot"))
    fig.update_layout(xaxis_title="Hour of day", yaxis_title="Orders / hour")
    st.plotly_chart(fig, use_container_width=True)
    a, b = st.columns(2)
    a.dataframe(ros, use_container_width=True, hide_index=True)
    b.write("**Shifts**")
    b.dataframe(pd.concat([sh_pk.assign(role="picker"), sh_pa.assign(role="packer")]),
                use_container_width=True, hide_index=True)
with t4:
    orders["hour"] = (orders.t // 3600).astype(int)
    hourly = orders.groupby("hour").agg(p90=("total", lambda x: x.quantile(.9)),
                                        within=("total", lambda x: (x <= sla).mean()))
    fig = px.line(hourly.reset_index(), x="hour", y="p90", markers=True,
                  labels=dict(p90="p90 order-to-ready (s)"))
    fig.add_hline(y=sla, line_dash="dash", annotation_text="SLA")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(trucks.assign(arrive_h=lambda x: x.arrive / 3600)
                 [["arrive_h", "cases", "dock_to_stock_min"]].round(1), hide_index=True)
    st.caption("Putaway is done by pickers only when no order is waiting, so dock-to-stock "
               "time stretches on busy days.")
with t5:
    m = stage_means(p)
    svc = m["pick_fixed"] + m["pick_line"] * p.lines_mean
    rows = []
    for c, lam in ((4, 80), (5, 110), (6, 140), (8, 190)):
        sim = np.mean([steady_state(p, c, 8, lam, hours=8, seed=s)["mean_wait_pick"]
                       for s in (1, 2, 3)])
        rows.append(dict(pickers=c, orders_hr=lam, utilisation=lam * svc / 3600 / c,
                         simulated_wait_s=sim,
                         allen_cunneen_wait_s=allen_cunneen_wait(lam, svc, c, p.cv)))
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, use_container_width=True)
    st.caption("Mean wait for a picker from the simulation vs the Allen-Cunneen G/G/c "
               "approximation. Close agreement is a sanity check on the model.")

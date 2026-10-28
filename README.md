# Clubhouse Capacity & Staffing Planner

A discrete-event simulation (SimPy) of one quick-commerce dark store ("clubhouse"). It turns a time-and-motion study into standard times, finds how many orders each crew can handle within the in-store SLA, and builds the cheapest hourly roster for a day's demand curve.

> All inputs are illustrative assumptions and all outputs are simulated. On a real floor the element times would come from a stopwatch study.

## Model

`order arrives → wait for picker → pick → wait for packer → pack → ready for rider`

plus **inbound**: 3 DC trucks a day share one dock (25 min to unload). The cases are put away by pickers, but only when no order is waiting, so putaway competes with orders for picker time.

**Time-motion standard** (5-line order, 9 m of walking per line, 15% PF&D allowance):

| Stage | Elements | Standard time | Norm |
|---|---|---|---|
| Pick | accept task, walk, search, pick, scan ×5 lines, return tote | 124 s | 29.0 orders / picker-hour |
| Pack | verify & QC, bag ×5, label, hand-over | 40 s | 89.4 orders / packer-hour |

Each task time is lognormal (CV 0.35), and lines per order are 1 + Poisson(4).

## What it computes

1. **Crew capacity table.** For every crew of 1–10 pickers and 1–4 packers, it finds the highest arrival rate at which the p90 order-to-ready time stays within 240 s (4 of the 10 minutes). It uses bisection over simulations, averaging 3 seeds per point.
2. **Hourly roster.** For each hour it picks the smallest crew whose capacity covers forecast demand plus 10% headroom. That requirement is then covered with 8-hour shifts by a greedy algorithm.
3. **Full-day validation.** It replays the whole day, including roster changes, the hourly demand curve and inbound trucks, to check SLA attainment and dock-to-stock time.
4. **Scenarios.** It reruns everything for velocity slotting (walk per line 9 m → 6 m) and for a peak day at 1.5× normal demand.
5. **Queueing sanity check.** It compares simulated picker waits with the Allen–Cunneen G/G/c approximation.

## Results (1,600 orders/day, 06:00–24:00, peak hour 144 orders)

| Scenario | Peak heads | Labour hours (8h shifts) | vs flat peak staffing | Orders within SLA | Worst dock-to-stock |
|---|---|---|---|---|---|
| Current layout | 10 | 151 | −16% (180) | 91.3% | 115 min |
| Velocity-slotted (6 m/line) | 9 | 125 | −23% (162) | 93.9% | 124 min |
| Peak day 1.5×, current layout | 14 | 195 | −23% (252) | 92.1% | 155 min |
| Peak day 1.5×, slotted | 11 | 157 | −21% (198) | 92.9% | 174 min |

What the results show:
- **Staffing to the demand curve instead of the peak saves 16% of labour hours** at the same SLA.
- **Velocity slotting raises picker productivity by 14%** (29.0 → 33.2 orders per picker-hour) and cuts the daily roster from 151 to 125 labour hours (−17%).
- **On peak days, putaway by idle pickers breaks down.** Dock-to-stock time exceeds 2 hours because pickers are never free. Peak days need a dedicated putaway associate during inbound windows.
- **The queueing check agrees with the simulation:** simulated waits are 13–17 s versus Allen–Cunneen's 8–11 s at 60–71% utilisation. The gap is expected, because the approximation ignores the variation in lines per order.

Full outputs are in `results/` (`summary.json`, capacity tables, rosters, standard times).

## Run it

```bash
pip install -r requirements.txt
python analysis.py          # reproduces results/
streamlit run app.py        # interactive planner
```

## Limitations
- There is one dock and one pick zone. Batch picking and zone picking are not modelled.
- The SLA covers in-store time only; rider wait and delivery time are outside the model.
- Shift building is greedy, not an integer program, and ignores breaks.

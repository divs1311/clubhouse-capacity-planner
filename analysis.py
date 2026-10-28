"""Runs the full study and writes results/ (CSV) - numbers quoted in the README come from here."""
import json
import os
import numpy as np
import pandas as pd
from planner import (Params, PFD_ALLOWANCE, allen_cunneen_wait, capacity_table, hourly_demand,
                     norms, plan_roster, simulate_day, stage_means, standard_times, steady_state,
                     to_shifts)

os.makedirs("results", exist_ok=True)
out = {}
base = Params()
slot = Params(walk_m_per_line=6.0)

standard_times(base).to_csv("results/standard_times.csv", index=False)
out["norms_current"] = norms(base)
out["norms_slotted"] = norms(slot)

cap_b = capacity_table(base, 10, 4); cap_b.to_csv("results/capacity_current.csv", index=False)
cap_s = capacity_table(slot, 10, 4); cap_s.to_csv("results/capacity_slotted.csv", index=False)


def scenario(p, cap, scale, tag):
    dem = hourly_demand(p, scale)
    ros = plan_roster(p, cap, dem)
    heads = ros.pickers + ros.packers
    sh_pk = to_shifts(ros.set_index("hour").pickers)
    sh_pa = to_shifts(ros.set_index("hour").packers)
    shift_hours = int(sh_pk.hours.sum() + sh_pa.hours.sum())
    peak = ros.loc[heads.idxmax()]
    flat_hours = int((peak.pickers + peak.packers) * len(ros))
    res = []
    for seed in (5, 6, 7):
        df, tr = simulate_day(p, ros, scale, seed)
        res.append(dict(within=(df.total <= p.sla_s).mean(), p90=df.total.quantile(.9),
                        orders=len(df), dock_to_stock=tr.dock_to_stock_min.max()))
    r = pd.DataFrame(res).mean()
    ros.to_csv(f"results/roster_{tag}.csv", index=False)
    return dict(orders=float(dem.sum()), peak_hour_orders=float(dem.max()),
                roster_heads_peak=int(heads.max()), roster_labor_hours=int(heads.sum()),
                shift_labor_hours=shift_hours, n_shifts=len(sh_pk) + len(sh_pa),
                flat_labor_hours=flat_hours, sla_attainment=float(r.within),
                p90_s=float(r.p90), worst_dock_to_stock_min=float(r.dock_to_stock),
                orders_per_labor_hour=float(dem.sum() / shift_hours))


out["normal_current"] = scenario(base, cap_b, 1.0, "normal_current")
out["normal_slotted"] = scenario(slot, cap_s, 1.0, "normal_slotted")
out["peak_current"] = scenario(base, cap_b, 1.5, "peak_current")
out["peak_slotted"] = scenario(slot, cap_s, 1.5, "peak_slotted")

# Queueing cross-check: pick stage only (packers ample)
m = stage_means(base)
svc = m["pick_fixed"] + m["pick_line"] * base.lines_mean
checks = []
for c, lam in ((4, 80), (5, 110), (6, 140), (8, 190)):
    sims = [steady_state(base, c, 6, lam, hours=8, seed=s) for s in (1, 2, 3)]
    checks.append(dict(pickers=c, orders_hr=lam, util=lam * svc / 3600 / c,
                       des_wait_s=float(np.mean([x["mean_wait_pick"] for x in sims])),
                       allen_cunneen_wait_s=allen_cunneen_wait(lam, svc, c, base.cv)))
out["queue_check"] = checks
json.dump(out, open("results/summary.json", "w"), indent=2, default=float)
print(json.dumps(out, indent=2, default=float))

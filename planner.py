"""
Clubhouse Capacity & Staffing Planner
=====================================

Discrete-event simulation (SimPy) of a quick-commerce dark store ("clubhouse"):

    order arrives -> waits for a picker -> pick (time-motion elements) ->
    waits for a packer -> pack -> handed to rider

plus inbound trucks from the DC (one dock, unload, then case putaway done by pickers
only when no order is waiting).

What it answers
  1. Standard times: element-level time-motion table -> standard time per order
     (normal time x (1 + PF&D allowance)) -> orders/hour per picker and packer.
  2. Capacity: the highest order rate a (pickers, packers) crew can take while
     90% of orders are ready inside the in-store SLA.
  3. Hourly staffing roster for a day's demand curve, then converted to 8-hour shifts,
     validated by a full-day simulation with roster changes and inbound trucks.
  4. Layout scenario (velocity slotting -> shorter walk per line) and peak-day load.
  5. Cross-check of the pick stage against an Erlang-C (M/M/c) queue.

All inputs are illustrative assumptions; all outputs are SIMULATED.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import simpy

# ----------------------------------------------------------------------------
# Time-motion element table (seconds). "per_line" elements repeat for each line.
# ----------------------------------------------------------------------------
PICK_ELEMENTS = [
    # element,                      seconds, per_line
    ("Accept task, grab tote",          12, False),
    ("Walk to bin",                    None, True),   # = walk metres / walk speed
    ("Search & confirm bin",             4, True),
    ("Pick item(s)",                     3, True),
    ("Scan item",                        2, True),
    ("Return tote to pack station",      10, False),
]
PACK_ELEMENTS = [
    ("Verify order & quality check",      8, False),
    ("Bag items (cold-chain split)",      3, True),
    ("Print & stick label",               6, False),
    ("Hand to rider / staging shelf",     6, False),
]
WALK_SPEED = 1.1          # m/s with tote
PFD_ALLOWANCE = 0.15      # personal, fatigue & delay allowance


@dataclass
class Params:
    lines_mean: float = 5.0          # mean lines per order (1 + Poisson)
    walk_m_per_line: float = 9.0     # current layout; velocity slotting ~6 m
    cv: float = 0.35                 # variability of each task time (lognormal)
    sla_s: float = 240.0             # order-to-ready target (p90): 4 of the 10 minutes
    daily_orders: int = 1600
    open_hour: int = 6
    close_hour: int = 24
    trucks: tuple = ((5.5, 220), (13.0, 120), (18.5, 160))   # (arrival hour, cases)
    unload_min: float = 25.0
    putaway_s_per_case: float = 22.0
    seed: int = 42

    # demand share by hour, 06:00 .. 23:00 (sums to 1)
    hourly_share: tuple = (0.025, 0.055, 0.075, 0.075, 0.060, 0.050, 0.050, 0.050,
                           0.045, 0.040, 0.040, 0.045, 0.060, 0.080, 0.090, 0.075,
                           0.055, 0.030)


def standard_times(p: Params) -> pd.DataFrame:
    """Element table -> normal and standard time for an average order."""
    rows = []
    for stage, elements in (("Pick", PICK_ELEMENTS), ("Pack", PACK_ELEMENTS)):
        for name, sec, per_line in elements:
            if sec is None:
                sec = p.walk_m_per_line / WALK_SPEED
            mult = p.lines_mean if per_line else 1
            rows.append(dict(stage=stage, element=name, seconds_each=round(sec, 1),
                             per_line=per_line, seconds_per_order=round(sec * mult, 1)))
    df = pd.DataFrame(rows)
    return df


def stage_means(p: Params) -> dict:
    """Mean normal time (s) per order for pick and pack, given lines per order."""
    def fixed_and_line(elements):
        fixed = sum(s for _, s, pl in elements if not pl and s is not None)
        line = sum((p.walk_m_per_line / WALK_SPEED if s is None else s)
                   for _, s, pl in elements if pl)
        return fixed, line
    pf, pl = fixed_and_line(PICK_ELEMENTS)
    kf, kl = fixed_and_line(PACK_ELEMENTS)
    return dict(pick_fixed=pf, pick_line=pl, pack_fixed=kf, pack_line=kl)


def norms(p: Params) -> dict:
    m = stage_means(p)
    pick = (m["pick_fixed"] + m["pick_line"] * p.lines_mean) * (1 + PFD_ALLOWANCE)
    pack = (m["pack_fixed"] + m["pack_line"] * p.lines_mean) * (1 + PFD_ALLOWANCE)
    return dict(pick_std_s=pick, pack_std_s=pack,
                orders_per_picker_hr=3600 / pick, orders_per_packer_hr=3600 / pack)


# ----------------------------------------------------------------------------
# Simulation core
# ----------------------------------------------------------------------------
class Clubhouse:
    def __init__(self, env, p: Params, rng, pickers: int, packers: int):
        self.env, self.p, self.rng = env, p, rng
        self.m = stage_means(p)
        self.pickers = simpy.Container(env, init=pickers, capacity=200)
        self.packers = simpy.Container(env, init=packers, capacity=200)
        self.dock = simpy.Resource(env, capacity=1)
        self.waiting_orders = 0
        self.records = []
        self.putaway_backlog = 0
        self.truck_log = []
        self.busy_pick_s = 0.0

    def _ln(self, mean):
        s = math.sqrt(math.log(1 + self.p.cv ** 2))
        return float(self.rng.lognormal(math.log(mean) - s * s / 2, s))

    def order(self, lines):
        env, t0 = self.env, self.env.now
        self.waiting_orders += 1
        yield self.pickers.get(1)
        self.waiting_orders -= 1
        t_pick_start = env.now
        pick = self._ln(self.m["pick_fixed"] + self.m["pick_line"] * lines)
        yield env.timeout(pick)
        self.busy_pick_s += pick
        yield self.pickers.put(1)
        yield self.packers.get(1)
        pack = self._ln(self.m["pack_fixed"] + self.m["pack_line"] * lines)
        yield env.timeout(pack)
        yield self.packers.put(1)
        self.records.append(dict(t=t0, hour=int(t0 // 3600), wait_pick=t_pick_start - t0,
                                 total=env.now - t0, lines=lines))

    def arrivals(self, rate_fn, until):
        """Non-homogeneous Poisson arrivals by thinning; rate_fn(t) in orders/s."""
        lam_max = max(rate_fn(h * 3600 + 1) for h in range(int(until // 3600) + 1)) or 1e-9
        while True:
            yield self.env.timeout(self.rng.exponential(1 / lam_max))
            if self.env.now >= until:
                return
            if self.rng.random() < rate_fn(self.env.now) / lam_max:
                lines = 1 + self.rng.poisson(self.p.lines_mean - 1)
                self.env.process(self.order(lines))

    def truck(self, arrive_s, cases):
        yield self.env.timeout(max(arrive_s - self.env.now, 0))
        t0 = self.env.now
        with self.dock.request() as req:
            yield req
            yield self.env.timeout(self.p.unload_min * 60)
        self.putaway_backlog += cases
        self.truck_log.append(dict(arrive=t0, cases=cases, unloaded=self.env.now, done=None))

    def putaway_worker(self):
        """Pickers put away cases in batches of 10, only when no order is waiting."""
        while True:
            if self.putaway_backlog > 0 and self.waiting_orders == 0 and self.pickers.level > 0:
                yield self.pickers.get(1)
                n = min(10, self.putaway_backlog)
                self.putaway_backlog -= n
                yield self.env.timeout(self._ln(n * self.p.putaway_s_per_case))
                yield self.pickers.put(1)
                if self.putaway_backlog == 0:
                    for tl in self.truck_log:
                        if tl["done"] is None:
                            tl["done"] = self.env.now
            else:
                yield self.env.timeout(20)

    def roster(self, plan: dict):
        """plan: {hour: (pickers, packers)}; changes headcount at the top of each hour."""
        cur_pk, cur_pa = self.pickers.level, self.packers.level
        for h in sorted(plan):
            yield self.env.timeout(max(h * 3600 - self.env.now, 0))
            pk, pa = plan[h]
            for res, cur, new in ((self.pickers, cur_pk, pk), (self.packers, cur_pa, pa)):
                if new > cur:
                    yield res.put(new - cur)
                elif new < cur:
                    self.env.process(self._remove(res, cur - new))
            cur_pk, cur_pa = pk, pa

    def _remove(self, res, n):
        for _ in range(n):           # associate leaves once they finish the current task
            yield res.get(1)


def steady_state(p: Params, pickers: int, packers: int, orders_per_hr: float,
                 hours: float = 3.0, warm_h: float = 0.5, seed: int | None = None) -> dict:
    """Constant arrival rate; returns SLA stats after warm-up."""
    rng = np.random.default_rng(p.seed if seed is None else seed)
    env = simpy.Environment()
    ch = Clubhouse(env, p, rng, pickers, packers)
    env.process(ch.arrivals(lambda t: orders_per_hr / 3600, hours * 3600))
    env.run(until=hours * 3600 + 3600)
    df = pd.DataFrame(ch.records)
    df = df[df.t >= warm_h * 3600]
    if df.empty:
        return dict(p90=np.inf, within_sla=0.0, n=0)
    return dict(p90=df.total.quantile(.9), within_sla=(df.total <= p.sla_s).mean(),
                mean=df.total.mean(), n=len(df), mean_wait_pick=df.wait_pick.mean(),
                picker_util=ch.busy_pick_s / (pickers * hours * 3600))


def _p90(p, pickers, packers, rate, seeds=(1, 2, 3)):
    hours = min(max(4.0, 300.0 / rate), 150.0)     # at least ~300 orders per run
    return float(np.mean([steady_state(p, pickers, packers, rate, hours=hours, seed=s)["p90"]
                          for s in seeds]))


def max_rate(p: Params, pickers: int, packers: int, lo=2.0, hi=600.0, tol=2.0) -> float:
    """Highest orders/hour with p90 order-to-ready <= SLA (bisection, 3 seeds averaged)."""
    if _p90(p, pickers, packers, lo) > p.sla_s:
        return 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if _p90(p, pickers, packers, mid) <= p.sla_s:
            lo = mid
        else:
            hi = mid
    return lo


def capacity_table(p: Params, max_pickers=10, max_packers=5) -> pd.DataFrame:
    rows = []
    for pk in range(1, max_pickers + 1):
        for pa in range(1, max_packers + 1):
            if pa > pk:
                continue
            rows.append(dict(pickers=pk, packers=pa, max_orders_hr=max_rate(p, pk, pa)))
    return pd.DataFrame(rows)


def hourly_demand(p: Params, scale: float = 1.0) -> pd.Series:
    hours = range(p.open_hour, p.close_hour)
    share = np.array(p.hourly_share) / sum(p.hourly_share)
    return pd.Series(share * p.daily_orders * scale, index=list(hours), name="orders")


def plan_roster(p: Params, cap: pd.DataFrame, demand: pd.Series, headroom=1.10) -> pd.DataFrame:
    """Cheapest crew (fewest heads) per hour whose capacity covers demand x headroom."""
    rows = []
    for h, d in demand.items():
        ok = cap[cap.max_orders_hr >= d * headroom].copy()
        if ok.empty:
            best = cap.sort_values("max_orders_hr").iloc[-1]
        else:
            ok["heads"] = ok.pickers + ok.packers
            best = ok.sort_values(["heads", "max_orders_hr"], ascending=[True, False]).iloc[0]
        rows.append(dict(hour=h, demand=d, pickers=int(best.pickers), packers=int(best.packers),
                         capacity=best.max_orders_hr))
    return pd.DataFrame(rows)


def to_shifts(req: pd.Series, length: int = 8, min_len: int = 4) -> pd.DataFrame:
    """Greedy cover of an hourly headcount requirement with shifts.
    Opens 8-hour shifts when coverage falls short; last shifts of the day are trimmed
    to closing time (allowed down to min_len hours)."""
    hours = list(req.index)
    end = hours[-1] + 1
    cover = pd.Series(0, index=hours)
    shifts = []
    for h in hours:
        short = req[h] - cover[h]
        for _ in range(max(int(short), 0)):
            stop = min(h + length, end)
            if stop - h < min_len:          # start earlier so the shift is long enough
                start = max(hours[0], stop - min_len)
            else:
                start = h
            shifts.append((start, stop))
            cover.loc[start:stop - 1] += 1
    return pd.DataFrame(shifts, columns=["start", "end"]).assign(hours=lambda x: x.end - x.start)


def simulate_day(p: Params, roster: pd.DataFrame, scale: float = 1.0, seed: int | None = None):
    rng = np.random.default_rng(p.seed if seed is None else seed)
    env = simpy.Environment(initial_time=0)
    first = roster.iloc[0]
    ch = Clubhouse(env, p, rng, int(first.pickers), int(first.packers))
    dem = hourly_demand(p, scale)

    def rate(t):
        h = int(t // 3600)
        return dem.get(h, 0.0) / 3600

    env.run(until=p.open_hour * 3600)
    env.process(ch.roster({int(r.hour): (int(r.pickers), int(r.packers))
                           for r in roster.itertuples()}))
    env.process(ch.arrivals(rate, p.close_hour * 3600))
    for hr, cases in p.trucks:
        env.process(ch.truck(hr * 3600, int(cases * scale)))
    env.process(ch.putaway_worker())
    env.run(until=p.close_hour * 3600 + 1800)
    df = pd.DataFrame(ch.records)
    trucks = pd.DataFrame(ch.truck_log)
    if not trucks.empty:
        trucks["dock_to_stock_min"] = (trucks["done"] - trucks["arrive"]) / 60
    return df, trucks


def allen_cunneen_wait(lam_hr: float, mean_service_s: float, c: int, cv_s: float) -> float:
    """G/G/c approximation: Erlang-C wait scaled by (ca^2 + cs^2) / 2, Poisson arrivals."""
    return erlang_c_wait(lam_hr, mean_service_s, c) * (1 + cv_s ** 2) / 2


def erlang_c_wait(lam_hr: float, mean_service_s: float, c: int) -> float:
    """Mean queue wait (s) for M/M/c."""
    lam = lam_hr / 3600
    a = lam * mean_service_s
    rho = a / c
    if rho >= 1:
        return math.inf
    s = sum(a ** k / math.factorial(k) for k in range(c))
    top = a ** c / math.factorial(c) / (1 - rho)
    pw = top / (s + top)
    return pw * mean_service_s / (c - a)


if __name__ == "__main__":
    p = Params()
    print(standard_times(p))
    print(norms(p))
    print(steady_state(p, 4, 2, 100))

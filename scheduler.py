"""
Smart RR Scheduler Lab  (Uniprocessor, preemptive Round Robin)
Compares: Dynamic-Quantum RR  vs  fixed-quantum RR, and finds the best fixed quantum.

Run:   python scheduler.py                  -> asks you to type processes
       python scheduler.py processes.txt    -> reads processes from a file
File format (one process per line):  arrival burst      e.g.  0 24
"""
import sys, math, statistics

# ---------------------------------------------------------------- core simulator
def dynamic_quantum(rems):
    """Quantum for one round, from remaining burst times of ready processes."""
    mean = sum(rems) / len(rems)
    sd = statistics.pstdev(rems)
    s = sorted(rems)
    if sd / mean < 0.5:                      # similar jobs -> longest remaining burst
        q = s[-1]
    else:                                    # mixed jobs -> 75th percentile
        q = s[min(len(s) - 1, int(len(s) * 0.75))]
    return max(1, math.ceil(q))

def simulate(procs, cs, fixed_q=None):
    """procs: list of (arrival, burst). fixed_q=None means Dynamic RR.
    Returns dict with metrics, gantt list [(label,start,end)], quanta used."""
    n = len(procs)
    order = sorted(range(n), key=lambda i: (procs[i][0], i))
    rem = [b for _, b in procs]
    first = [-1] * n
    comp = [0] * n
    ready, gantt, qlog = [], [], []
    t = idx = done = switches = busy = left = 0
    last, Q = -1, 1

    def admit():
        nonlocal idx
        while idx < n and procs[order[idx]][0] <= t:
            ready.append(order[idx]); idx += 1
    admit()
    while done < n:
        if not ready:                                   # CPU idle until next arrival
            nt = procs[order[idx]][0]
            gantt.append(("IDLE", t, nt)); t = nt; last = -1; admit(); continue
        i = ready.pop(0)
        if left <= 0:                                   # start of a new round
            rems = [rem[i]] + [rem[k] for k in ready]
            Q = fixed_q if fixed_q else dynamic_quantum(rems)
            if not fixed_q and len(rems) == 1 and idx < n:
                Q = max(1, procs[order[idx]][0] - t)    # lone process: run till next arrival
            qlog.append(Q); left = len(rems)
        left -= 1
        if last >= 0 and last != i:                     # context switch
            switches += 1
            if cs > 0:
                gantt.append(("CS", t, t + cs)); t += cs; admit()
        if first[i] < 0: first[i] = t
        run = min(Q, rem[i])
        gantt.append((f"P{i+1}", t, t + run)); t += run; busy += run
        rem[i] -= run; last = i; admit()
        if rem[i] > 0: ready.append(i)
        else: comp[i] = t; done += 1

    rows = []
    for i, (at, bt) in enumerate(procs):
        tat = comp[i] - at
        rows.append(dict(pid=f"P{i+1}", at=at, bt=bt, ct=comp[i], tat=tat, wt=tat - bt, rt=first[i] - at))
    avg = lambda k: sum(r[k] for r in rows) / n
    return dict(rows=rows, wt=avg("wt"), tat=avg("tat"), rt=avg("rt"),
                switches=switches, util=100 * busy / t, gantt=gantt, qlog=qlog)

# ---------------------------------------------------------------- auto-tuner
def auto_tune(procs, cs):
    """Try every fixed q from 1..longest burst (max 60). Return best q and the full curve."""
    curve = []
    for q in range(1, min(60, max(b for _, b in procs)) + 1):
        curve.append((q, simulate(procs, cs, q)))
    best = curve[0]
    for q, r in curve:
        if r["wt"] < best[1]["wt"] - 1e-9 or (abs(r["wt"] - best[1]["wt"]) < 1e-9 and r["switches"] <= best[1]["switches"]):
            best = (q, r)
    return best, curve

# ---------------------------------------------------------------- display helpers
def show_table(r):
    print(f"{'PID':<5}{'Arrival':>8}{'Burst':>7}{'Complete':>10}{'TAT':>6}{'Wait':>6}{'Resp':>6}")
    for x in r["rows"]:
        print(f"{x['pid']:<5}{x['at']:>8}{x['bt']:>7}{x['ct']:>10}{x['tat']:>6}{x['wt']:>6}{x['rt']:>6}")

def show_gantt(g):
    print("Gantt:", " ".join(f"|{l}({s}-{e})" for l, s, e in g) + "|")

def line(name, r):
    print(f"{name:<24} WT={r['wt']:6.2f}  TAT={r['tat']:6.2f}  RT={r['rt']:6.2f}  switches={r['switches']:<3} CPU={r['util']:5.1f}%")

def read_input():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            procs = [tuple(map(int, l.split()[:2])) for l in f if l.strip() and not l.startswith("#")]
    else:
        n = int(input("Number of processes: "))
        procs = []
        for i in range(n):
            a, b = map(int, input(f"P{i+1}  arrival burst: ").split())
            procs.append((a, b))
    if any(a < 0 or b < 1 for a, b in procs):
        sys.exit("Arrival must be >= 0 and burst >= 1")
    return procs

def plot(curve, dyn, best_q, my_q, my_wt):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed - skipping graph. Install: pip install matplotlib)"); return
    qs = [q for q, _ in curve]; wts = [r["wt"] for _, r in curve]
    plt.figure(figsize=(8, 4.5))
    plt.plot(qs, wts, label="Fixed RR", lw=2)
    plt.axhline(dyn["wt"], color="green", ls="--", lw=2, label=f"Dynamic RR ({dyn['wt']:.1f})")
    plt.scatter([best_q], [wts[best_q - 1]], color="orange", zorder=5, s=70, label=f"Best fixed q={best_q}")
    if my_q <= len(qs): plt.scatter([my_q], [my_wt], color="red", zorder=5, label=f"Your q={my_q}")
    plt.xlabel("Fixed time quantum"); plt.ylabel("Average waiting time"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Auto-Tuner: waiting time vs quantum"); plt.tight_layout(); plt.savefig("quantum_graph.png", dpi=150)
    print("Graph saved: quantum_graph.png")

# ---------------------------------------------------------------- main
def main():
    procs = read_input()
    cs = int(input("Context-switch cost [1]: ") or 1)
    my_q = int(input("Your own fixed quantum [4]: ") or 4)

    dyn = simulate(procs, cs)
    (bq, best), curve = auto_tune(procs, cs)
    mine = simulate(procs, cs, my_q)

    print("\n=== Dynamic RR: per-process results ===")
    show_table(dyn); show_gantt(dyn["gantt"])
    print("Quantum chosen each round:", dyn["qlog"])
    print(f"\n=== Best fixed RR (q={bq}) Gantt ===")
    show_gantt(best["gantt"])

    print("\n=== Comparison (lower waiting time is better) ===")
    line("Dynamic RR", dyn); line(f"Best fixed RR (q={bq})", best); line(f"Your fixed RR (q={my_q})", mine)
    gap = (dyn["wt"] - best["wt"]) / best["wt"] * 100
    if dyn["wt"] <= best["wt"] + 1e-9:
        print(f"\nVerdict: Dynamic RR matched or beat the best fixed quantum, with no tuning.")
    else:
        print(f"\nVerdict: Dynamic RR is within {gap:.1f}% of the best fixed quantum (q={bq}), found without trying every value.")
    print(f"Against your q={my_q}, Dynamic RR changes waiting time by {(mine['wt']-dyn['wt'])/mine['wt']*100:.1f}% (positive = better).")
    plot(curve, dyn, bq, my_q, mine["wt"])

if __name__ == "__main__":
    main()

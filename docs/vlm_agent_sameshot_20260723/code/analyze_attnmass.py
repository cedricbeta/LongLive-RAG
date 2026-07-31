import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "attnmass_longlive_8x120.json"
data = json.load(open(path))

records = [r for p in data for r in p["records"]]
print(f"prompts={len(data)} records={len(records)}")

def ratio(rs, key, ukey):
    m = sum(r[key] for r in rs) / len(rs)
    u = sum(r[ukey] for r in rs) / len(rs)
    return m, u, m / u

print("\n=== overall (all layers, all calls) ===")
for name, key, ukey in [("sink", "mass_sink", "uniform_sink"),
                        ("mem", "mass_mem", "uniform_mem"),
                        ("local", "mass_local", "uniform_local")]:
    m, u, r = ratio(records, key, ukey)
    print(f"{name:6s} mass={m:.4f} uniform={u:.4f} ratio={r:.3f}")

print("\n=== by timestep (0 = clean recache) ===")
by_t = defaultdict(list)
for r in records:
    by_t[r.get("timestep", -1)].append(r)
for t in sorted(by_t, reverse=True):
    m, u, rr = ratio(by_t[t], "mass_mem", "uniform_mem")
    print(f"t={t:5d} n={len(by_t[t]):6d} mem mass={m:.4f} uniform={u:.4f} ratio={rr:.3f}")

print("\n=== by layer (denoise calls only) ===")
den = [r for r in records if r.get("timestep", -1) > 0]
by_l = defaultdict(list)
for r in den:
    by_l[r["layer"]].append(r)
for l in sorted(by_l):
    m, u, rr = ratio(by_l[l], "mass_mem", "uniform_mem")
    hmax = max(r["mem_head_max"] for r in by_l[l])
    print(f"layer={l:2d} mem ratio={rr:.3f} (mass={m:.4f}) head_max={hmax:.3f}")

print("\n=== by generation progress (frame_start, denoise only) ===")
by_f = defaultdict(list)
for r in den:
    by_f[r["frame_start"] // 12 * 12].append(r)
for f in sorted(by_f):
    m, u, rr = ratio(by_f[f], "mass_mem", "uniform_mem")
    print(f"frame {f:3d}-{f+11:3d} n={len(by_f[f]):6d} mem ratio={rr:.3f}")

print("\n=== distribution of per-record mem ratio (denoise only) ===")
ratios = sorted(r["mass_mem"] / r["uniform_mem"] for r in den if r["uniform_mem"] > 0)
n = len(ratios)
for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
    print(f"p{int(q*100):02d} = {ratios[int(q*(n-1))]:.3f}")

print("\n=== retrieval behaviour (memory_indices_log) ===")
for p in data[:3]:
    log = p["memory_log"]
    if not log:
        continue
    picks = [tuple(e["selected_global_frames"][0]) for e in log[-5:]]
    print(f"prompt {p['prompt_index']}: {len(log)} retrieval events; last-5 picks {picks}")

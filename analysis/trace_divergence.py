"""Where along the forward pass does a failing decision diverge from its
counterfactual match?

For every decision taken in a failing attempt, find the nearest decision among
the successful attempts of the same task -- nearest in true input space, joint
state plus the three camera frames -- and walk the forward pass in order,
measuring how far apart the pair is at each stage. Matching on the input is what
removes scene difficulty as a confounder: the two decisions start from nearly the
same place, so whatever separates them afterwards arose inside the computation.

Every gap is read against a baseline built the same way from success-success
pairs. Two similar successful decisions also differ; only the excess over that
counts. The earliest stage whose excess is large is the divergence point.

Matching never uses the model's own image embedding. That is already the output
of perception, and matching on it would condition perception out of suspicion
before the trace begins.
"""

import argparse
import glob
import json

import numpy as np

STAGES = ["perception", "language", "routing", "denoise", "action"]
KMAX = 8
IMG_DS = 8


def load_task(task, rollouts):
    """Per-decision inputs and internals for one task."""
    rows = []
    for f in sorted(glob.glob(f"{rollouts}/{task}_*/episodes/*.npz")):
        meta = json.load(open(f.replace(".npz", ".json")))
        with np.load(f) as z:
            if "decision_images_head_camera" not in z.files or "intro_h_query" not in z.files:
                continue
            n = min(KMAX, z["decision_states"].shape[0])
            img = np.concatenate(
                [z[f"decision_images_{c}"][:n, ::IMG_DS, ::IMG_DS, :]
                 .astype(np.float32).reshape(n, -1) / 255.0
                 for c in ("head_camera", "left_camera", "right_camera")], axis=1)
            rows.append(dict(
                ep=len(rows), fail=int(not meta["success"]), n=n,
                state=z["decision_states"][:n].astype(np.float32),
                img=img,
                perception=z["intro_h_query"][:n].astype(np.float32),
                language=z["intro_h_lang"][:n].astype(np.float32),
                # routing as the per-layer entropy profile, denoise as the whole
                # trajectory: comparing the full vector rather than a summary
                # keeps the comparison from depending on one scalar reduction.
                routing=z["intro_router_entropy"][:n].astype(np.float32).reshape(n, -1),
                denoise=z["intro_denoise_x"][:n].astype(np.float32).reshape(n, -1),
                action=z["predicted_chunks"][:n].astype(np.float32).reshape(n, -1),
            ))
    return rows


def stack(rows, key):
    return np.concatenate([r[key] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="/data/whn/robotwin_eval/rollouts")
    ap.add_argument("--tasks", nargs="*", default=["click_bell", "click_alarmclock",
                    "open_microwave", "hanging_mug", "place_can_basket", "stamp_seal"])
    ap.add_argument("--img-weight", type=float, default=0.5,
                    help="weight on the image half of the matching distance")
    cfg = ap.parse_args()

    print(f"匹配距离中图像占 {cfg.img_weight:.0%}，关节状态占 {1-cfg.img_weight:.0%}\n")
    per_task = {}

    for task in cfg.tasks:
        rows = load_task(task, cfg.rollouts)
        if len(rows) < 4:
            continue
        ep = np.concatenate([np.full(r["n"], r["ep"]) for r in rows])
        fail = np.concatenate([np.full(r["n"], r["fail"]) for r in rows])

        S = stack(rows, "state"); I = stack(rows, "img")
        Sz = (S - S.mean(0)) / (S.std(0) + 1e-6)
        Iz = (I - I.mean(0)) / (I.std(0) + 1e-6)
        X = np.concatenate([
            np.sqrt(1 - cfg.img_weight) * Sz / np.sqrt(Sz.shape[1]),
            np.sqrt(cfg.img_weight) * Iz / np.sqrt(Iz.shape[1])], axis=1)

        # Each stage standardised over this task, so gaps are comparable across
        # stages of very different dimension and scale.
        St = {}
        for s in STAGES:
            V = stack(rows, s)
            St[s] = (V - V.mean(0)) / (V.std(0) + 1e-6) / np.sqrt(V.shape[1])

        f_idx, s_idx = np.where(fail == 1)[0], np.where(fail == 0)[0]
        if len(f_idx) < 5 or len(s_idx) < 5:
            continue

        def match(q_idx, bank_idx):
            """Nearest bank row for each query, excluding its own episode."""
            out = np.empty(len(q_idx), dtype=int)
            B = X[bank_idx]
            for i, qi in enumerate(q_idx):
                d = np.linalg.norm(B - X[qi], axis=1)
                d[ep[bank_idx] == ep[qi]] = np.inf
                out[i] = bank_idx[int(d.argmin())]
            return out

        m_fs = match(f_idx, s_idx)     # failure -> nearest success
        m_ss = match(s_idx, s_idx)     # success -> nearest success (baseline)

        gaps_f = {s: np.linalg.norm(St[s][f_idx] - St[s][m_fs], axis=1) for s in STAGES}
        gaps_s = {s: np.linalg.norm(St[s][s_idx] - St[s][m_ss], axis=1) for s in STAGES}

        print(f"=== {task} ===  失败决策 {len(f_idx)} 个，成功决策 {len(s_idx)} 个")
        print(f"{'环节':<14}{'失败↔成功':>12}{'成功↔成功':>12}{'超出倍数':>10}")
        print("-" * 50)
        rec = {}
        for s in STAGES:
            a, b = np.median(gaps_f[s]), np.median(gaps_s[s])
            r = a / (b + 1e-9)
            rec[s] = r
            mark = "  <--" if r > 1.10 else ""
            print(f"{s:<14}{a:>12.3f}{b:>12.3f}{r:>10.3f}{mark}")
        per_task[task] = rec
        first = [s for s in STAGES if rec[s] > 1.10]
        print(f"  最早显著分岔: {first[0] if first else '无（各环节都在基线范围内）'}\n")

    print("=== 汇总：各环节超出基线的倍数 ===")
    print(f"{'任务':<20}" + "".join(f"{s[:9]:>12}" for s in STAGES))
    print("-" * (20 + 12 * len(STAGES)))
    for t, rec in per_task.items():
        print(f"{t:<20}" + "".join(f"{rec[s]:>12.3f}" for s in STAGES))
    if per_task:
        print(f"{'平均':<20}" + "".join(
            f"{np.mean([r[s] for r in per_task.values()]):>12.3f}" for s in STAGES))
    print("\n  倍数 ≈ 1 表示该环节和两个成功决策之间的差异一样大，即未分岔；")
    print("  沿顺序第一个明显大于 1 的环节，就是失败开始偏离的地方。")


if __name__ == "__main__":
    main()

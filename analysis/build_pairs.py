"""Build counterfactual pairs and measure how far apart each component is.

For every decision taken in a failing attempt, the nearest decision among the
successful attempts of the same task -- nearest in true input space, the 14 joint
values plus the three camera frames. Matching on the input is what makes the
comparison fair: the pair starts from nearly the same place, so scene difficulty
is held fixed rather than modelled, and whatever separates them afterwards arose
inside the computation.

Success-success pairs are built the same way and carry the normal spread. Raw
gaps and match distances are saved rather than percentiles, because the
comparison is only meaningful between pairings in equally similar states: a
success-success pairing whose two decisions are far apart says nothing about how
much two decisions in the *same* state normally differ. The distance cut and the
conversion to percentiles therefore happen downstream, where both arms can be
restricted to the same distance regime at once.

Matching never uses the model's own image embedding: that is already the output
of perception, and matching on it would condition perception out of suspicion.
"""

import argparse
import glob
import json

import numpy as np

# The component set the Bayesian model reasons over. noise and routing are the
# two that can actually be borrowed from the matched success; the rest are kept
# so the model can say "this failure is not one the available corrections reach".
COMPONENTS = ["perception", "language", "routing", "noise", "denoise", "execution"]
KMAX = 8
IMG_DS = 8


def load_task(task, rollouts):
    rows = []
    for f in sorted(glob.glob(f"{rollouts}/{task}_*/episodes/*.npz")):
        meta = json.load(open(f.replace(".npz", ".json")))
        with np.load(f) as z:
            need = ("decision_images_head_camera", "intro_h_query", "intro_noise")
            if any(k not in z.files for k in need):
                continue
            n = min(KMAX, z["decision_states"].shape[0])
            img = np.concatenate(
                [z[f"decision_images_{c}"][:n, ::IMG_DS, ::IMG_DS, :]
                 .astype(np.float32).reshape(n, -1) / 255.0
                 for c in ("head_camera", "left_camera", "right_camera")], axis=1)

            states = z["states"].astype(np.float32)
            ex = z["executed_actions"].astype(np.float32)
            st = z["decision_steps"]
            h = int(np.median(np.diff(st))) if len(st) > 2 else 25
            T = min(len(states) - 1, len(ex))
            step_err = (np.abs(states[1:T + 1] - ex[:T]).mean(axis=1)
                        if T > 0 else np.zeros(1))
            ex_err = np.zeros((n, 1), np.float32)
            for k in range(n):
                lo, hi = k * h, min((k + 1) * h, T)
                if hi > lo:
                    ex_err[k, 0] = step_err[lo:hi].mean()

            rows.append(dict(
                ep=len(rows), fail=int(not meta["success"]), n=n, seed=meta["seed"],
                state=z["decision_states"][:n].astype(np.float32), img=img,
                perception=z["intro_h_query"][:n].astype(np.float32),
                language=z["intro_h_lang"][:n].astype(np.float32),
                routing=z["intro_router_entropy"][:n].astype(np.float32).reshape(n, -1),
                noise=z["intro_noise"][:n].astype(np.float32).reshape(n, -1),
                # the denoising path with its starting point removed, so this is
                # what the solver did rather than which noise it started from
                denoise=(z["intro_denoise_x"][:n].astype(np.float32)
                         - z["intro_denoise_x"][:n, :1].astype(np.float32)).reshape(n, -1),
                execution=ex_err,
            ))
    return rows


def match_task(rows, img_weight):
    """Input-space nearest-neighbour matching for one task's decisions.

    Shared by the pair builder and the correction prep so the two can never
    disagree about which success a failure was matched to.
    """
    ep = np.concatenate([np.full(r["n"], r["ep"]) for r in rows])
    fail = np.concatenate([np.full(r["n"], r["fail"]) for r in rows])
    S = np.concatenate([r["state"] for r in rows])
    I = np.concatenate([r["img"] for r in rows])
    Sz = (S - S.mean(0)) / (S.std(0) + 1e-6)
    Iz = (I - I.mean(0)) / (I.std(0) + 1e-6)
    X = np.concatenate([
        np.sqrt(1 - img_weight) * Sz / np.sqrt(Sz.shape[1]),
        np.sqrt(img_weight) * Iz / np.sqrt(Iz.shape[1])], axis=1)

    f_idx, s_idx = np.where(fail == 1)[0], np.where(fail == 0)[0]

    def match(q, bank):
        out = np.empty(len(q), dtype=int)
        dist = np.empty(len(q))
        B = X[bank]
        for i, qi in enumerate(q):
            d = np.linalg.norm(B - X[qi], axis=1)
            d[ep[bank] == ep[qi]] = np.inf
            j = int(d.argmin())
            out[i], dist[i] = bank[j], d[j]
        return out, dist

    m_f, dist_f = match(f_idx, s_idx) if len(f_idx) and len(s_idx) else (np.empty(0, int), np.empty(0))
    m_s, dist_s = match(s_idx, s_idx) if len(s_idx) else (np.empty(0, int), np.empty(0))
    return dict(ep=ep, fail=fail, f_idx=f_idx, s_idx=s_idx,
                m_f=m_f, dist_f=dist_f, m_s=m_s, dist_s=dist_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="/data/whn/robotwin_eval/rollouts")
    ap.add_argument("--tasks", nargs="*", default=["click_bell", "click_alarmclock",
                    "open_microwave", "hanging_mug", "place_can_basket", "stamp_seal"])
    ap.add_argument("--img-weight", type=float, default=0.5)
    ap.add_argument("--out", default="/data/whn/robotwin_eval/pairs.npz")
    cfg = ap.parse_args()

    A, Y, TID, PAIR, MDIST, EPI = [], [], [], [], [], []
    ep_base = 0
    tasks_used = []
    for task in cfg.tasks:
        rows = load_task(task, cfg.rollouts)
        if len(rows) < 6:
            continue
        mt = match_task(rows, cfg.img_weight)
        ep, fail = mt["ep"], mt["fail"]
        f_idx, s_idx = mt["f_idx"], mt["s_idx"]
        m_f, dist_f, m_s, dist_s = mt["m_f"], mt["dist_f"], mt["m_s"], mt["dist_s"]
        comp = {c: np.concatenate([r[c] for r in rows]) for c in COMPONENTS}
        if len(f_idx) < 5 or len(s_idx) < 5:
            continue

        for q, m, dist, lab in ((f_idx, m_f, dist_f, 1), (s_idx, m_s, dist_s, 0)):
            a = np.empty((len(q), len(COMPONENTS)), np.float32)
            for j, c in enumerate(COMPONENTS):
                a[:, j] = np.linalg.norm(comp[c][q] - comp[c][m], axis=1)
            A.append(a)
            # Episode id, so decisions of one attempt can be aggregated later:
            # they share an outcome and are not independent observations.
            EPI.append(ep[q] + ep_base)
            Y.append(np.full(len(q), lab))
            TID.append(np.full(len(q), len(tasks_used)))
            PAIR.append(np.stack([q, m], axis=1))
            MDIST.append(dist)
        ep_base += int(ep.max()) + 1
        tasks_used.append(task)
        print(f"{task:<20} 失败决策 {len(f_idx):>4}  成功决策 {len(s_idx):>4}  "
              f"匹配距离中位数 失败 {np.median(dist_f):.3f} / 成功 {np.median(dist_s):.3f}")

    A = np.concatenate(A); Y = np.concatenate(Y)
    TID = np.concatenate(TID); PAIR = np.concatenate(PAIR); MDIST = np.concatenate(MDIST)
    EPI = np.concatenate(EPI)
    np.savez_compressed(cfg.out, a=A, y=Y, task_id=TID, pair=PAIR, match_dist=MDIST,
                        episode=EPI,
                        components=np.array(COMPONENTS), tasks=np.array(tasks_used))
    print(f"\nwrote {cfg.out}: {len(Y)} 个配对（失败 {int(Y.sum())}）\n")

    print("保存的是原始差异和匹配距离；按距离筛选与换算成分位在 fit_pairs.py 中进行，")
    print("以保证失败配对和成功配对被限制在同样的「状态相近」范围内。")


if __name__ == "__main__":
    main()

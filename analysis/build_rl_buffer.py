"""Build the offline RL transition buffer from the recorded corpus.

Decision-level semi-MDP: one transition per policy decision (25 executed ticks).
State features are the frozen VLM's own prefix readouts captured at inference
time (h_image / h_lang / h_query, 2560-d each) plus the 14-d joint state -- so
critic training never needs to run the VLM. The action is stored both ways:
the policy-native normalized chunk (50x55) and the physically-executed prefix
(25x14); the unexecuted tail of a chunk cannot affect the environment, so a
critic on the executed prefix loses nothing.

Reward is the terminal task outcome only (RoboTwin's check_success is a binary
contact latch; the envs expose no stage score -- verified across envs/). With
4-12 decisions per episode the sparse signal is shallow enough for Monte-Carlo
returns, which is what the Phase-1 critic gate uses.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

FEATS = ("intro_h_image", "intro_h_lang", "intro_h_query")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=["click_bell", "click_alarmclock"])
    ap.add_argument("--roots", nargs="*",
                    default=["/data/whn/robotwin_eval/rollouts",
                             "/data/whn/robotwin_eval/rollouts_labelled"])
    ap.add_argument("--outdir", default="/data/whn/robotwin_eval/rl_buffer")
    cfg = ap.parse_args()
    out = Path(cfg.outdir)
    out.mkdir(parents=True, exist_ok=True)

    for task in cfg.tasks:
        files = []
        for root in cfg.roots:
            files += sorted(glob.glob(f"{root}/{task}_*/episodes/*.npz")
                            + glob.glob(f"{root}/{task}_*/*/episodes/*.npz"))
        rows = {k: [] for k in
                ("h_image", "h_lang", "h_query", "h_query_tokens", "state",
                 "chunk_norm", "chunk_exec",
                 "episode", "decision", "n_decisions", "success", "seed", "terminal")}
        n_ep = 0
        skipped = 0
        seen = set()
        for f in files:
            meta = json.load(open(f.replace(".npz", ".json")))
            # the same (seed, replicate-0 noise) scene can appear in both roots;
            # keep the first occurrence so no episode is double-counted
            key = (int(meta["seed"]), meta["instruction"])
            if key in seen:
                skipped += 1
                continue
            with np.load(f) as z:
                if not all(k in z.files for k in FEATS):
                    skipped += 1
                    continue
                n = int(meta["num_decisions"])
                seen.add(key)
                rows["h_image"].append(z["intro_h_image"][:n])
                rows["h_lang"].append(z["intro_h_lang"][:n])
                rows["h_query"].append(z["intro_h_query"][:n])
                # unpooled spatial query tokens (8 x 2560): the Phase-1 gate
                # showed pooled means poison state-action fusion; these carry
                # the spatial scene readout the critic actually needs
                rows["h_query_tokens"].append(z["intro_h_query_tokens"][:n])
                rows["state"].append(z["decision_states"][:n])
                rows["chunk_norm"].append(z["predicted_chunks_normalized"][:n].astype(np.float16))
                rows["chunk_exec"].append(z["predicted_chunks"][:n, :25].astype(np.float32))
                rows["episode"].append(np.full(n, n_ep, dtype=np.int32))
                rows["decision"].append(np.arange(n, dtype=np.int32))
                rows["n_decisions"].append(np.full(n, n, dtype=np.int32))
                rows["success"].append(np.full(n, bool(meta["success"])))
                rows["seed"].append(np.full(n, int(meta["seed"]), dtype=np.int64))
                term = np.zeros(n, dtype=bool)
                term[-1] = True
                rows["terminal"].append(term)
            n_ep += 1
        arrs = {k: np.concatenate(v) for k, v in rows.items()}
        n = len(arrs["state"])
        succ_ep = int(sum(a[0] for a in rows["success"]))
        np.savez_compressed(out / f"{task}.npz", **arrs)
        print(f"{task}: {n_ep} 集 ({succ_ep} 成功) -> {n} 决策转移, 跳过 {skipped}"
              f"  [{out / f'{task}.npz'}]")


if __name__ == "__main__":
    main()

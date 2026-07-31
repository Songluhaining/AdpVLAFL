"""Download everything needed for a RoboTwin 2.0 closed-loop eval of LingBot-VLA 2.0.

Deliberately does NOT pull the RoboTwin demonstration dataset (multiple TB) -
closed-loop evaluation only needs the simulator assets.
"""

import os
import sys
import time

# Route everything through the domestic HF mirror on a direct connection: it is
# faster than the SOCKS proxy here and, more importantly, spends no proxy quota.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# Xet dedup transfers go to transfer.xethub.hf.co, which the mirror does not
# front - without this the download stalls at a few MB per file.
os.environ["HF_HUB_DISABLE_XET"] = "1"
for _var in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
    os.environ.pop(_var, None)

from huggingface_hub import snapshot_download

ROOT = "/data/whn/robotwin_eval"

JOBS = [
    dict(
        name="lingbot-vla-v2-6b-robotwin (post-trained ckpt, ~25.5GB)",
        repo_id="robbyant/lingbot-vla-v2-6b-robotwin",
        repo_type="model",
        local_dir=f"{ROOT}/models/lingbot-vla-v2-6b-robotwin",
        allow_patterns=None,
    ),
    dict(
        name="Qwen3-VL-4B-Instruct (base VLM for config/processor, ~8.9GB)",
        repo_id="Qwen/Qwen3-VL-4B-Instruct",
        repo_type="model",
        local_dir=f"{ROOT}/models/Qwen3-VL-4B-Instruct",
        allow_patterns=None,
    ),
    dict(
        name="RoboTwin2.0 sim assets (embodiments + objects + background_texture, ~15GB)",
        repo_id="TianxingChen/RoboTwin2.0",
        repo_type="dataset",
        local_dir=f"{ROOT}/robotwin_assets",
        allow_patterns=["embodiments.zip", "objects.zip", "background_texture.zip"],
    ),
]


def main():
    only = sys.argv[1:] or None
    for job in JOBS:
        if only and not any(o in job["repo_id"] for o in only):
            continue
        print(f"\n===== {job['name']} =====", flush=True)
        t0 = time.time()
        for attempt in range(1, 21):
            try:
                snapshot_download(
                    repo_id=job["repo_id"],
                    repo_type=job["repo_type"],
                    local_dir=job["local_dir"],
                    allow_patterns=job["allow_patterns"],
                    max_workers=1,
                )
                break
            except Exception as e:  # network flakiness through the proxy is expected
                print(f"[attempt {attempt}] {type(e).__name__}: {e}", flush=True)
                time.sleep(10)
        else:
            print(f"FAILED: {job['repo_id']}", flush=True)
            continue
        print(f"done in {time.time() - t0:.0f}s -> {job['local_dir']}", flush=True)


if __name__ == "__main__":
    main()

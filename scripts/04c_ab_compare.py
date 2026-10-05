"""Step 4c — side-by-side A/B evaluation: base Qwen3-4B vs the LoRA fine-tuned model.

Invokes BOTH deployed endpoints on the same held-out test contracts and scores
every answer with the workshop's scorer logic (contractnli_scorer.score_record),
locally — no managed evaluation pipeline or service quota required.

Prereqs: 04_deploy_sft.py deploy AND 04b_deploy_base.py deploy have both run.

Usage:
    python 04c_ab_compare.py          # 10 test contracts
    python 04c_ab_compare.py 25       # custom count
"""
import json
import sys
import time
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import boto3
from botocore.config import Config

import config

sys.path.insert(0, config.LAB1_DIR)
import contractnli as C  # noqa: E402
import contractnli_scorer as S  # noqa: E402

N_DOCS = int(sys.argv[1]) if len(sys.argv) > 1 else 10

ENDPOINTS = {
    "base": (
        "huggingface-reasoning-qwen3-4b-contractnli-base-ep",
        "huggingface-reasoning-qwen3-4b-contractnli-base-ic",
    ),
    "sft": (
        "huggingface-reasoning-qwen3-4b-contractnli-sft-ep",
        "huggingface-reasoning-qwen3-4b-contractnli-sft-ic",
    ),
}

C.ensure_dataset("./data")
test_docs, labels = C.load("test")
docs = test_docs[:N_DOCS]

smr = boto3.client("sagemaker-runtime", region_name=config.AWS_REGION,
                   config=Config(read_timeout=300, retries={"total_max_attempts": 3}))


def gold_reference(doc):
    """Expert annotation in the scorer's label/evidence format."""
    return {k: {"label": v["choice"], "evidence": list(v["spans"])}
            for k, v in C.gold_for(doc).items()}


def invoke(which, doc):
    ep, ic = ENDPOINTS[which]
    body = {
        "model_name": ic,
        "messages": [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                     for m in C.build_messages(doc, labels)],
        "max_tokens": 4000, "temperature": 0.0, "stop": ["<|im_end|>"], "stream": False,
    }
    t0 = time.time()
    last = None
    for attempt in range(4):
        try:
            resp = smr.invoke_endpoint(EndpointName=ep, InferenceComponentName=ic,
                                       ContentType="application/json", Body=json.dumps(body))
            text = json.loads(resp["Body"].read())["choices"][0]["message"]["content"]
            return text, time.time() - t0
        except Exception as e:  # container-side timeouts under load: back off and retry
            last = e
            print(f"[{which}] doc retry {attempt + 1}: {type(e).__name__}", flush=True)
            time.sleep(30 * (attempt + 1))
    raise last


jobs = [(which, i, doc) for i, doc in enumerate(docs) for which in ENDPOINTS]
results = {which: [] for which in ENDPOINTS}

with ThreadPoolExecutor(max_workers=1) as pool:
    futs = {pool.submit(invoke, which, doc): (which, i) for which, i, doc in jobs}
    for fut in as_completed(futs):
        which, i = futs[fut]
        try:
            text, elapsed = fut.result()
        except Exception as e:
            print(f"[{which}] doc{i}: FAILED {type(e).__name__} (scored 0)")
            results[which].append((i, {"label_correct": 0.0, "evidence_f1": 0.0,
                                       "reward": 0.0, "json_valid": 0.0, "secs": 0.0}))
            continue
        record = {"model_response": text, "response": json.dumps(gold_reference(docs[i]))}
        score = S.score_record(record, index=i)
        row = {m["name"]: m["value"] for m in score["metrics_list"]}
        row["reward"] = score["aggregate_reward_score"]
        row["secs"] = round(elapsed, 1)
        results[which].append((i, row))
        print(f"[{which}] doc{i}: label {row['label_correct']:.2f} "
              f"evF1 {row['evidence_f1']:.2f} json {row['json_valid']:.0f} ({elapsed:.0f}s)")

print("\n==== A/B SUMMARY (avg over %d contracts) ====" % len(docs))
for which in ENDPOINTS:
    rows = [r for _, r in sorted(results[which])]
    avg = lambda k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
    print(f"{which.upper():>4}: label accuracy {avg('label_correct'):.3f} | "
          f"evidence F1 {avg('evidence_f1'):.3f} | reward {avg('reward'):.3f} | "
          f"valid JSON {avg('json_valid') * 100:.0f}%")

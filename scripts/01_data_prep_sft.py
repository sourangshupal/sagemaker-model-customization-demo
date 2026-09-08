"""Step 1 — Lab 1 data prep: ContractNLI -> S3 -> SageMaker AI Registry.

Reuses the workshop's dataset helpers (contractnli.py) from your local
WORKSHOP_DIR checkout. Idempotent: re-registering an existing dataset name
fails, so re-runs should use new names or delete old registrations.
"""
import json
import pathlib
import shutil
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import boto3
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.dataset_utils import CustomizationTechnique
from sagemaker.core.helper.session_helper import Session

import config

sys.path.insert(0, config.LAB1_DIR)
import contractnli as C  # noqa: E402

DATASET_PREFIX = "contractnli-nda-review"

sess = Session()
bucket_name = sess.default_bucket()
default_prefix = sess.default_bucket_prefix
s3_client = boto3.client("s3")
print(f"bucket: {bucket_name}  region: {sess.boto_region_name}")

C.ensure_dataset("./data")
train_docs, labels = C.load("train")
dev_docs, _ = C.load("dev")
test_docs, _ = C.load("test")
print(f"train {len(train_docs)} | dev {len(dev_docs)} | test {len(test_docs)}")

label_keys = list(labels.keys())


def gold_json(doc):
    g = C.gold_for(doc)
    return json.dumps({k: {"label": g[k]["choice"], "evidence": list(g[k]["spans"])}
                       for k in label_keys if k in g})


records = {
    "train": [{"prompt": C.build_prompt(d, labels, no_think=False), "completion": gold_json(d)} for d in train_docs],
    "val":   [{"prompt": C.build_prompt(d, labels, no_think=False), "completion": gold_json(d)} for d in dev_docs],
    "test":  [{"query": C.build_prompt(d, labels), "response": gold_json(d)} for d in test_docs],
}
for name, rows in records.items():
    field = "query" if name == "test" else "prompt"
    avg = sum(len(r[field]) for r in rows) // len(rows)
    print(f"{name:5s}: {len(rows):4d} records, avg {avg} chars")

local = pathlib.Path("./sft_data")
if local.exists():
    shutil.rmtree(local)
for name, rows in records.items():
    d = local / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "dataset.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

input_path = (f"{default_prefix}/datasets/{DATASET_PREFIX}" if default_prefix
              else f"datasets/{DATASET_PREFIX}")
s3_paths = {}
for name in records:
    key = f"{input_path}/{name}/dataset.jsonl"
    s3_client.upload_file(str(local / name / "dataset.jsonl"), bucket_name, key)
    s3_paths[name] = f"s3://{bucket_name}/{key}"
    print("uploaded:", s3_paths[name])


def register(name, source, technique=None):
    kwargs = dict(name=name, source=source, wait=True, role=config.ROLE_ARN)
    if technique is not None:
        kwargs["customization_technique"] = technique
    ds = DataSet.create(**kwargs)
    print(f"created dataset: {name}")
    return ds


register(f"{DATASET_PREFIX}-train", s3_paths["train"], CustomizationTechnique.SFT)
register(f"{DATASET_PREFIX}-val", s3_paths["val"], CustomizationTechnique.SFT)
register(f"{DATASET_PREFIX}-test", s3_paths["test"])
print("DATA PREP DONE")

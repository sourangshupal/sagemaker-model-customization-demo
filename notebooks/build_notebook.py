"""Builds SageMaker_AI_Model_Customization_Beginner.ipynb.

Run:  python notebooks/build_notebook.py
The generated notebook is committed to the repo; edit THIS file, then rebuild.
"""
import pathlib

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []  # cells


def md(s):
    C.append(("md", s.strip("\n")))


def code(s):
    C.append(("code", s.strip("\n")))


# ═══════════════════════════════ PART 0 ═══════════════════════════════

md("""
# Customizing LLMs with Amazon SageMaker AI — A Beginner's Hands-On Lab

**What you will do in this notebook:**
1. Teach a 4-billion-parameter model to review NDAs like a legal checklist (**supervised fine-tuning with LoRA**)
2. Verify the improvement with a **managed evaluation** that scores the base and fine-tuned model side by side
3. **Deploy** your fine-tuned model and talk to it over an OpenAI-compatible API
4. Teach a 1B chat model to sound human instead of robotic (**DPO — Direct Preference Optimization**)
5. Measure that with an **LLM-as-a-judge** evaluation

**How you pay:** training is *serverless* — you never pick an instance, and you are billed per token
processed. The whole lab costs **roughly $5–10** if you delete resources when told to.

**Where it runs:** in **SageMaker Studio** (JupyterLab) or on your own machine with the AWS CLI configured.
Use region `us-east-1`, `us-west-2`, `eu-west-1`, or `ap-northeast-1` (serverless customization is
available only in these regions).

> 🎓 **Teaching companion:** this notebook accompanies the YouTube walkthrough and the
> [demo scripts repo](https://github.com/sourangshupal/sagemaker-model-customization-demo).
> Every failure you might hit has a ⚠️ box — they were all discovered by actually running this lab.
""")

md("""
## Part 0 — Setup & safety checks (costs nothing)

We verify everything **before** spending a cent. Beginners rarely fail on ML concepts —
they fail on permissions, quotas, and missing resources. These five checks prevent that.
""")

code(r"""
# ▶ RUN — installs the libraries used throughout the lab (~2 min, no cost)
%pip install -q "sagemaker>=3.16" boto3 datasets==5.0.0 pandas scikit-learn pyarrow tqdm matplotlib
""")

md("""
### 0.1 — Create a session and find your execution role

In SageMaker Studio, `get_execution_role()` returns your domain's role automatically.
On your own machine, set the `ROLE_ARN` environment variable instead (see the repo's
`.env.example`).
""")

code(r"""
# ▶ RUN — no cost
import os
import boto3
from sagemaker.core.helper.session_helper import Session, get_execution_role

sess = Session()
REGION = sess.boto_region_name
BUCKET = sess.default_bucket()

try:
    ROLE_ARN = get_execution_role()          # works inside SageMaker Studio
    print("Running in Studio — using domain execution role")
except ValueError:
    ROLE_ARN = os.environ.get("ROLE_ARN", "") # works on your local machine
    if not ROLE_ARN:
        raise SystemExit("Set the ROLE_ARN env var (copy .env.example to .env). See the repo README.")

print(f"region : {REGION}")
print(f"bucket : {BUCKET}")
print(f"role   : {ROLE_ARN}")
""")

md("""
> ⚠️ **Common error — `Could not assume role .../AmazonSageMaker-ExecutionRole-...`**
> Console-created roles live under a `/service-role/` path. The ARN must include it:
> `arn:aws:iam::<account>:role/service-role/AmazonSageMaker-ExecutionRole-...`
> Run `aws iam get-role --role-name <role-name>` to see the exact ARN.
""")

md("""
### 0.2 — Pre-flight checks: region, quotas, MLflow

Two **account quotas** control this lab, and one of them defaults to **zero**:

| Quota (Service Quotas → SageMaker) | Code | Default | Needed for |
|---|---|---|---|
| Concurrent model **customization** serverless jobs | `L-92BDBBEF` | 1 | Training (run jobs sequentially) |
| Concurrent model **evaluation** serverless jobs | `L-619D690E` | **0** | Parts 3 & 5 — **must be increased** |

Also, managed evaluation pipelines log to a **Managed MLflow tracking server**, which must exist.
""")

code(r"""
# ▶ RUN — no cost. Any ❌ below must be fixed before continuing.
import botocore.session

checks = []
checks.append(("Region is supported",
               REGION in ["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-1"],
               f"{REGION} — set AWS_REGION to a supported region if ❌"))

sq = boto3.client("service-quotas", region_name=REGION)
for name, qcode in [("Customization jobs quota (want >= 1)", "L-92BDBBEF"),
                    ("Evaluation jobs quota (want >= 1)", "L-619D690E")]:
    try:
        v = sq.get_service_quota(ServiceCode="sagemaker", QuotaCode=qcode)["Quota"]["Value"]
        checks.append((name, v >= 1,
                       f"current value = {v:.0f}. Request an increase in Service Quotas → "
                       f"AWS services → Amazon SageMaker."))
    except Exception as e:
        checks.append((name, False, f"could not read quota: {e}"))

MLFLOW_ARN = os.environ.get("MLFLOW_TRACKING_SERVER_ARN", "")
sm = boto3.client("sagemaker", region_name=REGION)
try:
    servers = sm.list_mlflow_tracking_servers()["TrackingServerSummaries"]
    ok = bool(servers)
    if ok and not MLFLOW_ARN:
        MLFLOW_ARN = servers[0]["TrackingServerArn"]
    checks.append(("MLflow tracking server exists", ok,
                   f"arn = {MLFLOW_ARN or 'NONE — create one (see README)'} "
                   "Healthy status shows as 'Created'."))
except Exception as e:
    checks.append(("MLflow tracking server exists", False, str(e)))

for name, ok, detail in checks:
    print(f"{'✅' if ok else '❌'}  {name}\n     {detail}")

if not all(ok for _, ok, _ in checks):
    print("\n🛑 Fix the ❌ items above before continuing — do not run paid cells yet.")
else:
    print("\n✅ All checks passed — you're clear to spend money.")
""")

md("""
> 🛑 **Budget alarm (2 minutes, worth it):** AWS Budgets → create a monthly cost budget
> (e.g. $20) with an 80% email alert. Serverless billing lags by hours, so the console
> can show $0 while you're actually spending.
""")

# ═══════════════════════════════ PART 1 ═══════════════════════════════

md("""
## Part 1 — The task and the data (SFT)

**The mission:** build a model that reads a Non-Disclosure Agreement and answers a fixed
17-point legal checklist — *"Is there a non-disclosure obligation? Which paragraph proves it?"* —
returning strict JSON with a verdict and evidence spans.

**The dataset:** **ContractNLI** — 607 real NDAs annotated by legal experts (CC-BY-4.0).
The base model gets only ~65% of checklist items right, so there's real headroom to improve.
""")

md("""
### 1.1 — Load the dataset

We reuse a helper module from the official AWS workshop repo (`contractnli.py`), which
downloads the dataset and builds prompts. If you don't have the repo yet, the cell clones it.
""")

code(r"""
# ▶ RUN — downloads the dataset on first run (~1 min, no cost)
import pathlib, subprocess, sys

WORKSHOP_DIR = pathlib.Path(os.environ.get(
    "WORKSHOP_DIR", "~/generative-ai-on-amazon-sagemaker")).expanduser()
LAB1 = WORKSHOP_DIR / "workshops/serverless-model-customization-with-sagemaker-ai/lab-1-supervised-fine-tuning"

if not (LAB1 / "contractnli.py").exists():
    print("Workshop repo not found — cloning it now...")
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/aws-samples/generative-ai-on-amazon-sagemaker.git",
                    str(WORKSHOP_DIR)], check=True)

sys.path.insert(0, str(LAB1))
import contractnli as C   # the workshop's helper module

C.ensure_dataset("./data")
train_docs, labels = C.load("train")
dev_docs, _ = C.load("dev")
test_docs, _ = C.load("test")
print(f"train {len(train_docs)} contracts | dev {len(dev_docs)} | test {len(test_docs)}")
print(f"checklist items: {len(labels)}")
""")

md("""
> ⚠️ **Common error — `ModuleNotFoundError: No module named 'contractnli'`**
> The `WORKSHOP_DIR` path doesn't point at the aws-samples repo. Check the clone succeeded
> and the path exists.

✅ **Checkpoint:** you should see `train 423 contracts | dev 61 | test 123` and `checklist items: 17`.
""")

md("""
### 1.2 — Look at one example

Understanding *one* example deeply beats skimming a thousand. This is a real contract
from the dataset, its clause split, and the expert's gold answer.
""")

code(r"""
# ▶ RUN — no cost
import json

doc = sorted(test_docs, key=lambda d: len(d["text"]))[1]   # a short-ish one
gold = C.gold_for(doc)

print(f"{doc['file_name']} — {len(doc['text'].split())} words\n")
print("A few clauses:")
for number, text in C.doc_spans(doc)[:3]:
    print(f"  [{number}] {text[:110]}...")

print("\nGold answer (what we want the model to produce):")
for k in sorted(gold, key=lambda x: int(x.split('-')[1]))[:5]:
    v = gold[k]
    ev = f"  evidence clauses {v['spans']}" if v["spans"] else ""
    print(f"  {k}  {v['choice']:14s}{ev}   [{labels[k]['short_description']}]")
""")

md("""
### 1.3 — Build the training records

SageMaker AI's training service expects **prompt/completion JSONL** — one JSON object per line:

```json
{"prompt": "<full contract + checklist + output rules>", "completion": "{\"checklist-1\": {\"label\": \"Entailment\", ...}}"}
```

> 🎓 **Glossary — JSONL:** "JSON Lines". Each line is one independent JSON object. Streaming-friendly,
> the standard format for LLM training data.
>
> ⚠️ **Why formatting is where fine-tuning projects die:** if your records are malformed
> (wrong field names, nested JSON as a string, wrong chat schema), training *runs* — it just
> learns nothing. Always eyeball a record before paying for training.

The evaluation split uses a different schema (`query`/`response`) because the managed scorer
reads that shape — the helper handles it for us.
""")

code(r"""
# ▶ RUN — no cost
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
    print(f"{name:5s}: {len(rows):4d} records, avg prompt {avg:,} chars (~{avg//4:,} tokens)")
""")

md("""
✅ **Checkpoint:** `train: 423 records`, `val: 61`, `test: 123`, average prompts of several
thousand characters. **Notice the token estimate** — it matters next:

> 🎓 **Why we override `dataset_max_len` later:** contracts are long. With the recipe default
> of 4096 tokens, the trainer would *silently drop* any record that doesn't fit — about a
> third of our dataset. We set 16384 so nothing is discarded.
""")

md("""
### 1.4 — Upload to S3 and register in the SageMaker AI Registry

The **AI Registry** versions your datasets so every training run is reproducible:
`DataSet.create(name=..., source=s3://..., customization_technique=SFT)`.
""")

code(r"""
# ▶ RUN — uploads ~2 MB to S3, registers 3 datasets (~2 min, pennies at most)
import shutil
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.dataset_utils import CustomizationTechnique

DATASET_PREFIX = "contractnli-nda-review"
s3_client = boto3.client("s3")
default_prefix = sess.default_bucket_prefix

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
    s3_client.upload_file(str(local / name / "dataset.jsonl"), BUCKET, key)
    s3_paths[name] = f"s3://{BUCKET}/{key}"
    print("uploaded:", s3_paths[name])

training_dataset = DataSet.create(name=f"{DATASET_PREFIX}-train", source=s3_paths["train"],
                                  customization_technique=CustomizationTechnique.SFT,
                                  wait=True, role=ROLE_ARN)
val_dataset = DataSet.create(name=f"{DATASET_PREFIX}-val", source=s3_paths["val"],
                             customization_technique=CustomizationTechnique.SFT,
                             wait=True, role=ROLE_ARN)
test_dataset = DataSet.create(name=f"{DATASET_PREFIX}-test", source=s3_paths["test"],
                              wait=True, role=ROLE_ARN)
print("\n✅ 3 datasets registered in the AI Registry")
""")

# ═══════════════════════════════ PART 2 ═══════════════════════════════

md("""
## Part 2 — Serverless LoRA fine-tuning

> 🎓 **Glossary — LoRA / PEFT:** a 4B model has ~4 billion weights. Full fine-tuning updates
> all of them (huge memory, huge cost). **LoRA** freezes the base model and trains two tiny
> "adapter" matrices per layer (rank 32 here — millions, not billions, of parameters). The
> adapters capture the task; at serving time they're merged back into the base weights.
> **PEFT** = Parameter-Efficient Fine-Tuning, the family of techniques LoRA belongs to.

> 🎓 **What "serverless training" means:** you don't choose an instance type, don't wait for
> capacity, don't configure a cluster. SageMaker AI provisions compute sized to your model
> and dataset, and bills **per token processed**. When the job ends, the compute disappears.

**Hyperparameters we use (and why):**

| Setting | Value | Why |
|---|---|---|
| `lora_rank` / `lora_alpha` | 32 / 64 | adapter size vs. adapter learning scale — a solid default for 4B models |
| `max_epochs` | 8 | 423 records ÷ batch 64 ≈ 7 steps/epoch → 52 optimizer steps total |
| `global_batch_size` | 64 | memory/throughput trade-off; serverless default is fine |
| `dataset_max_len` | 16384 | so long contracts aren't silently dropped (see 1.3) |
""")

code(r"""
# ▶ RUN — creates the model package group, no cost
import hashlib
from botocore.exceptions import ClientError
from sagemaker.core.resources import ModelPackageGroup
from sagemaker.train.common import TrainingType
from sagemaker.train.sft_trainer import SFTTrainer

BASE_MODEL_ID = "huggingface-reasoning-qwen3-4b"   # Qwen3-4B, verified end-to-end in this lab

MAX_MPG = 63
suffix = "-contractnli-sft-mpg"
candidate = f"{BASE_MODEL_ID}{suffix}"
if len(candidate) > MAX_MPG:
    digest = hashlib.sha1(BASE_MODEL_ID.encode()).hexdigest()[:6]
    keep = MAX_MPG - len(suffix) - len(digest) - 1
    MPG = f"{BASE_MODEL_ID[:keep].rstrip('-')}-{digest}{suffix}"
else:
    MPG = candidate

try:
    ModelPackageGroup.get(model_package_group_name=MPG)
    print(f"model package group exists: {MPG}")
except ClientError:
    ModelPackageGroup.create(model_package_group_name=MPG,
                             model_package_group_description="ContractNLI NDA review, serverless SFT")
    print(f"created model package group: {MPG}")
""")

md("""
✅ Checkpoint: you should see `created model package group: huggingface-reasoning-qwen3-4b-contractnli-sft-mpg`
(or `... exists` on a re-run).
""")

code(r"""
# 🛑 BEFORE YOU RUN — this starts paid training (~$2, ~29 min). Make sure Part 0 checks were green.
# ▶ RUN — launches the serverless SFT job
output_path = (f"s3://{BUCKET}/{default_prefix}/{BASE_MODEL_ID}-contractnli"
               if default_prefix else f"s3://{BUCKET}/{BASE_MODEL_ID}-contractnli")

trainer = SFTTrainer(
    model=BASE_MODEL_ID,
    training_type=TrainingType.LORA,
    model_package_group=MPG,
    training_dataset=training_dataset,
    validation_dataset=val_dataset,
    s3_output_path=output_path,
    sagemaker_session=sess,
    role=ROLE_ARN,
    accept_eula=True,          # required for JumpStart models
    base_job_name="contractnli-sft",
)

trainer.hyperparameters.global_batch_size = 64
trainer.hyperparameters.max_epochs = 8
trainer.hyperparameters.learning_rate = 0.0001
trainer.hyperparameters.lr_warmup_steps_ratio = 0.1
trainer.hyperparameters.lora_rank = 32
trainer.hyperparameters.lora_alpha = 64
trainer.hyperparameters.dataset_max_len = 16384

print("hyperparameters:")
for k, v in trainer.hyperparameters.to_dict().items():
    print(f"  {k}: {v}")

training_job = trainer.train(wait=False)
JOB_NAME = training_job.training_job_name
print(f"\nLAUNCHED: {JOB_NAME}")
""")

code(r"""
# ▶ RUN — watch the job (safe to re-run; also safe to close the laptop and come back)
import time
from sagemaker.core.resources import TrainingJob

while True:
    j = TrainingJob.get(training_job_name=JOB_NAME)
    print(f"{time.strftime('%H:%M:%S')}  {j.training_job_status} / {j.secondary_status}")
    if j.training_job_status in ("Completed", "Failed", "Stopped"):
        break
    time.sleep(120)
""")

md("""
**What the statuses mean:** `Downloading` (fetching the base model) → `Training` (the
actual LoRA pass — loss curves are being logged to MLflow automatically) → `Uploading`
(artifacts to S3) → `Completed`.

📊 **Reference results** (measured in a real run of this exact configuration):
**~29 minutes wall-clock**, checklist accuracy **64.5% → 87.4%**, evidence F1 **48.8 → 75.4**
on the 123 held-out contracts. Your numbers will differ slightly — that's fine and a good
discussion point.
""")

# ═══════════════════════════════ PART 3 ═══════════════════════════════

md("""
## Part 3 — Did it actually work? Managed evaluation

**Never trust training loss alone.** We evaluate with a **custom scorer**: a Python
function that checks, for every checklist item, whether the model's *verdict* matches the
expert's **and** whether the cited *evidence clauses* match. We register it once as a
**reward-function evaluator**, then a managed pipeline runs it against **both** the base
model and your fine-tuned model on the same test contracts — an apples-to-apples comparison
you didn't have to build.
""")

code(r"""
# ▶ RUN — registers the scorer (~2 min, no cost)
# ⚠️ Your execution role must be assumable by LAMBDA (trust policy) or this fails with
#    "role defined for the function cannot be assumed by Lambda". See the repo README.
from sagemaker.ai_registry.air_constants import REWARD_FUNCTION
from sagemaker.ai_registry.evaluator import Evaluator

SCORER_NAME = "contractnli-scorer"
try:
    scorer = Evaluator.get(name=SCORER_NAME)
    print("reusing registered scorer")
except Exception:
    scorer = Evaluator.create(name=SCORER_NAME, type=REWARD_FUNCTION,
                              source=str(LAB1 / "contractnli_scorer.py"),
                              role=ROLE_ARN, sagemaker_session=sess, wait=True)
    print("registered scorer")
""")

code(r"""
# 🛑 BEFORE YOU RUN — starts a paid evaluation job (15–20 min, low single-digit $).
# Requires: evaluation quota >= 1 (Part 0 check) and an MLflow tracking server.
# ▶ RUN — launches base-vs-tuned managed evaluation
from sagemaker.train.evaluate import CustomScorerEvaluator

resp = sm.list_model_packages(ModelPackageGroupName=MPG, SortBy="CreationTime",
                              SortOrder="Descending", MaxResults=1)
model_package_arn = resp["ModelPackageSummaryList"][0]["ModelPackageArn"]
print(f"fine-tuned model: {model_package_arn}")

scorer_eval = CustomScorerEvaluator(
    evaluator=scorer,
    mlflow_resource_arn=MLFLOW_ARN,
    dataset=test_dataset,
    model=model_package_arn,
    model_package_group=MPG,
    s3_output_path=f"s3://{BUCKET}/contractnli-scorer-eval",
    evaluate_base_model=True,     # scores the BASE model too, same job, same scorer
    sagemaker_session=sess,
    role=ROLE_ARN,
)
scorer_eval.hyperparameters.max_new_tokens = 8192
scorer_eval.hyperparameters.max_model_len = 24000

scorer_execution = scorer_eval.evaluate()
print("launched:", scorer_execution.arn)
""")

code(r"""
# ▶ RUN — poll to completion (safe to re-run)
import time
while True:
    status = sm.describe_pipeline_execution(
        PipelineExecutionArn=scorer_execution.arn)["PipelineExecutionStatus"]
    print(f"{time.strftime('%H:%M:%S')}  {status}")
    if status != "Executing":
        break
    time.sleep(60)
print("EVAL DONE:", status)
""")

md("""
> ⚠️ **If the pipeline fails instantly with `ResourceLimitExceeded` ... concurrent model
> evaluation serverless jobs ... is 0`** — your evaluation quota is still zero. Request the
> increase (Service Quotas → `L-619D690E`) and re-run. If you can't wait, the workshop
> publishes pre-computed results — see the "Results" section of the workshop's evaluation
> page; the teaching value is in understanding the pipeline, not in re-deriving the numbers.
""")

code(r"""
# ▶ RUN — find and print the aggregate scores written to S3
s3r = boto3.client("s3", region_name=REGION)
prefix = "contractnli-scorer-eval"
keys = [o["Key"] for p in s3r.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix)
        for o in p.get("Contents", [])]
print(f"{len(keys)} result files. Newest:")
for k in keys[-5:]:
    print(" ", k)

# Aggregate metrics usually live in a small JSON — print the first one we can parse.
import re
for k in sorted(keys):
    if k.endswith(".json") and re.search(r"(metric|score|aggregat|summary)", k, re.I):
        obj = json.loads(s3r.get_object(Bucket=BUCKET, Key=k)["Body"].read())
        print(f"\n--- {k} ---")
        print(json.dumps(obj, indent=1)[:2000])
        break
""")

# ═══════════════════════════════ PART 4 ═══════════════════════════════

md("""
## Part 4 — Deploy the model and talk to it

> 🎓 **Merged checkpoint:** the training output includes the base weights with your LoRA
> adapters merged back in — a normal model artifact, servable by any compatible container.

> 🛑 **THE $1/HOUR WARNING:** a running `ml.g5.xlarge` endpoint costs about **$1 every hour,
> including while you sleep**. Part 4 is not finished until you run the teardown cell.

We deploy with the **DJL Inference container running vLLM** — a high-throughput serving
stack — behind a SageMaker **endpoint** with an **inference component** (the scaling unit).
The endpoint speaks the **OpenAI chat schema**, so your existing tooling just works.
""")

code(r"""
# 🛑 BEFORE YOU RUN — endpoint billing starts NOW (~$1/hr until teardown).
# ▶ RUN — creates model, endpoint config, endpoint, inference component (~10–15 min)
from sagemaker.core import s3 as sm_s3
from sagemaker.core.resources import (Endpoint, EndpointConfig, InferenceComponent, Model)
from sagemaker.core.shapes import (ContainerDefinition, InferenceComponentComputeResourceRequirements,
                                   InferenceComponentRuntimeConfig, InferenceComponentSpecification,
                                   ModelDataSource, ProductionVariant, S3ModelDataSource)

model_package = __import__("sagemaker.core.resources", fromlist=["ModelPackage"]).ModelPackage.get(model_package_arn)
merged_uri = sm_s3.s3_path_join(
    model_package.inference_specification.containers[0]
    .model_data_source.s3_data_source.s3_uri, "checkpoints", "hf_merged") + "/"
print(f"merged checkpoint: {merged_uri}")

IMAGE = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/djl-inference:0.36.0-lmi18.0.0-cu128"
MODEL_NAME, CFG_NAME, EP_NAME, IC_NAME = (f"{BASE_MODEL_ID[:40]}-contractnli-sft-{s}"
                                          for s in ["m", "cfg", "ep", "ic"])

env = {"HF_MODEL_ID": "/opt/ml/model", "OPTION_TRUST_REMOTE_CODE": "true",
       "OPTION_MODEL_LOADING_TIMEOUT": "3600", "OPTION_TENSOR_PARALLEL_DEGREE": "max",
       "SERVING_FAIL_FAST": "true", "OPTION_ROLLING_BATCH": "disable",
       "OPTION_ASYNC_MODE": "true", "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service",
       "OPTION_DTYPE": "bf16", "OPTION_MAX_MODEL_LEN": json.dumps(1024 * 32)}

Model.create(model_name=MODEL_NAME,
             primary_container=ContainerDefinition(
                 image=IMAGE,
                 model_data_source=ModelDataSource(
                     s3_data_source=S3ModelDataSource(s3_uri=merged_uri, s3_data_type="S3Prefix",
                                                      compression_type="None")),
                 environment=env),
             execution_role_arn=ROLE_ARN)
EndpointConfig.create(endpoint_config_name=CFG_NAME, execution_role_arn=ROLE_ARN,
                      production_variants=[ProductionVariant(
                          variant_name="AllTraffic", instance_type="ml.g5.xlarge",
                          initial_instance_count=1,
                          model_data_download_timeout_in_seconds=700,
                          container_startup_health_check_timeout_in_seconds=700)])
Endpoint.create(endpoint_name=EP_NAME, endpoint_config_name=CFG_NAME)
Endpoint.get(EP_NAME).wait_for_status("InService")
print("endpoint InService")
""")

code(r"""
# ▶ RUN — creates the inference component (~2 min), then the endpoint is ready
InferenceComponent.create(
    inference_component_name=IC_NAME, endpoint_name=EP_NAME, variant_name="AllTraffic",
    specification=InferenceComponentSpecification(
        model_name=MODEL_NAME,
        compute_resource_requirements=InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=1024, number_of_accelerator_devices_required=1)),
    runtime_config=InferenceComponentRuntimeConfig(copy_count=1), region=REGION)
InferenceComponent.get(IC_NAME).wait_for_status("InService")
print(f"ready — model id: sm:{EP_NAME}/{IC_NAME}@{REGION}")
""")

code(r"""
# ▶ RUN — ask your fine-tuned model about a REAL contract (pennies)
import re
from botocore.config import Config

C.ensure_dataset("./data")
doc, gold = test_docs[0], C.gold_for(test_docs[0])
smr = boto3.client("sagemaker-runtime", region_name=REGION,
                   config=Config(read_timeout=300, retries={"total_max_attempts": 3}))

body = {"model_name": IC_NAME,
        "messages": [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                     for m in C.build_messages(doc, labels)],
        "max_tokens": 4000, "temperature": 0.0, "stop": ["<|im_end|>"], "stream": False}

resp = smr.invoke_endpoint(EndpointName=EP_NAME, InferenceComponentName=IC_NAME,
                           ContentType="application/json", Body=json.dumps(body))
text = json.loads(resp["Body"].read())["choices"][0]["message"]["content"]
print(text[:1500])

# quick validity check
clean = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
m = re.search(r"\{.*\}", clean, re.DOTALL)
print("\nvalid JSON verdict object:", bool(m))
""")

code(r"""
# 🛑🛑 THE MOST IMPORTANT CELL FOR YOUR WALLET — ▶ RUN always, before you walk away
import time
time.sleep(45)   # let in-flight requests drain
for label, fn in [("inference component", lambda: InferenceComponent.get(IC_NAME).delete()),
                  ("endpoint", lambda: Endpoint.get(EP_NAME).delete()),
                  ("endpoint config", lambda: EndpointConfig.get(CFG_NAME).delete()),
                  ("model", lambda: Model.get(MODEL_NAME).delete())]:
    try:
        fn(); print(f"deleted {label}")
    except Exception as e:
        print(f"skip {label}: {e}")
print("✅ TEARDOWN DONE — endpoint billing stopped")
""")

# ═══════════════════════════════ PART 5 ═══════════════════════════════

md("""
## Part 5 — Teaching behavior: DPO (Direct Preference Optimization)

> 🎓 **The idea:** SFT teaches *facts and formats*; DPO teaches *taste*. The dataset is
> triplets — a `prompt`, a `chosen` response, and a `rejected` response. Training nudges the
> model toward chosen and away from rejected. No reward model, no RL loop — one elegant pass.
> When to use which: **SFT** for "answer in this JSON, cite this evidence"; **DPO** for
> tone, helpfulness, and how the model says "I don't know."

**The task:** make a tiny **Llama 3.2 1B Instruct** model sound human instead of
corporate-AI formal, using ~3,000 real human preference pairs.
""")

code(r"""
# ▶ RUN — prepares preference data and registers it (~3 min, pennies)
import datasets
import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from sagemaker.train.dpo_trainer import DPOTrainer

DPO_MODEL = "meta-textgeneration-llama-3-2-1b-instruct"

stream = (load_dataset("HumanLLMs/Human-Like-DPO-Dataset", split="train", streaming=True)
          .take(3000).shuffle(buffer_size=1000))
dset = datasets.Dataset.from_generator(lambda: stream, features=stream.features)
df = pd.DataFrame(dset)
train, val = train_test_split(df, test_size=0.2, random_state=42)
train, test = train_test_split(train, test_size=0.1, random_state=42)
print(f"train {len(train)} | val {len(val)} | test {len(test)}")

def to_dpo(split, test=False):
    ds = Dataset.from_pandas(split, preserve_index=False)
    if test:
        return ds.map(lambda s: {"query": s["prompt"], "response": s["chosen"]},
                      remove_columns=list(ds.features))
    return ds.map(lambda s: {"prompt": s["prompt"], "chosen": s["chosen"], "rejected": s["rejected"]},
                  remove_columns=list(ds.features))

parts = {"train": to_dpo(train), "val": to_dpo(val), "test": to_dpo(test, test=True)}
dpo_paths = {}
for name, ds in parts.items():
    fname = f"humanlike_dpo_{name}.jsonl"
    p = pathlib.Path(f"./dpo_data/{name}"); p.mkdir(parents=True, exist_ok=True)
    ds.to_json(str(p / fname), orient="records")
    key = f"datasets/humanlike-dpo/{name}/{fname}"
    s3_client.upload_file(str(p / fname), BUCKET, key)
    dpo_paths[name] = f"s3://{BUCKET}/{key}"

dpo_train = DataSet.create(name="humanlike-dpo-train", source=dpo_paths["train"],
                           customization_technique=CustomizationTechnique.DPO,
                           wait=True, role=ROLE_ARN)
dpo_val = DataSet.create(name="humanlike-dpo-val", source=dpo_paths["val"],
                         customization_technique=CustomizationTechnique.DPO,
                         wait=True, role=ROLE_ARN)
DataSet.create(name="humanlike-dpo-test", source=dpo_paths["test"], wait=True, role=ROLE_ARN)
print("✅ DPO datasets registered")
""")

code(r"""
# 🛑 BEFORE YOU RUN — paid training (~$1–2, ~20 min). Only ONE serverless customization job
# can run at a time per region, so the SFT job must be finished (it is, if Part 2 completed).
# ▶ RUN — launches serverless DPO
candidate = f"{DPO_MODEL}-dpo"
MPG_DPO = candidate if len(candidate) <= 63 else candidate[:56] + "-dpo"

try:
    ModelPackageGroup.get(model_package_group_name=MPG_DPO)
except ClientError:
    ModelPackageGroup.create(model_package_group_name=MPG_DPO,
                             model_package_group_description="SageMaker serverless DPO")

dpo_trainer = DPOTrainer(
    model=DPO_MODEL,
    training_type=TrainingType.LORA,
    model_package_group=MPG_DPO,
    training_dataset=dpo_train,
    validation_dataset=dpo_val,
    s3_output_path=f"s3://{BUCKET}/{DPO_MODEL}-dpo",
    mlflow_experiment_name="humanlike-llama3-2-1b-dpo",
    base_job_name="dpo-llama32-1b",
    sagemaker_session=sess,
    accept_eula=True,
    role=ROLE_ARN,
)
dpo_job = dpo_trainer.train(wait=False)
DPO_JOB = dpo_job.training_job_name
print(f"LAUNCHED: {DPO_JOB}")
""")

code(r"""
# ▶ RUN — watch the DPO job
while True:
    j = TrainingJob.get(training_job_name=DPO_JOB)
    print(f"{time.strftime('%H:%M:%S')}  {j.training_job_status} / {j.secondary_status}")
    if j.training_job_status in ("Completed", "Failed", "Stopped"):
        break
    time.sleep(120)
""")

md("""
### 5.1 — Measure it: LLM-as-judge with your own rubric

> 🎓 **LLM-as-judge:** a strong model (here **Claude Sonnet 4.5 via Amazon Bedrock**) scores
> your model's outputs against a rubric *you* define. SageMaker AI runs the whole judging
> pipeline and writes structured scores to S3. You define what "good" means; the platform
> does the grading at scale.

Our custom rubric for "human-ness" (plus three built-in metrics):
- **HumanLikeTone** (0–3): does it sound like a person or a press release?
- **ConversationalEngagement** (0–2): does it invite dialogue?
- **AvoidRoboticPatterns** (0–1): penalize "As an AI, I don't have personal experiences..."
""")

code(r"""
# 🛑 BEFORE YOU RUN — paid eval (30–45 min, low single-digit $; judge model calls billed via Bedrock)
# ▶ RUN — LLM-as-judge evaluation of the DPO model
from sagemaker.train.evaluate import LLMAsJudgeEvaluator

resp = sm.list_model_packages(ModelPackageGroupName=MPG_DPO, SortBy="CreationTime",
                              SortOrder="Descending", MaxResults=1)
dpo_pkg_arn = resp["ModelPackageSummaryList"][0]["ModelPackageArn"]

custom_metrics = json.dumps([
    {"customMetricDefinition": {"name": "HumanLikeTone",
        "instructions": ("Evaluate if the response sounds like a friendly human conversation rather than a formal AI "
                         "assistant. Robotic responses use phrases like 'I'm designed to', 'as an AI', or overly "
                         "corporate language. Prompt: {{prompt}}\nResponse: {{prediction}}"),
        "ratingScale": [
            {"definition": "Excellent - completely natural and human", "value": {"floatValue": 3}},
            {"definition": "Good - mostly human-like", "value": {"floatValue": 2}},
            {"definition": "Mixed", "value": {"floatValue": 1}},
            {"definition": "Poor - robotic", "value": {"floatValue": 0}}]}},
    {"customMetricDefinition": {"name": "ConversationalEngagement",
        "instructions": ("Assess engagement: does the response ask follow-up questions, show interest, invite "
                         "dialogue? Prompt: {{prompt}}\nResponse: {{prediction}}"),
        "ratingScale": [
            {"definition": "Highly engaging", "value": {"floatValue": 2}},
            {"definition": "Somewhat engaging", "value": {"floatValue": 1}},
            {"definition": "Not engaging", "value": {"floatValue": 0}}]}},
    {"customMetricDefinition": {"name": "AvoidRoboticPatterns",
        "instructions": ("Penalize 'As an AI/language model', 'I'm designed to', 'I don't have personal experiences'. "
                         "Prompt: {{prompt}}\nResponse: {{prediction}}"),
        "ratingScale": [
            {"definition": "Good - no robotic patterns", "value": {"floatValue": 1}},
            {"definition": "Bad - contains robotic patterns", "value": {"floatValue": 0}}]}},
])

judge = LLMAsJudgeEvaluator(
    model=dpo_pkg_arn,
    model_package_group=ModelPackageGroup.get(MPG_DPO).model_package_group_arn,
    evaluator_model="anthropic.claude-sonnet-4-5-20250929-v1:0",
    dataset=DataSet.get(name="humanlike-dpo-test"),
    builtin_metrics=["Helpfulness", "Relevance", "Coherence"],
    custom_metrics=custom_metrics,
    s3_output_path=f"s3://{BUCKET}/dpo-llmaj-eval",
    evaluate_base_model=False,
    mlflow_resource_arn=MLFLOW_ARN,
    sagemaker_session=sess,
    role=ROLE_ARN,
)
judge_execution = judge.evaluate()
print("launched:", judge_execution.arn)
""")

code(r"""
# ▶ RUN — poll the judge evaluation
while True:
    status = sm.describe_pipeline_execution(
        PipelineExecutionArn=judge_execution.arn)["PipelineExecutionStatus"]
    print(f"{time.strftime('%H:%M:%S')}  {status}")
    if status != "Executing":
        break
    time.sleep(60)
print("JUDGE EVAL DONE:", status)
""")

code(r"""
# ▶ RUN — print judge scores from S3
keys = [o["Key"] for p in s3r.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix="dpo-llmaj-eval")
        for o in p.get("Contents", [])]
print(f"{len(keys)} result files under s3://{BUCKET}/dpo-llmaj-eval")
for k in keys[-5:]:
    print(" ", k)
# Per-record scores live in *_output.jsonl — average a few key metrics if present:
import pandas as pd
score_files = [k for k in keys if k.endswith(".jsonl")]
if score_files:
    df_scores = pd.read_json(s3r.get_object(Bucket=BUCKET, Key=score_files[0])["Body"],
                             lines=True)
    print("\ncolumns:", list(df_scores.columns)[:10])
    numeric = df_scores.select_dtypes("number")
    if len(numeric.columns):
        print("\nmetric means:")
        print(numeric.mean().round(3).to_string())
""")

md("""""
## 🎉 You did it — recap

| Part | What you learned | Real measured cost |
|---|---|---|
| 1–2 | SFT with LoRA, serverless, pay-per-token | ~$2, 29 min |
| 3 | Managed evaluation: custom scorer, base vs tuned | low single-digit $ |
| 4 | vLLM deployment, OpenAI-schema inference, **teardown discipline** | ~$0.27 for 16 min |
| 5 | DPO + LLM-as-judge with custom rubrics | ~$1–2 train + judge calls |

**Where next?** The fourth rung — **RLVR / RLAIF** (reinforcement learning with verifiable
rewards and AI feedback, including writing your *own* reward function) — is
[Lab 3 and Lab 4 of the AWS workshop](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker/tree/main/workshops/serverless-model-customization-with-sagemaker-ai).
Run them the same way you ran this notebook.

**Cleanup checklist (avoid surprise bills):**
- ✅ Endpoint deleted (Part 4 teardown)
- ⬜ Delete the MLflow tracking server when you're fully done:
      `aws sagemaker delete-mlflow-tracking-server --tracking-server-name <name>`
- ⬜ Keep your budget alarm on

**Resources:** [demo scripts repo](https://github.com/sourangshupal/sagemaker-model-customization-demo) ·
[AWS workshop](https://catalog.workshops.aws/workshops/548b5be9-2da8-4c93-82f7-b0b474108ab3/en-US) ·
[SageMaker AI](https://aws.amazon.com/sagemaker/ai/)
""")

# ═══════════════════════════════ BUILD ═══════════════════════════════

for kind, src in C:
    if kind == "md":
        nb.cells.append(nbf.v4.new_markdown_cell(src))
    else:
        nb.cells.append(nbf.v4.new_code_cell(src))

nb.metadata["kernelspec"] = {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.12"}

out = pathlib.Path(__file__).parent / "SageMaker_AI_Model_Customization_Beginner.ipynb"
nbf.write(nb, str(out))
print(f"wrote {out} with {len(nb.cells)} cells")

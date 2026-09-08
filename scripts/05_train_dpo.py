"""Step 5 — Lab 2: DPO data prep + launch a serverless DPO job (Llama 3.2 1B).

Preference optimization on the Human-Like DPO dataset (~3k pairs).
Note: serverless customization allows ONE concurrent job per region — wait for
the SFT job to finish before launching this one.
"""
import hashlib
import os
import pathlib
import shutil
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import boto3
import datasets
import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split

from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.dataset_utils import CustomizationTechnique
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.resources import ModelPackageGroup
from sagemaker.train.common import TrainingType
from sagemaker.train.dpo_trainer import DPOTrainer

import config

BASE_MODEL_ID = "meta-textgeneration-llama-3-2-1b-instruct"

sess = Session()
bucket_name = sess.default_bucket()
default_prefix = sess.default_bucket_prefix
s3_client = boto3.client("s3")

stream = (load_dataset("HumanLLMs/Human-Like-DPO-Dataset", split="train", streaming=True)
          .take(3000).shuffle(buffer_size=1000))
dataset = datasets.Dataset.from_generator(lambda: stream, features=stream.features)
df = pd.DataFrame(dataset)
train, val = train_test_split(df, test_size=0.2, random_state=42)
train, test = train_test_split(train, test_size=0.1, random_state=42)
print(f"train {len(train)} | val {len(val)} | test {len(test)}")


def to_dpo(split, test=False):
    ds = Dataset.from_pandas(split, preserve_index=False)

    def f_train(sample):
        return {"prompt": sample["prompt"], "chosen": sample["chosen"], "rejected": sample["rejected"]}

    def f_test(sample):
        return {"query": sample["prompt"], "response": sample["chosen"]}

    return ds.map(f_test if test else f_train, remove_columns=list(ds.features))


parts = {"train": to_dpo(train), "val": to_dpo(val), "test": to_dpo(test, test=True)}

input_path = (f"{default_prefix}/datasets/serverless-model-customization-sft" if default_prefix
              else "datasets/serverless-model-customization-sft")
paths = {}
for name, ds in parts.items():
    fname = f"humanlike_dpo_{name}.jsonl"
    local = f"./data/{name}"
    os.makedirs(local, exist_ok=True)
    ds.to_json(f"{local}/{fname}", orient="records")
    key = f"{input_path}/{name}/{fname}"
    s3_client.upload_file(f"{local}/{fname}", bucket_name, key)
    paths[name] = f"s3://{bucket_name}/{key}"
    print("uploaded:", paths[name])
shutil.rmtree("./data")

train_ds = DataSet.create(name="humanlike-dpo-train", source=paths["train"],
                          customization_technique=CustomizationTechnique.DPO, wait=True,
                          role=config.ROLE_ARN)
val_ds = DataSet.create(name="humanlike-dpo-val", source=paths["val"],
                        customization_technique=CustomizationTechnique.DPO, wait=True,
                        role=config.ROLE_ARN)
DataSet.create(name="humanlike-dpo-test", source=paths["test"], wait=True,
               role=config.ROLE_ARN)
print("datasets registered")

candidate = f"{BASE_MODEL_ID}-dpo"
if len(candidate) > 63:
    digest = hashlib.sha1(BASE_MODEL_ID.encode()).hexdigest()[:6]
    keep = 63 - len("-dpo") - len(digest) - 1
    mpg = f"{BASE_MODEL_ID[:keep].rstrip('-')}-{digest}-dpo"
else:
    mpg = candidate
try:
    ModelPackageGroup.get(model_package_group_name=mpg)
    print(f"MPG exists: {mpg}")
except Exception:
    ModelPackageGroup.create(model_package_group_name=mpg,
                             model_package_group_description="SageMaker serverless DPO")
    print(f"created MPG: {mpg}")

output_path = f"s3://{bucket_name}/{default_prefix}/{BASE_MODEL_ID}-dpo" if default_prefix \
    else f"s3://{bucket_name}/{BASE_MODEL_ID}-dpo"

trainer = DPOTrainer(
    model=BASE_MODEL_ID,
    training_type=TrainingType.LORA,
    model_package_group=mpg,
    training_dataset=train_ds,
    validation_dataset=val_ds,
    s3_output_path=output_path,
    mlflow_experiment_name="humanlike-llama3-2-1b-dpo",
    base_job_name=f"dpo-{BASE_MODEL_ID.split('/')[-1].replace('.', '-')}"[:48].rstrip("-"),
    sagemaker_session=sess,
    accept_eula=True,
    role=config.ROLE_ARN,
)

training_job = trainer.train(wait=False)
print(f"LAUNCHED: {training_job.training_job_name}")

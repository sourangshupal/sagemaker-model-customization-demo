"""Step 2 — Lab 1 training: launch a serverless LoRA SFT job (Qwen3-4B).

Prints the job name; monitor with:
    aws sagemaker describe-training-job --training-job-name <name> \
        --query '[TrainingJobStatus,SecondaryStatus]'
"""
import hashlib
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from botocore.exceptions import ClientError
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.resources import ModelPackageGroup
from sagemaker.train.common import TrainingType
from sagemaker.train.sft_trainer import SFTTrainer

import config

sys.path.insert(0, config.LAB1_DIR)
from config import BASE_MODEL_ID  # workshop's lab config: huggingface-reasoning-qwen3-4b  # noqa: E402

DATASET_PREFIX = "contractnli-nda-review"

sess = Session()
bucket_name = sess.default_bucket()
default_prefix = sess.default_bucket_prefix

training_dataset = DataSet.get(name=f"{DATASET_PREFIX}-train")
val_dataset = DataSet.get(name=f"{DATASET_PREFIX}-val")

output_path = (f"s3://{bucket_name}/{default_prefix}/{BASE_MODEL_ID}-contractnli"
               if default_prefix else f"s3://{bucket_name}/{BASE_MODEL_ID}-contractnli")

MAX_MPG_NAME_LENGTH = 63
suffix = "-contractnli-sft-mpg"
candidate = f"{BASE_MODEL_ID}{suffix}"
if len(candidate) > MAX_MPG_NAME_LENGTH:
    digest = hashlib.sha1(BASE_MODEL_ID.encode()).hexdigest()[:6]
    keep = MAX_MPG_NAME_LENGTH - len(suffix) - len(digest) - 1
    model_package_group_name = f"{BASE_MODEL_ID[:keep].rstrip('-')}-{digest}{suffix}"
else:
    model_package_group_name = candidate
print(f"Model Package Group: {model_package_group_name}")

try:
    ModelPackageGroup.get(model_package_group_name=model_package_group_name)
    print(f"already exists: {model_package_group_name}")
except ClientError:
    ModelPackageGroup.create(
        model_package_group_name=model_package_group_name,
        model_package_group_description="ContractNLI NDA checklist review, serverless SFT",
    )
    print(f"created: {model_package_group_name}")

trainer = SFTTrainer(
    model=BASE_MODEL_ID,
    training_type=TrainingType.LORA,
    model_package_group=model_package_group_name,
    training_dataset=training_dataset,
    validation_dataset=val_dataset,
    s3_output_path=output_path,
    sagemaker_session=sess,
    role=config.ROLE_ARN,
    accept_eula=True,
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
print(f"LAUNCHED: {training_job.training_job_name}")

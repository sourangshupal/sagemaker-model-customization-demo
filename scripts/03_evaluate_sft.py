"""Step 3 — Lab 1 evaluation: custom scorer, managed pipeline, base vs tuned.

Registers the workshop's contractnli_scorer.py as a reward-function evaluator
and launches a CustomScorerEvaluator that scores BOTH the base JumpStart model
and your fine-tuned model package on the held-out test split.

Requires MLFLOW_TRACKING_SERVER_ARN (managed evaluation pipelines log to MLflow).
"""
import hashlib
import pathlib
import sys
import time

SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import boto3
from sagemaker.ai_registry.air_constants import REWARD_FUNCTION
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.evaluator import Evaluator
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.evaluate import CustomScorerEvaluator

import config

sys.path.insert(0, config.LAB1_DIR)
from config import BASE_MODEL_ID  # noqa: E402

assert config.MLFLOW_TRACKING_SERVER_ARN, "Set MLFLOW_TRACKING_SERVER_ARN (see README)"

DATASET_PREFIX = "contractnli-nda-review"

sess = Session()
REGION = sess.boto_region_name
BUCKET = sess.default_bucket()
sm = boto3.client("sagemaker", region_name=REGION)

MAX_MPG_NAME_LENGTH = 63
suffix = "-contractnli-sft-mpg"
candidate = f"{BASE_MODEL_ID}{suffix}"
if len(candidate) > MAX_MPG_NAME_LENGTH:
    digest = hashlib.sha1(BASE_MODEL_ID.encode()).hexdigest()[:6]
    keep = MAX_MPG_NAME_LENGTH - len(suffix) - len(digest) - 1
    model_package_group_name = f"{BASE_MODEL_ID[:keep].rstrip('-')}-{digest}{suffix}"
else:
    model_package_group_name = candidate

SCORER_NAME = "contractnli-scorer"
try:
    scorer = Evaluator.get(name=SCORER_NAME)
    print("reusing registered scorer")
except Exception:
    scorer = Evaluator.create(name=SCORER_NAME, type=REWARD_FUNCTION,
                              source=str(pathlib.Path(config.LAB1_DIR) / "contractnli_scorer.py"),
                              role=config.LAMBDA_ROLE_ARN,
                              sagemaker_session=sess, wait=True)
    print("registered scorer")

test_dataset = DataSet.get(name=f"{DATASET_PREFIX}-test")
resp = sm.list_model_packages(ModelPackageGroupName=model_package_group_name,
                              SortBy="CreationTime", SortOrder="Descending", MaxResults=1)
assert resp["ModelPackageSummaryList"], "no model packages found - run 02_train_sft.py first"
model_package_arn = resp["ModelPackageSummaryList"][0]["ModelPackageArn"]
print(f"fine-tuned model: {model_package_arn}")

scorer_eval = CustomScorerEvaluator(
    evaluator=scorer,
    mlflow_resource_arn=config.MLFLOW_TRACKING_SERVER_ARN,
    dataset=test_dataset,
    model=model_package_arn,
    model_package_group=model_package_group_name,
    s3_output_path=f"s3://{BUCKET}/contractnli-scorer-eval",
    evaluate_base_model=True,
    sagemaker_session=sess,
    role=config.ROLE_ARN,
)
scorer_eval.hyperparameters.max_new_tokens = 8192
scorer_eval.hyperparameters.max_model_len = 24000

scorer_execution = scorer_eval.evaluate()
print("launched:", scorer_execution.arn)

while True:
    status = sm.describe_pipeline_execution(
        PipelineExecutionArn=scorer_execution.arn)["PipelineExecutionStatus"]
    print(f"{time.strftime('%H:%M:%S')}  {status}")
    if status != "Executing":
        break
    time.sleep(60)
print("EVAL DONE:", status)

"""Shared configuration — all values come from environment variables.

Copy .env.example to .env and fill in your values, or export the variables
before running the scripts. See README.md for how to find each value.
"""
import os
import sys

# --- AWS ---
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
# SageMaker execution role used for training / evaluation / deployment.
# Note: IAM roles created by the SageMaker console live under a /service-role/
# path — the full ARN must include it, e.g.
# arn:aws:iam::<account>:role/service-role/AmazonSageMaker-ExecutionRole-...
ROLE_ARN = os.environ.get("ROLE_ARN", "")
if not ROLE_ARN:
    sys.exit("ROLE_ARN is not set — copy .env.example to .env and fill it in (see README).")
# Optional separate role for the custom-scorer Lambda. Defaults to ROLE_ARN
# (the README explains the trust-policy requirement for that).
LAMBDA_ROLE_ARN = os.environ.get("LAMBDA_ROLE_ARN", ROLE_ARN)

# --- Managed MLflow (required by managed evaluation pipelines) ---
MLFLOW_TRACKING_SERVER_ARN = os.environ.get("MLFLOW_TRACKING_SERVER_ARN", "")

# --- Local checkout of the AWS workshop repo ---
# The scripts reuse the workshop's dataset helpers and scorer code:
#   git clone https://github.com/aws-samples/generative-ai-on-amazon-sagemaker.git
WORKSHOP_DIR = os.environ.get(
    "WORKSHOP_DIR",
    os.path.expanduser("~/generative-ai-on-amazon-sagemaker"),
)
LAB1_DIR = os.path.join(
    WORKSHOP_DIR,
    "workshops/serverless-model-customization-with-sagemaker-ai/lab-1-supervised-fine-tuning",
)
LAB2_DIR = os.path.join(
    WORKSHOP_DIR,
    "workshops/serverless-model-customization-with-sagemaker-ai/lab-2-direct-preference-optimization-DPO",
)

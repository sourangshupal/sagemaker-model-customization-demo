"""Step 6 — Lab 2 evaluation: LLM-as-judge on the DPO-tuned model.

A judge model (Claude Sonnet 4.5 via Amazon Bedrock) scores the tuned model's
outputs against built-in metrics (Helpfulness, Relevance, Coherence) and three
custom rubric metrics (HumanLikeTone, ConversationalEngagement,
AvoidRoboticPatterns).

Requires MLFLOW_TRACKING_SERVER_ARN and a role with bedrock:CreateEvaluationJob
(see README "IAM requirements").
"""
import hashlib
import json
import pathlib
import sys
import time

SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import boto3
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.resources import ModelPackageGroup
from sagemaker.train.evaluate import LLMAsJudgeEvaluator

import config

assert config.MLFLOW_TRACKING_SERVER_ARN, "Set MLFLOW_TRACKING_SERVER_ARN (see README)"

BASE_MODEL_ID = "meta-textgeneration-llama-3-2-1b-instruct"
EVALUATOR_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"

sess = Session()
BUCKET = sess.default_bucket()
default_prefix = sess.default_bucket_prefix
sm_client = boto3.client("sagemaker", region_name=sess.boto_region_name)

candidate = f"{BASE_MODEL_ID}-dpo"
if len(candidate) > 63:
    digest = hashlib.sha1(BASE_MODEL_ID.encode()).hexdigest()[:6]
    keep = 63 - len("-dpo") - len(digest) - 1
    mpg = f"{BASE_MODEL_ID[:keep].rstrip('-')}-{digest}-dpo"
else:
    mpg = candidate

resp = sm_client.list_model_packages(ModelPackageGroupName=mpg, SortBy="CreationTime",
                                     SortOrder="Descending", MaxResults=1)
ft_arn = resp["ModelPackageSummaryList"][0]["ModelPackageArn"]
mpg_arn = ModelPackageGroup.get(mpg).model_package_group_arn
print(f"fine-tuned: {ft_arn}")

test_dataset = DataSet.get(name="humanlike-dpo-test")
output_path = (f"s3://{BUCKET}/{default_prefix}/dpo-llmaj-eval" if default_prefix
               else f"s3://{BUCKET}/dpo-llmaj-eval")

BUILTIN_METRICS = ["Helpfulness", "Relevance", "Coherence"]
custom_metrics_list = [
    {"customMetricDefinition": {
        "name": "HumanLikeTone",
        "instructions": ("Evaluate if the response sounds like a friendly human conversation rather than "
                         "a formal AI assistant. Human-like responses use casual language, show personality, "
                         "and feel warm and approachable. Robotic responses use formal phrases like "
                         "'I'm designed to', 'as an AI', 'I'm pleased to report', or overly corporate language. "
                         "Prompt: {{prompt}}\nResponse: {{prediction}}"),
        "ratingScale": [
            {"definition": "Excellent - Sounds completely natural and human", "value": {"floatValue": 3}},
            {"definition": "Good - Mostly human-like with minor formal elements", "value": {"floatValue": 2}},
            {"definition": "Mixed - Contains both human-like and robotic elements", "value": {"floatValue": 1}},
            {"definition": "Poor - Sounds robotic or like a corporate AI", "value": {"floatValue": 0}}]}},
    {"customMetricDefinition": {
        "name": "ConversationalEngagement",
        "instructions": ("Assess if the response engages the user in natural conversation. "
                         "Good responses ask follow-up questions, show genuine interest, and invite dialogue. "
                         "Poor responses are one-sided or end abruptly without engagement. "
                         "Prompt: {{prompt}}\nResponse: {{prediction}}"),
        "ratingScale": [
            {"definition": "Highly engaging - Asks questions, invites dialogue", "value": {"floatValue": 2}},
            {"definition": "Somewhat engaging - Some conversational elements", "value": {"floatValue": 1}},
            {"definition": "Not engaging - One-sided, no conversational flow", "value": {"floatValue": 0}}]}},
    {"customMetricDefinition": {
        "name": "AvoidRoboticPatterns",
        "instructions": ("Check if the response avoids robotic AI patterns. "
                         "PENALIZE responses containing: 'As an AI/language model', "
                         "'I'm designed to', 'I'm pleased to report', "
                         "'I don't have personal experiences/emotions', or overly formal corporate speak. "
                         "Prompt: {{prompt}}\nResponse: {{prediction}}"),
        "ratingScale": [
            {"definition": "Good - No robotic patterns, sounds naturally human", "value": {"floatValue": 1}},
            {"definition": "Bad - Contains robotic AI patterns or formal self-references", "value": {"floatValue": 0}}]}},
]

evaluator = LLMAsJudgeEvaluator(
    model=ft_arn,
    model_package_group=mpg_arn,
    evaluator_model=EVALUATOR_MODEL,
    dataset=test_dataset,
    builtin_metrics=BUILTIN_METRICS,
    custom_metrics=json.dumps(custom_metrics_list),
    s3_output_path=output_path,
    evaluate_base_model=False,
    mlflow_resource_arn=config.MLFLOW_TRACKING_SERVER_ARN,
    sagemaker_session=sess,
    role=config.ROLE_ARN,
)
execution = evaluator.evaluate()
print("launched:", execution.arn)

while True:
    status = sm_client.describe_pipeline_execution(
        PipelineExecutionArn=execution.arn)["PipelineExecutionStatus"]
    print(f"{time.strftime('%H:%M:%S')}  {status}")
    if status != "Executing":
        break
    time.sleep(60)
print("DPO EVAL DONE:", status)

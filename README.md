# SageMaker AI Model Customization — Demo Scripts

Standalone, reproducible driver scripts for **serverless model customization on
Amazon SageMaker AI**: supervised fine-tuning (SFT) with LoRA, Direct Preference
Optimization (DPO), managed evaluation (custom scorer + LLM-as-judge), and
deployment to a vLLM endpoint with an inference smoke test and full teardown.

These scripts are the runnable companion to the AWS workshop
**"Serverless Model Customization with Amazon SageMaker AI"**
([workshop](https://catalog.workshops.aws/workshops/548b5be9-2da8-4c93-82f7-b0b474108ab3/en-US) |
[source repo](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker)) —
they run the same steps **from your local machine**, outside SageMaker Studio and
without the workshop's CloudFormation stack. They are adapted from the workshop
notebooks and restructured for a one-command demo flow.

## What it demonstrates

| Step | Script | What happens | Measured duration* |
|------|--------|--------------|--------------------|
| 1 | `01_data_prep_sft.py` | ContractNLI (607 real NDAs) → prompt/completion JSONL → S3 → SageMaker AI Registry | ~5 min |
| 2 | `02_train_sft.py` | Serverless **LoRA SFT** of Qwen3-4B (`SFTTrainer`, pay-per-token, no instance selection) | ~29 min |
| 3 | `03_evaluate_sft.py` | Custom scorer registered as a reward function; managed pipeline scores **base vs fine-tuned** on 123 held-out contracts | 15–20 min |
| 4 | `04_deploy_sft.py` | Merged checkpoint → DJL/vLLM container on `ml.g5.xlarge` → OpenAI-schema inference test → teardown (`deploy`/`test`/`teardown`/`all`) | ~16 min |
| 5 | `05_train_dpo.py` | Human-Like DPO dataset (~3k preference pairs) → serverless **DPO** of Llama 3.2 1B (`DPOTrainer`) | ~20 min |
| 6 | `06_evaluate_dpo.py` | **LLM-as-judge** (Claude Sonnet 4.5 via Bedrock) with built-in + custom rubric metrics | 30–45 min |

\*Wall-clock measured in a real run (us-east-1, September 2026). Training/eval
times vary with service load.

## Prerequisites

1. **AWS account + region**: serverless customization is available in
   `us-east-1`, `us-west-2`, `eu-west-1`, `ap-northeast-1`. Everything below
   assumes `us-east-1`.
2. **SageMaker execution role.** The console-created role works, but note its
   ARN contains a `/service-role/` path — use the full ARN. The role needs:
   - trust: `sagemaker.amazonaws.com` **and** `lambda.amazonaws.com` (the custom
     scorer runs as a Lambda; without the Lambda trust,
     `Evaluator.create` fails with *"role defined for the function cannot be
     assumed by Lambda"*)
   - `AWSLambdaBasicExecutionRole` + S3 read access attached (for that scorer Lambda)
   - for the LLM-as-judge step, add `bedrock:CreateEvaluationJob`,
     `bedrock:GetEvaluationJob`, `bedrock:ListEvaluationJobs`, `bedrock:InvokeModel`
3. **Quotas — check these before demo day** (Service Quotas → SageMaker):
   - `L-92BDBBEF` concurrent model customization serverless jobs — **defaults to 1**
     (train SFT and DPO sequentially)
   - `L-619D690E` concurrent model evaluation serverless jobs — **defaults to 0**;
     you must request an increase or all evaluation jobs fail with
     `ResourceLimitExceeded`
4. **Managed MLflow tracking server** — managed evaluation pipelines require one:

   ```bash
   aws sagemaker create-mlflow-tracking-server \
     --tracking-server-name my-mlflow --role-arn $ROLE_ARN \
     --tracking-server-size Small \
     --artifact-store-uri s3://$DEFAULT_BUCKET/mlflow-artifacts
   ```
   Note: the status you want is `Created` (that **is** the healthy state).
5. **Budget alarm** (strongly recommended): AWS Budgets, e.g. $50/month with an
   80% email alert.
6. **Local environment**: Python 3.12, then:

   ```bash
   pip install -r requirements.txt
   git clone https://github.com/aws-samples/generative-ai-on-amazon-sagemaker.git
   ```

## Setup

```bash
git clone https://github.com/sourangshupal/sagemaker-model-customization-demo.git
cd sagemaker-model-customization-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Path to your aws-samples/generative-ai-on-amazon-sagemaker checkout
git clone https://github.com/aws-samples/generative-ai-on-amazon-sagemaker.git ~/generative-ai-on-amazon-sagemaker

cp .env.example .env   # fill in ROLE_ARN and MLFLOW_TRACKING_SERVER_ARN
export $(grep -v '^#' .env | xargs)   # or use direnv
```

The scripts reuse the workshop repo's `contractnli.py` / `contractnli_scorer.py`
helpers, so `WORKSHOP_DIR` must point at a local checkout.

## Run the demo

```bash
python scripts/01_data_prep_sft.py     # ~5 min
python scripts/02_train_sft.py         # launch, then monitor:
aws sagemaker describe-training-job --training-job-name <name> \
  --query '[TrainingJobStatus,SecondaryStatus]'
python scripts/03_evaluate_sft.py      # needs eval quota >= 1 + MLflow server
python scripts/04_deploy_sft.py all    # deploy -> smoke test -> TEARDOWN
python scripts/05_train_dpo.py         # after the SFT job finishes (quota = 1)
python scripts/06_evaluate_dpo.py
```

**Always finish with teardown.** An idle `ml.g5.xlarge` endpoint costs ~$1/hour.
The MLflow tracking server also bills hourly — delete it when done:

```bash
aws sagemaker delete-mlflow-tracking-server --tracking-server-name my-mlflow
```

## Expected results

Reference numbers from the workshop (verify with step 3 on your own run):

- **SFT**: checklist-item accuracy **64.5% → 87.4%**, evidence F1 **48.8 → 75.4**
  on 123 held-out contracts (base vs LoRA-tuned Qwen3-4B)
- **Training cost**: low single-digit dollars per job (per-token billing)
- **DPO**: judge-scored improvements on HumanLikeTone / ConversationalEngagement /
  AvoidRoboticPatterns plus built-in Helpfulness/Relevance/Coherence

## Troubleshooting (field notes from real runs)

- **`Could not assume role .../AmazonSageMaker-ExecutionRole-...`** — the role ARN
  is missing its `/service-role/` path component. Get the real ARN with
  `aws iam get-role --role-name <name>`.
- **`cannot be used for 'training' workloads`** (SDK `RoleValidationError`) — pass
  the role explicitly (`role=`); the SDK can't resolve one from an IAM user.
- **`role ... cannot be assumed by Lambda`** — add `lambda.amazonaws.com` to the
  role trust policy (see Prerequisites).
- **`MLflow Resource ARN must match format`** — create a tracking server (step 4 of
  Prerequisites) and set `MLFLOW_TRACKING_SERVER_ARN`.
- **Eval pipeline fails instantly with `ResourceLimitExceeded`** — quota
  `L-619D690E` is 0; request an increase and wait for the case to close.
- **`ModuleNotFoundError: No module named 'contractnli'`** — `WORKSHOP_DIR` doesn't
  point at the aws-samples repo checkout.

## Cost summary (measured/estimated, us-east-1)

| Item | Approx. cost |
|------|--------------|
| SFT job (Qwen3-4B LoRA, 29 min) | ~$2 |
| DPO job (Llama 3.2 1B, 20 min) | ~$1–2 |
| Endpoint (ml.g5.xlarge, 16 min incl. teardown) | ~$0.27 |
| LLM-as-judge eval (Claude Sonnet 4.5 judge calls) | ~$1–3 |
| MLflow tracking server (while it exists) | ~$0.25/hr — delete when done |

Full rehearsal came to **well under $10** end-to-end.

## Disclaimer

Unofficial companion code, provided as-is. Not affiliated with or endorsed by AWS.
Dataset: ContractNLI (CC-BY-4.0). Model names belong to their respective owners.
Review AWS service terms and pricing before running; you are responsible for all
charges in your account.

## License

MIT (see [LICENSE](LICENSE)). The workshop notebooks and helper modules this repo
references belong to the
[aws-samples/generative-ai-on-amazon-sagemaker](https://github.com/aws-samples/generative-ai-on-amazon-sagemaker)
repository and its own license.

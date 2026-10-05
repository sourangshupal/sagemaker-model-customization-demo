"""Step 4d — deploy the DPO-tuned Llama 3.2 1B model for the tone A/B comparison.

Usage:
    python 04d_deploy_dpo.py deploy    # create model / endpoint config / endpoint / IC
    python 04d_deploy_dpo.py test      # invoke with one casual prompt, print the answer
    python 04d_deploy_dpo.py teardown  # delete IC, endpoint, config, model
    python 04d_deploy_dpo.py all       # deploy -> test -> teardown (demo mode)

The endpoint uses an ml.g5.xlarge (~$1/hour). ALWAYS finish with teardown.
NOTE: ml.g5.xlarge endpoint quota defaults to 2 — delete other endpoints first if
both slots are taken.
"""
import hashlib
import json
import pathlib
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "all"

SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import boto3
from botocore.config import Config
from sagemaker.core import s3
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.resources import Endpoint, EndpointConfig, InferenceComponent, Model, ModelPackage
from sagemaker.core.shapes import (ContainerDefinition, InferenceComponentComputeResourceRequirements,
                                   InferenceComponentRuntimeConfig, InferenceComponentSpecification,
                                   ModelDataSource, ProductionVariant, S3ModelDataSource)

import config

sess = Session()
region = sess.boto_region_name
sm_client = boto3.client("sagemaker", region_name=region)

BASE_MODEL_ID = "meta-textgeneration-llama-3-2-1b-instruct"
MODEL_PACKAGE_GROUP = f"{BASE_MODEL_ID}-dpo"

MAX_NAME = 63


def rname(base, sfx):
    cand = f"{base}{sfx}"
    if len(cand) <= MAX_NAME:
        return cand
    digest = hashlib.sha1(base.encode()).hexdigest()[:6]
    keep = MAX_NAME - len(sfx) - len(digest) - 1
    return f"{base[:keep].rstrip('-')}-{digest}{sfx}"


stem = f"{BASE_MODEL_ID}-dpo"
model_name = rname(stem, "-m")
endpoint_config_name = rname(stem, "-cfg")
endpoint_name = rname(stem, "-ep")
ic_name = rname(stem, "-ic")

if MODE in ("all", "deploy"):
    resp = sm_client.list_model_packages(ModelPackageGroupName=MODEL_PACKAGE_GROUP,
                                         SortBy="CreationTime", SortOrder="Descending", MaxResults=1)
    assert resp["ModelPackageSummaryList"], "no model packages found - run 05_train_dpo.py first"
    model_package = ModelPackage.get(resp["ModelPackageSummaryList"][0]["ModelPackageArn"])
    merged_model_s3_uri = s3.s3_path_join(
        model_package.inference_specification.containers[0]
        .model_data_source.s3_data_source.s3_uri, "checkpoints", "hf_merged") + "/"
    print(f"merged model: {merged_model_s3_uri}")

    CONTAINER_VERSION = "0.36.0-lmi18.0.0-cu128"
    inference_image = f"763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:{CONTAINER_VERSION}"
    instance_type = "ml.g5.xlarge"
    health_check_timeout = 700
    env = {
        "SERVING_FAIL_FAST": "true",
        "OPTION_ASYNC_MODE": "true",
        "OPTION_ROLLING_BATCH": "disable",
        "OPTION_MAX_MODEL_LEN": "16384",
        "OPTION_TENSOR_PARALLEL_DEGREE": "max",
        "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service",
        "OPTION_TRUST_REMOTE_CODE": "true",
    }

    try:
        Model.get(model_name)
        print(f"model exists: {model_name}")
    except Exception:
        Model.create(model_name=model_name,
                     primary_container=ContainerDefinition(
                         image=inference_image,
                         model_data_source=ModelDataSource(
                             s3_data_source=S3ModelDataSource(
                                 s3_uri=merged_model_s3_uri, s3_data_type="S3Prefix",
                                 compression_type="None")),
                         environment=env),
                     execution_role_arn=config.ROLE_ARN)
        print(f"created model: {model_name}")

    try:
        EndpointConfig.get(endpoint_config_name)
        print(f"endpoint config exists: {endpoint_config_name}")
    except Exception:
        EndpointConfig.create(
            endpoint_config_name=endpoint_config_name,
            execution_role_arn=config.ROLE_ARN,
            production_variants=[ProductionVariant(
                variant_name="AllTraffic", instance_type=instance_type,
                initial_instance_count=1,
                model_data_download_timeout_in_seconds=health_check_timeout,
                container_startup_health_check_timeout_in_seconds=health_check_timeout)],
        )
        print(f"created endpoint config: {endpoint_config_name}")

    try:
        Endpoint.get(endpoint_name)
        print(f"endpoint exists: {endpoint_name}")
    except Exception:
        Endpoint.create(endpoint_name=endpoint_name, endpoint_config_name=endpoint_config_name)
        print(f"creating endpoint: {endpoint_name}")

    Endpoint.get(endpoint_name).wait_for_status("InService")
    print("endpoint InService")

    try:
        InferenceComponent.get(ic_name)
        print(f"inference component exists: {ic_name}")
    except Exception:
        InferenceComponent.create(
            inference_component_name=ic_name, endpoint_name=endpoint_name,
            variant_name="AllTraffic",
            specification=InferenceComponentSpecification(
                model_name=model_name,
                compute_resource_requirements=InferenceComponentComputeResourceRequirements(
                    min_memory_required_in_mb=1024, number_of_accelerator_devices_required=1)),
            runtime_config=InferenceComponentRuntimeConfig(copy_count=1), region=region)
        print(f"creating inference component: {ic_name}")
    InferenceComponent.get(ic_name).wait_for_status("InService")
    print(f"ready: sm:{endpoint_name}/{ic_name}@{region}")

if MODE in ("all", "test"):
    smr = boto3.client("sagemaker-runtime", region_name=region,
                       config=Config(read_timeout=300, retries={"total_max_attempts": 3}))

    prompt = "Hey, I just moved to a new city and don't really know anyone yet. How do you make friends as an adult?"
    body = {
        "model_name": ic_name,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": 400, "temperature": 0.7, "stream": False,
    }
    t0 = time.time()
    response = smr.invoke_endpoint(EndpointName=endpoint_name, InferenceComponentName=ic_name,
                                   ContentType="application/json", Body=json.dumps(body))
    text = json.loads(response["Body"].read())["choices"][0]["message"]["content"]
    print(f"prompt: {prompt}")
    print(f"answer ({time.time() - t0:.1f}s):")
    print(text)

if MODE in ("all", "teardown"):
    time.sleep(45)
    for label, fn in [
        ("inference component", lambda: InferenceComponent.get(ic_name).delete()),
        ("endpoint", lambda: Endpoint.get(endpoint_name).delete()),
        ("endpoint config", lambda: EndpointConfig.get(endpoint_config_name).delete()),
        ("model", lambda: Model.get(model_name).delete()),
    ]:
        try:
            fn()
            print(f"deleted {label}")
        except Exception as e:
            print(f"skip {label}: {e}")
    print("TEARDOWN DONE")

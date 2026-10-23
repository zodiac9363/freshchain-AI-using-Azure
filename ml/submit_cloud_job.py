"""
FreshChain AI — Phase 5: Submit Serverless Cloud Training Job to Azure ML
Executes model training and registration entirely on Microsoft Azure Cloud hardware.
"""

import os
import sys
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

from azure.ai.ml import MLClient, command
from azure.identity import DeviceCodeCredential, DefaultAzureCredential


def main():
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID", "d445f91e-27e9-4d8a-9d6f-170f2b88cd4f")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP", "freshchain-rg")
    workspace_name = os.getenv("AZURE_ML_WORKSPACE_NAME", "freshchain-mlw")

    print("=" * 70)
    print("🚀 FreshChain AI — Submitting Serverless Cloud Training Job to Azure")
    print(f"Workspace: {workspace_name} | Resource Group: {resource_group}")
    print("=" * 70)

    tenant_id = os.getenv("AZURE_TENANT_ID", "b3db39c4-ddc0-4755-8e89-ebd4c2cd2ab2")

    def prompt_callback(uri, code, expires_on):
        print("\n" + "=" * 65)
        print("  🔑 AZURE LOGIN REQUIRED TO SUBMIT CLOUD JOB")
        print(f"  1. Go to: {uri}")
        print(f"  2. Enter code: {code}")
        print("=" * 65 + "\n", flush=True)

    # Authenticate specifically with the Azure for Students tenant
    cred = DeviceCodeCredential(tenant_id=tenant_id, prompt_callback=prompt_callback)
    ml_client = MLClient(cred, subscription_id, resource_group, workspace_name, tenant_id=tenant_id)

    print("\n[1/3] Packaging project code and configuring cloud job definition...")

    job = command(
        code=PROJECT_ROOT,
        command="python ml/cloud_train_job.py",
        environment="azureml://registries/azureml/environments/sklearn-1.5/labels/latest",
        compute=None,  # Serverless compute: On-demand serverless VM in Azure, auto-terminates after training ($0.00 idle cost)
        display_name="freshchain-cloud-training-run",
        experiment_name="freshchain-produce-quality-classification",
        description="FreshChain AI 3-model benchmark and champion training executed on Azure Cloud Compute."
    )

    print("\n[2/3] Submitting training job to Microsoft Azure...")
    submitted_job = ml_client.jobs.create_or_update(job)

    print("\n[3/3] 🎉 Cloud Job Successfully Submitted to Azure!")
    print("=" * 70)
    print(f"Job Name:   {submitted_job.name}")
    print(f"Status:     {submitted_job.status}")
    print(f"Studio URL: {submitted_job.studio_url}")
    print("=" * 70)
    print("\n👉 Click the Studio URL above to watch the cloud VM execute the training live in Azure ML Studio!")


if __name__ == "__main__":
    main()

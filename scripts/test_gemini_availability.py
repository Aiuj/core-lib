import os
from google import genai
from dotenv import load_dotenv

# Load environment variables (if you use a .env file for credentials)
load_dotenv()

# Project ID and credentials
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
service_account_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# 1. Define the regions and models we want to hunt for
regions_to_check = ["us-central1", "global", "europe-west4", "europe-west9"]
target_models = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
]


def check_models(client, regions, targets):
    """Check model availability across regions for a given client factory."""
    print(f"{'REGION':<15} | {'MODEL ID':<30} | {'STATUS'}")
    print("-" * 60)
    for region in regions:
        try:
            c = client(region)
            available_models = list(c.models.list())
            available_ids = [m.name.split("/")[-1] for m in available_models]
            for target in targets:
                found = any(target in model_id for model_id in available_ids)
                status = "✅ AVAILABLE" if found else "❌ NOT FOUND"
                print(f"{region:<15} | {target:<30} | {status}")
        except Exception as e:
            print(f"{region:<15} | ERROR: {str(e)}")
    print("-" * 60)


# ---------------------------------------------------------------------------
# Configuration 1: Vertex AI — ADC / default credentials (no service account)
# Uses whatever credentials are active in the environment (gcloud auth, etc.)
# ---------------------------------------------------------------------------
print("\n=== Vertex AI (default ADC credentials) ===")
print(f"Project: {project_id or '(not set — GOOGLE_CLOUD_PROJECT missing)'}")

check_models(
    lambda region: genai.Client(vertexai=True, project=project_id, location=region),
    regions_to_check,
    target_models,
)

# ---------------------------------------------------------------------------
# Configuration 2: Vertex AI — explicit service account file
# Uses GOOGLE_APPLICATION_CREDENTIALS if set and the file exists.
# ---------------------------------------------------------------------------
print("\n=== Vertex AI (GOOGLE_APPLICATION_CREDENTIALS service account) ===")

if not service_account_file:
    print("SKIPPED — GOOGLE_APPLICATION_CREDENTIALS is not set.")
elif not os.path.isfile(service_account_file):
    print(f"SKIPPED — file not found: {service_account_file}")
else:
    print(f"Credentials file: {service_account_file}")
    print(f"Project: {project_id or '(not set — GOOGLE_CLOUD_PROJECT missing)'}")

    try:
        from google.oauth2 import service_account as sa_module

        _VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
        sa_credentials = sa_module.Credentials.from_service_account_file(
            service_account_file, scopes=_VERTEX_SCOPES
        )

        check_models(
            lambda region: genai.Client(
                vertexai=True,
                project=project_id,
                location=region,
                credentials=sa_credentials,
            ),
            regions_to_check,
            target_models,
        )
    except ImportError:
        print("ERROR — google-auth not installed. Run: pip install google-auth")

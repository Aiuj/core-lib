import os
from google import genai
from dotenv import load_dotenv

# Load environment variables (if you use a .env file for credentials)
load_dotenv()

# The specific project ID - optional if already set in environment
project_id = os.getenv("GOOGLE_CLOUD_PROJECT") 

# 1. Define the regions and models we want to hunt for
regions_to_check = ["us-central1", "europe-west1", "europe-west4", "europe-west9"]
target_models = [
    "gemini-2.5-flash", 
    "gemini-2.5-flash-lite", 
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
]

print(f"{'REGION':<15} | {'MODEL ID':<30} | {'STATUS'}")
print("-" * 60)

for region in regions_to_check:
    try:
        # 2. Initialize client specifically for this region
        # This is a metadata-only client, so no generation costs apply
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=region
        )

        # 3. List all models available in this region
        # The .list() method is free
        available_models = list(client.models.list())
        
        # Extract just the model IDs (names) for easier checking
        # Model names usually come as "publishers/google/models/gemini-..."
        # We split by '/' to get just the ID
        available_ids = [m.name.split('/')[-1] for m in available_models]

        # 4. Check if our targets exist in this list
        for target in target_models:
            # We look for partial matches to catch versions like "...-001" or "...-preview"
            found = any(target in model_id for model_id in available_ids)
            
            status = "✅ AVAILABLE" if found else "❌ NOT FOUND"
            
            # Only print if found, or if you want to see what's missing
            print(f"{region:<15} | {target:<30} | {status}")

    except Exception as e:
        print(f"{region:<15} | ERROR: {str(e)}")

print("-" * 60)

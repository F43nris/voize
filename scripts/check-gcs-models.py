#!/usr/bin/env python3
"""
Check if ML models are properly uploaded to Google Cloud Storage
"""

import os
import sys
from pathlib import Path

try:
    from google.cloud import storage
except ImportError:
    print("❌ Google Cloud Storage client not available!")
    print("Install with: pip install google-cloud-storage")
    sys.exit(1)


def check_gcs_models(bucket_name: str = "voize-ml-models", project_id: str = None):
    """Check if models exist in GCS bucket"""
    
    print("🔍 CHECKING GCS MODELS")
    print("=" * 30)
    print(f"Bucket: gs://{bucket_name}")
    if project_id:
        print(f"Project: {project_id}")
    print()
    
    try:
        # Initialize GCS client
        if project_id:
            client = storage.Client(project=project_id)
        else:
            client = storage.Client()
            
        bucket = client.bucket(bucket_name)
        
        # Check if bucket exists
        if not bucket.exists():
            print(f"❌ Bucket '{bucket_name}' does not exist!")
            print(f"Create it with: gsutil mb gs://{bucket_name}")
            return False
        
        print(f"✅ Bucket '{bucket_name}' exists")
        
        # Check for model files
        model_files = [
            "models/optimized_multinomial_nb.pkl",
            "models/feature_engine.pkl"
        ]
        
        print("\n📁 Checking model files:")
        all_files_exist = True
        
        for file_path in model_files:
            blob = bucket.blob(file_path)
            if blob.exists():
                # Get file info
                blob.reload()  # Refresh metadata
                size_mb = blob.size / (1024 * 1024)
                updated = blob.updated.strftime("%Y-%m-%d %H:%M:%S UTC")
                print(f"✅ {file_path}")
                print(f"   Size: {size_mb:.2f} MB")
                print(f"   Updated: {updated}")
            else:
                print(f"❌ {file_path} - NOT FOUND")
                all_files_exist = False
        
        if all_files_exist:
            print(f"\n🎉 All models found in gs://{bucket_name}!")
            print("Your Cloud Run service should be able to download them.")
        else:
            print(f"\n⚠️  Missing models in gs://{bucket_name}")
            print("Upload them with:")
            print(f"  python scripts/upload-models-to-gcs.py --bucket {bucket_name}")
        
        return all_files_exist
        
    except Exception as e:
        print(f"❌ Error checking GCS bucket: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check ML models in GCS")
    parser.add_argument("--bucket", default="voize-ml-models", help="GCS bucket name")
    parser.add_argument("--project", help="GCP project ID")
    
    args = parser.parse_args()
    
    success = check_gcs_models(args.bucket, args.project)
    sys.exit(0 if success else 1) 
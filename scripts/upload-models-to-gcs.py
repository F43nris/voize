#!/usr/bin/env python3
"""
Upload trained ML models to Google Cloud Storage for production deployment.

This script uploads the locally trained models to a GCS bucket so they can be
downloaded by production instances at runtime.

Usage:
    python scripts/upload-models-to-gcs.py --bucket voize-ml-models --project your-project-id
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from google.cloud import storage
except ImportError:
    print("❌ Google Cloud Storage client not available!")
    print("Install with: pip install google-cloud-storage")
    sys.exit(1)


def upload_models_to_gcs(bucket_name: str, project_id: str = None):
    """Upload trained models to Google Cloud Storage"""
    
    # Check if model files exist locally
    model_files = [
        ("data/optimized_multinomial_nb.pkl", "models/optimized_multinomial_nb.pkl"),
        ("data/feature_engine.pkl", "models/feature_engine.pkl")
    ]
    
    print("🔍 Checking for local model files...")
    for local_path, _ in model_files:
        if not Path(local_path).exists():
            print(f"❌ Model file not found: {local_path}")
            print("Please train the models first using the training pipeline.")
            return False
    
    print("✅ All model files found locally")
    
    try:
        # Initialize GCS client
        if project_id:
            client = storage.Client(project=project_id)
        else:
            client = storage.Client()  # Use default project from environment
        
        bucket = client.bucket(bucket_name)
        
        # Check if bucket exists
        if not bucket.exists():
            print(f"❌ Bucket '{bucket_name}' does not exist!")
            print(f"Create it with: gsutil mb gs://{bucket_name}")
            return False
        
        print(f"☁️  Uploading models to gs://{bucket_name}/")
        
        # Upload each model file
        for local_path, cloud_path in model_files:
            print(f"📤 Uploading {local_path} → {cloud_path}")
            
            blob = bucket.blob(cloud_path)
            blob.upload_from_filename(local_path)
            
            # Verify upload
            if blob.exists():
                size_mb = blob.size / (1024 * 1024)
                print(f"✅ Uploaded {cloud_path} ({size_mb:.2f} MB)")
            else:
                print(f"❌ Failed to upload {cloud_path}")
                return False
        
        print("\n🎉 All models uploaded successfully!")
        print(f"Production instances will download from: gs://{bucket_name}/models/")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to upload models: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload trained ML models to Google Cloud Storage"
    )
    parser.add_argument(
        "--bucket", 
        required=True,
        help="GCS bucket name (e.g., voize-ml-models)"
    )
    parser.add_argument(
        "--project",
        help="GCP project ID (optional, uses default from environment)"
    )
    
    args = parser.parse_args()
    
    print("🚀 ML Model Upload to Google Cloud Storage")
    print("=" * 50)
    print(f"Bucket: gs://{args.bucket}")
    if args.project:
        print(f"Project: {args.project}")
    print()
    
    success = upload_models_to_gcs(args.bucket, args.project)
    
    if success:
        print("\n✅ Upload completed successfully!")
        print("\nNext steps:")
        print("1. Deploy your application with the updated code")
        print("2. Set GCS_MODEL_BUCKET environment variable in production")
        print("3. Ensure production has GCS access permissions")
        sys.exit(0)
    else:
        print("\n❌ Upload failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 
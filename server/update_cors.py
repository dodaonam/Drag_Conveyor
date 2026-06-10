"""Update R2 CORS policy. Usage: python update_cors.py [tunnel_url]"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT_URL"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name="auto",
    config=Config(signature_version="s3v4"),
)

ALWAYS_ALLOWED = [
    "http://localhost:8000",
    "http://localhost:3000",
]

tunnel_url = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else None
origins = ALWAYS_ALLOWED + ([tunnel_url] if tunnel_url else [])

s3.put_bucket_cors(
    Bucket=os.environ["R2_BUCKET_NAME"],
    CORSConfiguration={
        "CORSRules": [
            {
                "AllowedOrigins": origins,
                "AllowedMethods": ["PUT", "GET", "HEAD"],
                "AllowedHeaders": ["*"],
                "ExposeHeaders": ["ETag"],
                "MaxAgeSeconds": 3600,
            }
        ]
    },
)

print("CORS updated:")
for o in origins:
    print(f"  {o}")

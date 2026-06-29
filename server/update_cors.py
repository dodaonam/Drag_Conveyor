"""Update R2 CORS policy."""
from __future__ import annotations

import os
import sys

ALWAYS_ALLOWED = [
    "http://localhost:8000",
    "http://localhost:3000",
]


def update_cors(tunnel_url: str | None = None) -> None:
    import boto3
    from botocore.config import Config
    import settings  # lazy — load_dotenv phải chạy trước khi import

    endpoint = settings.R2_ENDPOINT_URL
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

    url = tunnel_url.rstrip("/") if tunnel_url else None
    origins = ALWAYS_ALLOWED + ([url] if url else [])

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


if __name__ == "__main__":
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
    url = sys.argv[1] if len(sys.argv) > 1 else None
    update_cors(url)

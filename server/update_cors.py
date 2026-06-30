"""Update R2 CORS policy."""
from __future__ import annotations

ALWAYS_ALLOWED = [
    "http://localhost:8000",
    "http://localhost:3000",
]


def update_cors(tunnel_url: str | None = None) -> None:
    import boto3
    from botocore.config import Config
    import settings  # lazy — load_dotenv phải chạy trước khi import

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.R2_ENDPOINT_URL,
        aws_access_key_id=settings.R2_ACCESS_KEY_ID,
        aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

    url = tunnel_url.rstrip("/") if tunnel_url else None
    origins = ALWAYS_ALLOWED + ([url] if url else [])

    s3.put_bucket_cors(
        Bucket=settings.R2_BUCKET_NAME,
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
    import sys
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
    url = sys.argv[1] if len(sys.argv) > 1 else None
    update_cors(url)

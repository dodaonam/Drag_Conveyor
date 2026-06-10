from __future__ import annotations

import threading
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

import settings

_local = threading.local()


def _client():
    if not hasattr(_local, "s3"):
        _local.s3 = boto3.client(
            "s3",
            endpoint_url=settings.R2_ENDPOINT_URL,
            aws_access_key_id=settings.R2_ACCESS_KEY_ID,
            aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )
    return _local.s3


def presigned_put_url(object_key: str, content_type: str, expires: int) -> str:
    return _client().generate_presigned_url(
        "put_object",
        Params={
            "Bucket": settings.R2_BUCKET_NAME,
            "Key": object_key,
            "ContentType": content_type,
        },
        ExpiresIn=expires,
    )


def presigned_get_url(object_key: str, expires: int) -> str:
    return _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.R2_BUCKET_NAME, "Key": object_key},
        ExpiresIn=expires,
    )


def object_exists(object_key: str) -> bool:
    try:
        _client().head_object(Bucket=settings.R2_BUCKET_NAME, Key=object_key)
        return True
    except ClientError:
        return False


def delete_object(object_key: str) -> None:
    _client().delete_object(Bucket=settings.R2_BUCKET_NAME, Key=object_key)


def download_file(object_key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(settings.R2_BUCKET_NAME, object_key, str(dest))


def upload_file(local_path: Path, object_key: str, content_type: str = "application/octet-stream") -> None:
    _client().upload_file(
        str(local_path),
        settings.R2_BUCKET_NAME,
        object_key,
        ExtraArgs={"ContentType": content_type},
    )

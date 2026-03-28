"""
Stroke Detection System — AWS S3 Storage Service

Handles upload/download of CT scans and result images to/from S3.
"""

import io
import os
from typing import Optional
from loguru import logger
import boto3
from botocore.exceptions import ClientError

from app.core.config import get_settings

settings = get_settings()


class S3Service:
    """Thin wrapper around boto3 S3 operations."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = boto3.client(
                "s3",
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id or None,
                aws_secret_access_key=settings.aws_secret_access_key or None,
            )
        return self._client

    def upload_bytes(
        self,
        file_bytes: bytes,
        key: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload raw bytes to S3 and return the object URL."""
        try:
            self.client.put_object(
                Bucket=settings.s3_bucket_name,
                Key=key,
                Body=file_bytes,
                ContentType=content_type,
            )
            url = f"https://{settings.s3_bucket_name}.s3.{settings.aws_region}.amazonaws.com/{key}"
            logger.info(f"Uploaded to S3: {key}")
            return url
        except ClientError as e:
            logger.error(f"S3 upload failed for {key}: {e}")
            raise

    def download_bytes(self, key: str) -> bytes:
        """Download an object from S3 and return its bytes."""
        try:
            response = self.client.get_object(
                Bucket=settings.s3_bucket_name, Key=key
            )
            data = response["Body"].read()
            logger.info(f"Downloaded from S3: {key}")
            return data
        except ClientError as e:
            logger.error(f"S3 download failed for {key}: {e}")
            raise

    def generate_presigned_url(
        self, key: str, expiration: int = 3600
    ) -> Optional[str]:
        """Generate a pre-signed URL for temporary access."""
        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": settings.s3_bucket_name, "Key": key},
                ExpiresIn=expiration,
            )
            return url
        except ClientError as e:
            logger.error(f"Pre-signed URL generation failed: {e}")
            return None

    def delete_object(self, key: str) -> None:
        """Delete an object from S3."""
        try:
            self.client.delete_object(
                Bucket=settings.s3_bucket_name, Key=key
            )
            logger.info(f"Deleted from S3: {key}")
        except ClientError as e:
            logger.error(f"S3 delete failed for {key}: {e}")
            raise


# Module-level singleton
s3_service = S3Service()

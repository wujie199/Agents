import asyncio
from typing import Optional, List
import logging
import hashlib
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from core.ports.storage.object_store import ObjectStorePort, ObjectMetadata


class S3ObjectStoreAdapter:
    """
    对象存储适配器（支持 S3/OBS 或本地降级）
    
    生产环境：使用 boto3 连接 S3/OBS
    开发环境：降级到本地文件系统
    """
    
    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        access_key: str = "",
        secret_key: str = "",
        bucket_name: str = "agents-storage",
        region: str = "us-east-1",
        max_pool_connections: int = 50,
        multipart_threshold: int = 8 * 1024 * 1024,
        max_retries: int = 3
    ):
        self._bucket_name = bucket_name
        self._multipart_threshold = multipart_threshold
        self._logger = logging.getLogger(__name__)
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        self._use_local = True
        self._local_base = Path("data/objects")
        self._local_base.mkdir(parents=True, exist_ok=True)
        
        try:
            import boto3
            from botocore.config import Config
            from botocore.exceptions import ClientError
            
            if endpoint_url or (access_key and secret_key):
                config = Config(
                    max_pool_connections=max_pool_connections,
                    retries={'max_attempts': max_retries, 'mode': 'adaptive'}
                )
                
                self._client = boto3.client(
                    's3',
                    endpoint_url=endpoint_url,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=region,
                    config=config
                )
                self._resource = boto3.resource(
                    's3',
                    endpoint_url=endpoint_url,
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=region,
                    config=config
                )
                self._bucket = self._resource.Bucket(bucket_name)
                self._use_local = False
                self._logger.info(f"S3 adapter initialized: {endpoint_url or 'AWS S3'}")
        except ImportError:
            self._logger.warning("boto3 not installed, using local file storage")
        except Exception as e:
            self._logger.warning(f"S3 initialization failed, using local: {e}")
    
    def _get_local_path(self, key: str) -> Path:
        return self._local_base / key.replace(":", "/")
    
    def upload(self, key: str, data: bytes, content_type: Optional[str] = None) -> str:
        if self._use_local:
            path = self._get_local_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return key
        
        try:
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            self._client.put_object(
                Bucket=self._bucket_name,
                Key=key,
                Body=data,
                **extra_args
            )
            return key
        except Exception as e:
            self._logger.error(f"Upload failed: {e}")
            raise
    
    def upload_file(self, key: str, file_path: str) -> str:
        if self._use_local:
            source = Path(file_path)
            path = self._get_local_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.read_bytes())
            return key
        
        self._bucket.upload_file(file_path, key)
        return key
    
    def download(self, key: str) -> Optional[bytes]:
        if self._use_local:
            path = self._get_local_path(key)
            return path.read_bytes() if path.exists() else None
        
        try:
            response = self._client.get_object(Bucket=self._bucket_name, Key=key)
            return response['Body'].read()
        except Exception as e:
            if "NoSuchKey" in str(e):
                return None
            raise
    
    def download_to_file(self, key: str, file_path: str) -> bool:
        if self._use_local:
            data = self.download(key)
            if data is None:
                return False
            Path(file_path).write_bytes(data)
            return True
        
        try:
            self._bucket.download_file(key, file_path)
            return True
        except Exception:
            return False
    
    def delete(self, key: str) -> bool:
        if self._use_local:
            path = self._get_local_path(key)
            path.unlink(missing_ok=True)
            return True
        
        try:
            self._client.delete_object(Bucket=self._bucket_name, Key=key)
            return True
        except Exception:
            return False
    
    def exists(self, key: str) -> bool:
        if self._use_local:
            return self._get_local_path(key).exists()
        
        try:
            self._client.head_object(Bucket=self._bucket_name, Key=key)
            return True
        except Exception:
            return False
    
    def get_metadata(self, key: str) -> Optional[ObjectMetadata]:
        if self._use_local:
            path = self._get_local_path(key)
            if not path.exists():
                return None
            stat = path.stat()
            return ObjectMetadata(
                key=key,
                size=stat.st_size,
                last_modified=str(stat.st_mtime)
            )
        
        try:
            response = self._client.head_object(Bucket=self._bucket_name, Key=key)
            return ObjectMetadata(
                key=key,
                size=response['ContentLength'],
                content_type=response.get('ContentType'),
                etag=response.get('ETag', '').strip('"'),
                last_modified=response.get('LastModified').isoformat() if response.get('LastModified') else None
            )
        except Exception:
            return None
    
    def list_objects(self, prefix: str, limit: Optional[int] = None) -> list[ObjectMetadata]:
        if self._use_local:
            prefix_path = self._local_base / prefix.replace(":", "/")
            objects = []
            if prefix_path.exists():
                for path in prefix_path.rglob("*"):
                    if path.is_file():
                        rel_path = path.relative_to(self._local_base)
                        key = str(rel_path).replace("/", ":")
                        stat = path.stat()
                        objects.append(ObjectMetadata(
                            key=key,
                            size=stat.st_size,
                            last_modified=str(stat.st_mtime)
                        ))
                        if limit and len(objects) >= limit:
                            break
            return objects
        
        try:
            objects = []
            paginator = self._client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self._bucket_name, Prefix=prefix):
                for obj in page.get('Contents', []):
                    objects.append(ObjectMetadata(
                        key=obj['Key'],
                        size=obj['Size'],
                        etag=obj.get('ETag', '').strip('"')
                    ))
                    if limit and len(objects) >= limit:
                        return objects
            return objects
        except Exception:
            return []
    
    def get_signed_url(self, key: str, expires_in: int = 3600) -> Optional[str]:
        if self._use_local:
            return f"file://{self._get_local_path(key).absolute()}"
        
        try:
            return self._client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self._bucket_name, 'Key': key},
                ExpiresIn=expires_in
            )
        except Exception:
            return None
    
    async def upload_async(self, key: str, data: bytes, content_type: Optional[str] = None) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.upload, key, data, content_type)
    
    async def download_async(self, key: str) -> Optional[bytes]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.download, key)
    
    def health(self) -> dict:
        try:
            if self._use_local:
                return {
                    "status": "healthy",
                    "type": "local",
                    "base_dir": str(self._local_base)
                }
            
            start = time.time()
            self._client.head_bucket(Bucket=self._bucket_name)
            latency = (time.time() - start) * 1000
            
            return {
                "status": "healthy",
                "type": "s3",
                "bucket": self._bucket_name,
                "latency_ms": round(latency, 2)
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def close(self) -> None:
        self._executor.shutdown(wait=True)

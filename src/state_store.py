"""
LangGraph 체크포인터 저장소.

- DYNAMODB_TABLE_NAME 이 설정되면 DynamoDB 체크포인터 사용 (세션 상태 영구 저장).
- 미설정 시 메모리(MemorySaver) 사용 (개발/단일 인스턴스용).
"""
import os
from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver


def get_checkpointer() -> BaseCheckpointSaver:
    table_name = os.getenv("DYNAMODB_TABLE_NAME", "").strip()
    if table_name:
        return _get_dynamodb_checkpointer(table_name)
    return _get_memory_checkpointer()


def _get_memory_checkpointer() -> MemorySaver:
    if _get_memory_checkpointer._instance is None:
        _get_memory_checkpointer._instance = MemorySaver()
    return _get_memory_checkpointer._instance


_get_memory_checkpointer._instance: Optional[MemorySaver] = None


def _get_dynamodb_checkpointer(table_name: str) -> BaseCheckpointSaver:
    if _get_dynamodb_checkpointer._instance is None:
        import boto3

        endpoint_url = os.getenv("AWS_ENDPOINT_URL", "").strip()
        if endpoint_url:
            # 로컬 DynamoDB(Localstack, DynamoDB Local)용: boto3가 해당 endpoint 사용하도록 패치
            _region = os.getenv("AWS_REGION", "us-east-1")
            _patch_boto3_for_local(endpoint_url, _region)

        from langgraph_dynamodb_checkpoint import DynamoDBSaver

        ttl = os.getenv("DYNAMODB_TTL_SECONDS")
        saver_kwargs: dict = {"table_name": table_name}
        if ttl:
            try:
                saver_kwargs["ttl_seconds"] = int(ttl)
            except ValueError:
                pass
        _get_dynamodb_checkpointer._instance = DynamoDBSaver(**saver_kwargs)

    return _get_dynamodb_checkpointer._instance


def _patch_boto3_for_local(endpoint_url: str, region_name: str) -> None:
    """로컬 DynamoDB 사용 시 boto3.client 호출에 endpoint_url 주입."""
    import boto3

    _original = boto3.client

    def _patched(service_name: str, *args, **kwargs):
        if service_name == "dynamodb":
            kwargs["endpoint_url"] = endpoint_url
            kwargs.setdefault("region_name", region_name)
        return _original(service_name, *args, **kwargs)

    boto3.client = _patched


_get_dynamodb_checkpointer._instance: Optional[BaseCheckpointSaver] = None

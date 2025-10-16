"""核心数据类型定义。"""

from dataclasses import dataclass
from typing import List, Union


@dataclass
class ChunkInfo:
    """单个文本块的基本信息。"""

    chunk_id: str
    text: str
    source_url: str
    query_id: str | None = None


@dataclass
class FactObject:
    """策略A中使用的事实对象。"""

    summary: str  # 事实摘要
    chunk_ids: List[str]  # 来源chunk的ID列表
    source_url: str  # 主要来源页面URL


# 知识库的类型定义，根据不同策略有不同的数据结构
KnowledgeBase = Union[
    List[FactObject],  # 策略A: 事实对象列表
    List[str],  # 策略B: chunk_id 列表
    str,  # 策略C: 累积的摘要文本
]


__all__ = ["ChunkInfo", "FactObject", "KnowledgeBase"]

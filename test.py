"""测试 ChunkFilteringStrategy 的完整流程。"""

from __future__ import annotations

import json
from pathlib import Path

from core.strategies.chunk_filtering import ChunkFilteringStrategy
from utils.text_cleaning import clean_document_text


def main():
    # 1. 加载缓存的搜索结果
    cache_file = Path(__file__).parent / "cache" / "tavily_exa_results.json"
    print(f"📂 加载搜索结果: {cache_file}")

    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    query = data["query_text"]
    raw_results = data["results"]

    print(f"🔍 查询: {query}")
    print(f"📄 原始结果数: {len(raw_results)}\n")

    # 2. 清洗内容（在test中完成，不在策略内部）
    print("🧹 清洗文档内容...")
    cleaned_results = []

    for item in raw_results:
        url = item["url"]
        content = item.get("content", "")

        # 使用 clean_document_text 清洗
        cleaned = clean_document_text({"content": content})

        if cleaned:
            cleaned_results.append((url, cleaned[:8000]))
            print(f"  ✓ {url[:50]}... ({len(content)} chars) to ({len(cleaned)} chars)")
        else:
            print(f"  ✗ {url[:50]}... (empty after cleaning)")

    print(f"\n✅ 清洗后有效结果数: {len(cleaned_results)}\n")

    # 3. 创建策略并处理
    print("🔧 创建 ChunkFilteringStrategy 实例...")
    strategy = ChunkFilteringStrategy(verbose=True)

    print("⚙️  开始处理（分块 + 过滤）...")
    strategy.process(query, cleaned_results)
    breakpoint()

    print(f"📦 chunk_store 中共有 {len(strategy.chunk_store)} 个chunks")
    print(f"✨ knowledge_base 中保留 {len(strategy.knowledge_base)} 个相关chunks\n")
    breakpoint()

    # 4. 输出相关chunk IDs
    print("🎯 相关的 chunk IDs:")
    for chunk_id in strategy.knowledge_base:
        print(f"  - {chunk_id}")

    # 5. 获取完整上下文
    print("\n📝 生成 checklist context...")
    context = strategy.get_checklist_context()

    print(f"📏 上下文长度: {len(context)} 字符")
    print(f"📊 上下文预览（前500字符）:\n")
    print("=" * 80)
    print(context[:500])
    print("=" * 80)

    # 6. 保存结果到文件
    output_file = Path(__file__).parent / "cache" / "chunk_filtering_result.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n")
        f.write(f"Total chunks: {len(strategy.chunk_store)}\n")
        f.write(f"Relevant chunks: {len(strategy.knowledge_base)}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        f.write(context)

    print(f"\n💾 完整结果已保存到: {output_file}")


if __name__ == "__main__":
    main()

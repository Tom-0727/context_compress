"""测试压缩策略的完整流程。"""

import argparse
import json
from pathlib import Path

from core.strategies.chunk_filtering import ChunkFilteringStrategy
from core.strategies.fact_centric import FactCentricStrategy
from core.strategies.summarization import SummarizationStrategy
from utils.text_cleaning import clean_document_text


def main():
    # 0. 解析命令行参数
    parser = argparse.ArgumentParser(description="测试压缩策略")
    parser.add_argument(
        "--mode",
        choices=["chunk_filtering", "fact_centric", "summarization"],
        default="chunk_filtering",
        help="选择压缩策略模式"
    )
    args = parser.parse_args()

    print(f"🎯 使用策略: {args.mode}\n")

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

    # 3. 根据模式创建不同的策略实例
    print(f"🔧 创建 {args.mode} 策略实例...")
    if args.mode == "chunk_filtering":
        strategy = ChunkFilteringStrategy(verbose=True)
    elif args.mode == "fact_centric":
        strategy = FactCentricStrategy(verbose=True)
    elif args.mode == "summarization":
        strategy = SummarizationStrategy(verbose=True)
    else:
        raise NotImplementedError(f"策略 {args.mode} 尚未实现")

    # 4. 处理搜索结果
    print("⚙️  开始处理...")
    strategy.process(query, cleaned_results)

    # 根据不同策略显示不同的统计信息
    if args.mode == "summarization":
        print(f"📦 knowledge_base 中累积摘要长度: {len(strategy.knowledge_base)} 字符\n")
    else:
        print(f"📦 chunk_store 中共有 {len(strategy.chunk_store)} 个chunks")
        if args.mode == "chunk_filtering":
            print(f"✨ knowledge_base 中保留 {len(strategy.knowledge_base)} 个相关chunks\n")
        elif args.mode == "fact_centric":
            print(f"✨ knowledge_base 中提取 {len(strategy.knowledge_base)} 个事实\n")

    # 5. 获取 checklist 上下文
    print("📝 生成 checklist context...")
    context = strategy.get_checklist_context()

    print(f"📏 上下文长度: {len(context)} 字符")
    print(f"📊 上下文预览（前500字符）:\n")
    print("=" * 80)
    print(context[:500])
    print("=" * 80)

    # 6. 保存结果到文件
    output_file = Path(__file__).parent / "cache" / f"{args.mode}_result.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Mode: {args.mode}\n")
        f.write(f"Query: {query}\n")

        if args.mode == "summarization":
            f.write(f"Summary length: {len(strategy.knowledge_base)} chars\n")
        else:
            f.write(f"Total chunks: {len(strategy.chunk_store)}\n")
            if args.mode == "chunk_filtering":
                f.write(f"Relevant chunks: {len(strategy.knowledge_base)}\n")
            elif args.mode == "fact_centric":
                f.write(f"Extracted facts: {len(strategy.knowledge_base)}\n")

        f.write("\n" + "=" * 80 + "\n\n")
        f.write(context)

    print(f"\n💾 完整结果已保存到: {output_file}")


if __name__ == "__main__":
    main()

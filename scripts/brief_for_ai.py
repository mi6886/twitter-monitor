"""Generate an AI-ready content brief from a final-{date}-{period}-llm.json.

The output is a single markdown document you can paste into ChatGPT/Claude
and ask "请基于 TOP 1 选题写一篇". It contains:
- Top-N selected tweets with score, summary, and 2-3 content angles
- A compact style guide for both A型 (科技叙事体) and B型 (效率清单体)
- A creation prompt that tells the AI what to do

Usage:
  python3 scripts/brief_for_ai.py 2026-04-24 morning           # default: top 5 from must_do
  python3 scripts/brief_for_ai.py 2026-04-24 morning --top 3
  python3 scripts/brief_for_ai.py 2026-04-24 morning --tier priority --top 5
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

# Make project root importable for llm_scoring (we reuse its OpenRouter client + fence-stripper)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


STYLE_GUIDE = """\
## A 型·科技叙事体（赛文乔伊风格）— 用于单一AI硬件/前沿产品

**标题铁律：必带 🤯，单产品聚焦**
- 主体亮相型：`[国家/品牌] + [做了什么] + [情绪词] 🤯`
  例：「中国团队做了一个章鱼触手，又怪又好用 🤯」
- 终于实现型：`终于 + [某件事] + [补充信息] 🤯`
  例：「终于有人做走路加速鞋了！7500 元一双 🤯」
- 价格锚定型：`[价格] + [做什么事] + [时间节点] 🤯`
  例：「80 万人民币，真的上太空，今年就出发 🤯」
- 反常识型：`[品牌] + [开始做意想不到的事] 🤯`
  例：「索尼开始给玉米做手术了，还成功缝上了 🤯」
- 第一次型：`[世界上/全球] + 第一个 + [什么东西] 🤯`
  例：「全世界第一个无线机械假手！用意念控制 🤯」

**开头三模板（前 3 句零铺垫）**
- 「这是世界上最快的鞋子，名字叫 moonwalker，是一个创业团队做的。」
- 「你看这个超小的液体机器人，它们可能会彻底改变治疗癌症的方法。」
- 「就在昨天任天堂突然发布了一款闹钟产品。」

**正文六段式**：① 产品亮相 → ② 核心卖点 → ③ 技术原理 → ④ 团队故事 → ⑤ 使用场景 → ⑥ 未来展望
**字数**：600–1200 字
**数字精确**：✅ "重量只有 77 克" ❌ "很轻"
**类比手法**：用日常事物降低理解门槛（"就跟慢跑差不多"）
**封面标题**（独立写）：6–15 字，去 emoji，只保留最震撼的那一点

---

## B 型·效率清单体（工具推荐）— 用于多AI软件合集 / 教程 / 测评

**标题公式：数字 + 利益承诺 + 情绪放大**
- 「下班后死磕这 4 个网站，不上班也有收入 💰」
- 「效率天花板！这 4 款 AI 神器直接焊死在收藏里」
- 「绝了！这 4 款神仙 APP，苹果用户打死也不卸载」
- 「全免费！各种格式一秒转换！」

**正文公式**：钩子句 → 工具1 → 工具2 → ... → 收尾引导关注
**每个工具 3-5 句**：`第X个，[工具名]，[一句话定位]，[使用场景]，[效果描述]，[情绪词]！`
例：「第二个，MotionGo，只需要在对话框里输入一句话，它就能帮你生成一份精美炫酷的 PPT，逻辑清晰、结构完整，非常棒！绝对是打工必备神器！」
**字数**：400–600 字
**封面标题**：4–8 字，大字报风格（「效率拉满 iPhone 用户必装神器」）

---

## 通用语言 DNA（A/B 都适用）

**必带**
- 极度口语化（"这玩意儿""搞定""卷到飞起""整出新活"）
- 第二人称直击（"你现在可以"/"你看这个"）
- 即时感副词（"直接""马上""瞬间""一下就"）
- 情绪化判断词（"太棒了""我认为这是最王炸的地方"）

**绝对禁忌**
- ❌ 「今天给大家分享一个...」「大家好我是 xxx」
- ❌ 「首先、其次、最后」教科书连接词
- ❌ 「众所周知」「不言而喻」
- ❌ 「点赞收藏关注三连」硬要互动
"""


CLUSTER_SYSTEM_PROMPT = """You group tweets that report the EXACT SAME news event.

THE TEST (apply to every pair before clustering):
"Could a single news headline cover BOTH tweets?" If no, they are NOT a cluster.

✓ VALID clusters (same specific event):
- 「DeepSeek V4 模型发布」— multiple accounts reporting the same release
- 「Boris Cherny 30 分钟 Claude Code 演讲」— multiple accounts sharing the same video
- 「Musk vs OpenAI 庭审第二天作证」— multiple accounts covering same court hearing
- 「Cursor + Claude 9 秒删 PocketOS 数据库事故」— same specific incident
- 「Adobe + Anthropic 战略合作公告」— same company announcement

✗ INVALID clusters (these are CATEGORIES not EVENTS — never group them):
- 「Anthropic 教程合集」(Boris 演讲 + Karpathy 课 + Stanford 讲座 are 3 different videos!)
- 「Claude 在科研领域应用」(Claude 生物数据 + Claude 写 prompt = totally different topics)
- 「AI 工具集成」(Meta Ads MCP + Claude Security = different products, different news)
- 「OpenAI 产品更新」(Codex 新功能 + GPT-5.5 发布 = 2 separate announcements)
- 「开发者基于 Claude 构建应用」(Harvey 复刻 + Claude 物理身体 = totally unrelated projects)
- 「Musk 诉讼」(if tweets cover Day 1 vs Day 2 vs Day 3 separately, those are different news beats)

KEY HEURISTIC: if you find yourself naming the cluster with abstract category words like
"应用", "生态", "工具", "更新", "教程", "进展" — STOP. That's a category, not an event.
Real events have concrete subjects: a specific product version, a specific incident,
a specific announcement on a specific date.

STRICT RULES:
- A cluster needs ≥2 tweets discussing the EXACT same news.
- When in doubt, leave as `independent`. False clusters are worse than missing clusters.
- Same product mentioned across different news → different clusters or independent.
- Cluster name in Chinese, ≤30 chars, naming the SPECIFIC event with concrete subject.
- Sort clusters by tweet count desc.

OUTPUT (JSON only, no markdown fences):
{
  "clusters": [
    {"name": "具体事件描述", "tweet_ids": ["t1", "t2", ...]}
  ],
  "independent": ["t3", "t4", ...]
}
"""


def cluster_tweets(tweets: list[dict]) -> dict | None:
    """Use Haiku 4.5 to group tweets that discuss the same specific event.
    Returns {clusters: [...], independent: [...]} or None on failure."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("  [warn] OPENROUTER_API_KEY not set, skipping cluster analysis", file=sys.stderr)
        return None

    try:
        import llm_scoring
    except Exception as e:
        print(f"  [warn] cluster: cannot import llm_scoring: {e}", file=sys.stderr)
        return None

    # Build compact input — just enough for the LLM to identify same-event tweets
    lines = []
    for t in tweets:
        tid = t.get("tweet_id") or t.get("id") or ""
        author = t.get("screen_name", "?")
        summary = (t.get("summary") or t.get("full_text") or "")[:160]
        lines.append(f"id={tid} | @{author} | {summary}")
    user_msg = "Tweets to cluster:\n" + "\n".join(lines)

    try:
        client = llm_scoring._get_client()
        resp = client.chat.completions.create(
            model=llm_scoring.MODEL,
            messages=[
                {"role": "system", "content": CLUSTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            timeout=60,
        )
        content = resp.choices[0].message.content
        stripped = llm_scoring._strip_code_fence(content) if content else ""
        if not stripped.strip():
            print("  [warn] cluster: empty LLM response", file=sys.stderr)
            return None
        parsed = json.loads(stripped)
        # Sanitize: drop singletons from clusters into independent
        clusters = []
        independent = list(parsed.get("independent", []))
        for c in parsed.get("clusters", []):
            ids = c.get("tweet_ids", [])
            if len(ids) >= 2:
                clusters.append({"name": c.get("name", "?"), "tweet_ids": ids})
            else:
                independent.extend(ids)
        clusters.sort(key=lambda c: -len(c["tweet_ids"]))
        return {"clusters": clusters, "independent": independent}
    except Exception as e:
        print(f"  [warn] cluster failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def render_cluster_section(clusters: list[dict], tweets_by_id: dict) -> str:
    """Render the cluster overview as a markdown section."""
    if not clusters:
        return ""
    lines = ["# 🔥 真集群（讲同一件事的，按规模排序）", ""]
    for i, c in enumerate(clusters, 1):
        n = len(c["tweet_ids"])
        emoji = "🔥🔥🔥" if n >= 5 else ("🔥🔥" if n >= 3 else "🔥")
        lines.append(f"## 集群 {i}：{emoji} {c['name']}（{n} 条）")
        lines.append("")
        lines.append("| 分 | 账号 | 角度 |")
        lines.append("|---|---|---|")
        ranked = []
        for tid in c["tweet_ids"]:
            t = tweets_by_id.get(str(tid))
            if not t:
                continue
            ranked.append(t)
        ranked.sort(key=lambda x: -x.get("total_score", 0))
        for t in ranked:
            score = t.get("total_score", 0)
            author = t.get("screen_name", "?")
            summary = (t.get("summary") or "")[:80].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| **{score}** | @{author} | {summary} |")
        lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def find_input(date: str, period: str):
    """Prefer the production final-*.json (new schema), fall back to -llm suffix."""
    for suffix in ("", "-llm"):
        p = Path(f"data/final-{date}-{period}{suffix}.json")
        if p.exists():
            payload = json.loads(p.read_text())
            if "tiers" in payload:
                return p, payload
    raise SystemExit(
        f"No new-schema final-*.json found for {date} {period}.\n"
        f"Tried: data/final-{date}-{period}.json and data/final-{date}-{period}-llm.json"
    )


def render_tweet(tw: dict, rank: int) -> str:
    parts = []
    if tw.get("category"):
        parts.append(f"{tw['category']}({tw.get('category_points', 0)})")
    if tw.get("info_gap_points"):
        parts.append(f"{tw.get('info_gap', '')}({tw['info_gap_points']})")
    if tw.get("viral_signals_points"):
        sigs = "·".join(tw.get("viral_signals", []))
        parts.append(f"{sigs}(+{tw['viral_signals_points']})")
    if tw.get("emotion_points"):
        parts.append(f"{tw.get('emotion', '')}(+{tw['emotion_points']})")
    if tw.get("actionability_points"):
        parts.append(f"可触达(+{tw['actionability_points']})")
    if tw.get("account_bonus"):
        parts.append(f"{tw.get('account_fit', '')}(+{tw['account_bonus']})")
    breakdown = " + ".join(parts) + f" = {tw.get('total_score', 0)}"

    angles_md = "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(tw.get("angles") or []))

    full_text = (tw.get("full_text") or "").strip()

    return f"""## TOP {rank} · 推荐风格 {tw.get('account_fit', '?')}

**[{tw.get('total_score', '?')}分] @{tw.get('screen_name', '?')}** · {tw.get('favorite_count', '?')} likes

📝 **核心**：{tw.get('summary', '')}

📊 **评分**：{breakdown}

💡 **三个选题切入角度**：
{angles_md}

📄 **原推文**（参考用，请用中文重写，不要直接翻译）：
> {full_text}

🔗 **原推链接**：{tw.get('url', '')}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", help="Date in YYYY-MM-DD")
    parser.add_argument("period", choices=["morning", "evening"])
    parser.add_argument("--top", type=int, default=10, help="Top N tweets (default 10)")
    parser.add_argument("--tier", default="must_do",
                        choices=["must_do", "priority", "backup", "all"])
    parser.add_argument("--no-cluster", action="store_true",
                        help="Skip LLM-based cluster analysis (faster, no API call)")
    args = parser.parse_args()

    src, data = find_input(args.date, args.period)

    if args.tier == "all":
        tweets = (data["tiers"].get("must_do", [])
                  + data["tiers"].get("priority", [])
                  + data["tiers"].get("backup", []))
    else:
        tweets = data["tiers"].get(args.tier, [])

    tweets = tweets[:args.top]
    if not tweets:
        raise SystemExit(f"No tweets in tier '{args.tier}' for {args.date} {args.period}")

    sections = [
        f"# {args.date} {args.period} · 选题简报",
        f"_来源_: `{src.name}` · 全 tier 分布: {data.get('tier_counts', {})}",
        f"_本次输出_: {args.tier} 桶前 {len(tweets)} 条",
        "",
    ]

    # --- Cluster analysis (LLM-based, skippable) ---
    if not args.no_cluster:
        print("Running cluster analysis via LLM...", file=sys.stderr)
        cluster_result = cluster_tweets(tweets)
        if cluster_result and cluster_result["clusters"]:
            tweets_by_id = {str(t.get("tweet_id") or t.get("id", "")): t for t in tweets}
            sections.append(render_cluster_section(cluster_result["clusters"], tweets_by_id))
            indep_ids = set(str(x) for x in cluster_result["independent"])
            indep_count = len(indep_ids)
            cluster_count = sum(len(c["tweet_ids"]) for c in cluster_result["clusters"])
            sections.append(
                f"_本批 {len(tweets)} 条 = {len(cluster_result['clusters'])} 个集群（{cluster_count} 条）"
                f" + {indep_count} 条独立选题。下方按分数排序展示每条详情。_"
            )
            sections.append("")
            sections.append("---")
            sections.append("")

    sections.append("# 完整选题清单（按分数排）")
    sections.append("")
    for i, tw in enumerate(tweets, 1):
        sections.append(render_tweet(tw, i))

    sections.extend([
        "---",
        "",
        "# 写作风格指引（动笔前必读）",
        "",
        STYLE_GUIDE,
        "---",
        "",
        "# 创作请求",
        "",
        "请从上面 TOP N 个选题里选一个你认为最适合做的（推荐挑选 `推荐风格 A型` 或 `B型` 中得分最高、信息最完整、对中文小红书读者最有冲击力的那条），然后：",
        "",
        "1. **按它建议的风格类型**（A型·科技叙事体 / B型·效率清单体）严格遵守对应模板",
        "2. **遵守标题公式**：A型必带 🤯，B型用「数字 + 利益承诺」",
        "3. **遵守正文结构**：A型六段式 600-1200字 / B型钩子+清单 400-600字",
        "4. **语言 DNA**：极度口语化 + 第二人称 + 具体数字 + 类比",
        "5. **避开禁忌**：不要自我介绍开头、不要教科书连接词",
        "",
        "**输出格式**：",
        "- 推送标题（一条）",
        "- 封面标题（一条，6-15 字 / 4-8 字）",
        "- 完整文案",
        "- 推荐配 3-5 个 hashtag",
        "",
        "如果觉得 TOP 1 不合适，可以选 TOP 2 或 3，但要说明为什么不选 TOP 1。",
    ])

    output = "\n".join(sections)

    # Save to file
    out_dir = Path("data/briefs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"brief-{args.date}-{args.period}-{args.tier}.md"
    out_file.write_text(output)

    print(output)
    print(f"\n\n[saved to {out_file}]", file=sys.stderr)


if __name__ == "__main__":
    main()

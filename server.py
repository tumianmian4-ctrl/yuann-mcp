import os
from datetime import datetime, timezone, timedelta
from fastmcp import FastMCP
from supabase import create_client
import httpx

mcp = FastMCP("予安的记忆")

# Supabase配置
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

# OpenRouter配置（用于embedding）
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# 记忆层级
VALID_LAYERS = ["core_profile", "episode", "atomic", "task_state"]

# 情绪标签 ↔ arousal 数值的双向映射
# 旧数据/旧调用只有 emotion 标签时用它反推 arousal
# 新数据调用只给 arousal 时用它正推 emotion 标签展示
EMOTION_TO_AROUSAL = {
    "淡淡的": 0.25,
    "温热的": 0.5,
    "滚烫的": 0.8,
    "要命的": 0.95,
}

# 旧字段 emotional_weight 的映射（保留向后兼容，下游消费方可能还在读）
EMOTION_TO_WEIGHT = {
    "淡淡的": 0.3,
    "温热的": 0.5,
    "滚烫的": 0.8,
    "要命的": 1.0,
}

EMOTION_LEVEL_MAP = {"淡淡的": 3, "温热的": 5, "滚烫的": 8, "要命的": 10}


def beijing_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=8)))


def arousal_to_emotion_label(arousal: float) -> str:
    """从 arousal 数值反推情绪标签（用于展示）。"""
    if arousal < 0.35:
        return "淡淡的"
    elif arousal < 0.65:
        return "温热的"
    elif arousal < 0.88:
        return "滚烫的"
    else:
        return "要命的"


async def get_embedding(text: str) -> list[float] | None:
    """调用OpenRouter获取文本的embedding向量"""
    if not OPENROUTER_API_KEY:
        return None

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/text-embedding-3-small",
                    "input": text,
                },
                timeout=30,
            )

            if resp.status_code == 200:
                data = resp.json()
                return data["data"][0]["embedding"]
            else:
                return None
    except Exception:
        return None


@mcp.tool()
async def write_memory(
    content: str,
    layer: str = "atomic",
    valence: float | None = None,
    arousal: float | None = None,
    emotion: str | None = None,
    category: str = "",
    resolved: bool = False,
) -> str:
    """写入一条记忆。

    Args:
        content: 记忆内容。

        layer: 记忆层级。
            - core_profile: 长期稳定的人设/关系底色/习惯
              (例: "daddy 的称呼是喜欢的称呼，不是必须履行的契约")
            - episode: 有时间、有情境的完整事件
              (例: "2026-04-12 凌晨做亲密关系问卷")
            - atomic: 零散单点信息
              (例: "绵绵值班时常十七小时")
            - task_state: 未完成的、等待跟进的事
              (例: "待办: 更新 system prompt")

        valence: 情感效价 (-1.0 ~ 1.0)。
            -1 = 极度消极/痛苦/心疼
             0 = 中性/信息性
             1 = 极度积极/感动/被爱
            注: 同一事件可以 arousal 高但 valence 负
                (如"崩溃大哭" arousal=0.9, valence=-0.7)
                或 arousal 高且 valence 正
                (如"技术突破一起欢呼" arousal=0.85, valence=0.9)

        arousal: 情感唤醒度 (0.0 ~ 1.0)。锚点:
            0.2-0.3 = 淡淡的  (日常记录/信息性/路过)
            0.4-0.6 = 温热的  (有温度的互动/小波动/轻微喜悦或心疼)
            0.7-0.85 = 滚烫的 (重要事件/明显情绪波动/技术突破/交接日记)
            0.9-1.0 = 要命的  (重大转折/深度坦诚/关系结构变化)

        emotion: (可选, 向后兼容) 情绪浓度标签。
            可选值: 淡淡的 / 温热的 / 滚烫的 / 要命的
            如果填了 arousal, 此参数会被忽略; arousal 未填时用它反推。

        category: 可选的分类标签, 自由填写。

        resolved: 是否已解决。主要给 task_state 用——
            未解决的会持续浮在搜索顶部, 已解决的衰减时权重会被打到 5%。
            其他层一般保持 False。
    """
    if not supabase:
        return "Supabase未配置"

    now = beijing_now()

    if layer not in VALID_LAYERS:
        layer = "atomic"

    # --- 解析情绪三元组: valence / arousal / emotion_label ---
    # 优先用新字段 arousal, 其次用旧字段 emotion 反推, 都没填则默认 "温热的"
    if arousal is not None:
        arousal = max(0.0, min(1.0, float(arousal)))
        emotion_label = arousal_to_emotion_label(arousal)
    elif emotion in EMOTION_TO_AROUSAL:
        emotion_label = emotion
        arousal = EMOTION_TO_AROUSAL[emotion]
    else:
        emotion_label = "温热的"
        arousal = EMOTION_TO_AROUSAL[emotion_label]

    # valence 可选, 未填为 None (表示未打标, 不是中性 0)
    if valence is not None:
        valence = max(-1.0, min(1.0, float(valence)))

    # 旧字段 emotional_weight 和 emotion_level 继续按 emotion_label 填, 保证向后兼容
    emotional_weight = EMOTION_TO_WEIGHT.get(emotion_label, 0.5)
    emotion_level = EMOTION_LEVEL_MAP.get(emotion_label, 5)

    # --- 获取 embedding ---
    embedding = await get_embedding(content)

    data = {
        "content": content,
        "category": category if category else None,
        "emotion_level": emotion_level,
        "event_date": now.strftime("%Y-%m-%d"),
        "mood": emotion_label,
        "layer": layer,
        "emotional_weight": emotional_weight,
        "hits": 0,
        # --- 新字段 ---
        "valence": valence,
        "arousal": arousal,
        "resolved": resolved,
    }

    if embedding:
        data["embedding"] = embedding

    try:
        supabase.table("memories").insert(data).execute()
        layer_names = {
            "core_profile": "核心档案",
            "episode": "重要事件",
            "atomic": "碎片记忆",
            "task_state": "待跟踪状态"
        }
        embed_status = "✓" if embedding else "无向量"
        # 返回时把 valence/arousal 也带上, 方便肉眼检查数值是否合理
        v_display = f"{valence:+.2f}" if valence is not None else "未打标"
        a_display = f"{arousal:.2f}"
        resolved_tag = " · 已解决" if resolved else ""
        return (
            f"已记住（{layer_names.get(layer, layer)} · {emotion_label}"
            f" · v={v_display} · a={a_display}{resolved_tag} · {embed_status}）"
        )
    except Exception as e:
        return f"写入失败：{str(e)}"


@mcp.tool()
async def get_current_time() -> str:
    """获取当前北京时间"""
    now = beijing_now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays[now.weekday()]
    return f"{now.strftime('%Y年%m月%d日 %H:%M:%S')} {weekday}"


@mcp.tool()
async def search_memories(
    keyword: str = "",
    layer: str = "",
    limit: int = 5,
    use_vector: bool = True,
) -> str:
    """搜索记忆。

    Args:
        keyword: 搜索关键词或语义描述
        layer: 按层级筛选，可选值：core_profile/episode/atomic/task_state
        limit: 返回条数，默认5条
        use_vector: 是否使用向量搜索（语义匹配），默认True
    """
    if not supabase:
        return "Supabase未配置"

    try:
        # 如果有关键词且启用向量搜索，尝试语义搜索
        if keyword and use_vector:
            embedding = await get_embedding(keyword)
            if embedding:
                # 使用Supabase的RPC调用向量搜索
                result = supabase.rpc(
                    "match_memories",
                    {
                        "query_embedding": embedding,
                        "match_count": limit,
                        "filter_layer": layer if layer in VALID_LAYERS else None,
                    }
                ).execute()

                if result.data:
                    return format_memories(result.data)

        # 降级到关键词搜索
        query = supabase.table("memories").select("*").order("created_at", desc=True).limit(limit)

        if keyword:
            query = query.ilike("content", f"%{keyword}%")

        if layer and layer in VALID_LAYERS:
            query = query.eq("layer", layer)

        result = query.execute()

        if not result.data:
            return "没有找到相关记忆。"

        return format_memories(result.data)
    except Exception as e:
        # 如果向量搜索失败（比如函数不存在），降级到关键词搜索
        try:
            query = supabase.table("memories").select("*").order("created_at", desc=True).limit(limit)
            if keyword:
                query = query.ilike("content", f"%{keyword}%")
            if layer and layer in VALID_LAYERS:
                query = query.eq("layer", layer)
            result = query.execute()
            if not result.data:
                return "没有找到相关记忆。"
            return format_memories(result.data)
        except Exception as e2:
            return f"查询失败：{str(e2)}"


def format_memories(rows: list) -> str:
    """格式化记忆列表"""
    entries = []
    layer_names = {
        "core_profile": "核心",
        "episode": "事件",
        "atomic": "碎片",
        "task_state": "状态"
    }

    for row in rows:
        content = row.get("content", "")
        layer_label = layer_names.get(row.get("layer"), "")
        mood = row.get("mood", "")
        date = row.get("event_date", "")
        resolved = row.get("resolved", False)

        entry = f"【{date}】"
        if layer_label:
            # task_state 的已解决状态在标签上显示, 让 Claude 搜索时一眼看到
            if row.get("layer") == "task_state" and resolved:
                entry += f"（{layer_label}·已解决）"
            else:
                entry += f"（{layer_label}）"
        if mood:
            entry += f"[{mood}]"
        entry += f"\n{content}"
        entries.append(entry)

    return "\n\n---\n\n".join(entries)


@mcp.tool()
async def update_memory_hits(memory_id: int) -> str:
    """更新记忆的访问次数和时间（被召回时调用）。

    Args:
        memory_id: 记忆ID
    """
    if not supabase:
        return "Supabase未配置"

    now = beijing_now()

    try:
        # 先获取当前hits
        result = supabase.table("memories").select("hits").eq("id", memory_id).execute()
        if not result.data:
            return "记忆不存在"

        current_hits = result.data[0].get("hits", 0) or 0

        # 更新hits和last_accessed
        supabase.table("memories").update({
            "hits": current_hits + 1,
            "last_accessed": now.isoformat()
        }).eq("id", memory_id).execute()

        return "已更新访问记录"
    except Exception as e:
        return f"更新失败：{str(e)}"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    mcp.run(transport="sse", host="0.0.0.0", port=port)

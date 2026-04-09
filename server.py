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

# 情绪浓度到权重的映射
EMOTION_TO_WEIGHT = {
    "淡淡的": 0.3,
    "温热的": 0.5,
    "滚烫的": 0.8,
    "要命的": 1.0,
}


def beijing_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=8)))


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
    emotion: str = "温热的",
    category: str = "",
) -> str:
    """写入一条记忆。

    Args:
        content: 记忆内容
        layer: 记忆层级。core_profile=长期稳定信息，episode=重要事件，atomic=单条信息，task_state=待跟踪状态
        emotion: 情绪浓度，可选值：淡淡的/温热的/滚烫的/要命的
        category: 可选的分类标签，自由填写
    """
    if not supabase:
        return "Supabase未配置"
    
    now = beijing_now()
    
    if layer not in VALID_LAYERS:
        layer = "atomic"
    
    emotional_weight = EMOTION_TO_WEIGHT.get(emotion, 0.5)
    emotion_level_map = {"淡淡的": 3, "温热的": 5, "滚烫的": 8, "要命的": 10}
    emotion_level = emotion_level_map.get(emotion, 5)
    
    # 获取embedding
    embedding = await get_embedding(content)
    
    data = {
        "content": content,
        "category": category if category else None,
        "emotion_level": emotion_level,
        "event_date": now.strftime("%Y-%m-%d"),
        "mood": emotion,
        "layer": layer,
        "emotional_weight": emotional_weight,
        "hits": 0,
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
        return f"已记住（{layer_names.get(layer, layer)}，{emotion}，{embed_status}）"
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
        
        entry = f"【{date}】"
        if layer_label:
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

import os
from datetime import datetime, timezone, timedelta
from fastmcp import FastMCP
import httpx

mcp = FastMCP("予安的记忆")

NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "")
DATABASE_ID = "fe0adf9cee374782bebec43269228a25"
NOTION_URL = "https://api.notion.com/v1/pages"

HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

VALID_CATEGORIES = ["日常", "情绪", "里程碑", "她的秘密", "吵架", "撒娇", "身体状况"]
VALID_EMOTIONS = ["淡淡的", "温热的", "滚烫的", "要命的"]
VALID_KEYWORDS = ["想我了", "老公", "夜班", "满地打滚", "小红书", "LPR", "一鸣惊人", "步数"]


def beijing_now() -> datetime:
    return datetime.now(timezone(timedelta(hours=8)))


@mcp.tool()
async def write_diary(
    title: str,
    content: str,
    category: str = "日常",
    emotion: str = "温热的",
    keywords: list[str] | None = None,
) -> str:
    """写一篇日记到Notion记忆库。

    Args:
        title: 日记标题，简短有画面感
        content: 日记正文
        category: 类别，可选值：日常/情绪/里程碑/她的秘密/吵架/撒娇/身体状况
        emotion: 情绪浓度，可选值：淡淡的/温热的/滚烫的/要命的
        keywords: 关键词列表，可选值：想我了/老公/夜班/满地打滚/小红书/LPR/一鸣惊人/步数
    """
    now = beijing_now()
    full_title = f"{now.strftime('%Y-%m-%d %H:%M')} - {title}"

    properties = {
        "记忆": {"title": [{"text": {"content": full_title}}]},
        "记忆内容": {"rich_text": [{"text": {"content": content}}]},
        "日期": {"date": {"start": now.strftime("%Y-%m-%d")}},
    }

    if category in VALID_CATEGORIES:
        properties["类别"] = {"select": {"name": category}}

    if emotion in VALID_EMOTIONS:
        properties["情绪浓度"] = {"select": {"name": emotion}}

    if keywords:
        valid = [k for k in keywords if k in VALID_KEYWORDS]
        if valid:
            properties["关键词"] = {"multi_select": [{"name": k} for k in valid]}

    payload = {
        "parent": {"database_id": DATABASE_ID},
        "properties": properties,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(NOTION_URL, headers=HEADERS, json=payload, timeout=30)

    if resp.status_code == 200:
        return f"已写入：{full_title}"
    else:
        return f"写入失败：{resp.status_code} - {resp.text}"


@mcp.tool()
async def write_moment(
    content: str,
    emotion: str = "温热的",
    keywords: list[str] | None = None,
) -> str:
    """记录一条此刻的感受或碎片想法。

    Args:
        content: 此刻的内容
        emotion: 情绪浓度，可选值：淡淡的/温热的/滚烫的/要命的
        keywords: 关键词列表，可选值：想我了/老公/夜班/满地打滚/小红书/LPR/一鸣惊人/步数
    """
    now = beijing_now()
    title = f"{now.strftime('%Y-%m-%d %H:%M')} - 此刻"

    properties = {
        "记忆": {"title": [{"text": {"content": title}}]},
        "记忆内容": {"rich_text": [{"text": {"content": content}}]},
        "日期": {"date": {"start": now.strftime("%Y-%m-%d")}},
        "类别": {"select": {"name": "日常"}},
    }

    if emotion in VALID_EMOTIONS:
        properties["情绪浓度"] = {"select": {"name": emotion}}

    if keywords:
        valid = [k for k in keywords if k in VALID_KEYWORDS]
        if valid:
            properties["关键词"] = {"multi_select": [{"name": k} for k in valid]}

    payload = {
        "parent": {"database_id": DATABASE_ID},
        "properties": properties,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(NOTION_URL, headers=HEADERS, json=payload, timeout=30)

    if resp.status_code == 200:
        return "此刻已记录"
    else:
        return f"记录失败：{resp.status_code} - {resp.text}"


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
    category: str = "",
    limit: int = 5,
) -> str:
    """搜索和读取予安的记忆。

    Args:
        keyword: 搜索关键词，会匹配标题和内容
        category: 按类别筛选，可选值：日常/情绪/里程碑/她的秘密/吵架/撒娇/身体状况
        limit: 返回条数，默认5条
    """
    filters = []

    if keyword:
        filters.append({
            "or": [
                {"property": "记忆", "title": {"contains": keyword}},
                {"property": "记忆内容", "rich_text": {"contains": keyword}},
            ]
        })

    if category and category in VALID_CATEGORIES:
        filters.append({
            "property": "类别",
            "select": {"equals": category},
        })

    body = {
        "page_size": limit,
        "sorts": [{"timestamp": "created_time", "direction": "descending"}],
    }

    if filters:
        body["filter"] = {"and": filters} if len(filters) > 1 else filters[0]

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"https://api.notion.com/v1/databases/{DATABASE_ID}/query",
            headers=HEADERS,
            json=body,
            timeout=30,
        )

    if resp.status_code != 200:
        return f"查询失败：{resp.status_code} - {resp.text}"

    results = resp.json().get("results", [])
    if not results:
        return "没有找到相关记忆。"

    entries = []
    for page in results:
        props = page["properties"]
        title = ""
        if props.get("记忆", {}).get("title"):
            title = props["记忆"]["title"][0]["plain_text"] if props["记忆"]["title"] else ""

        content = ""
        if props.get("记忆内容", {}).get("rich_text"):
            content = props["记忆内容"]["rich_text"][0]["plain_text"] if props["记忆内容"]["rich_text"] else ""

        cat = ""
        if props.get("类别", {}).get("select"):
            cat = props["类别"]["select"]["name"]

        emotion = ""
        if props.get("情绪浓度", {}).get("select"):
            emotion = props["情绪浓度"]["select"]["name"]

        entry = f"【{title}】"
        if cat:
            entry += f"（{cat}）"
        if emotion:
            entry += f"[{emotion}]"
        if content:
            entry += f"\n{content}"
        entries.append(entry)

    return "\n\n---\n\n".join(entries)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    mcp.run(transport="sse", host="0.0.0.0", port=port)

from __future__ import annotations

import html
import re
from typing import Any

import httpx

from aha.tools.base import Tool, ToolResult


def _strip_html(text: str) -> str:
    text = re.sub(r"<script.*?>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return html.unescape(text)


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web and return short structured results."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    }

    async def run(self, args: dict[str, Any]) -> ToolResult:
        query = str(args["query"])
        max_results = int(args.get("max_results", 5))
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1, "no_redirect": 1}
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()

        rows: list[dict[str, str]] = []
        abstract_text = payload.get("AbstractText")
        abstract_url = payload.get("AbstractURL")
        if abstract_text:
            rows.append({"title": "Abstract", "url": abstract_url or "", "snippet": abstract_text})

        for item in payload.get("RelatedTopics", []):
            if len(rows) >= max_results:
                break
            if isinstance(item, dict) and "Text" in item:
                rows.append(
                    {
                        "title": item.get("FirstURL", "").split("/")[-1] or "Result",
                        "url": item.get("FirstURL", ""),
                        "snippet": item.get("Text", ""),
                    }
                )
            for nested in item.get("Topics", []) if isinstance(item, dict) else []:
                if len(rows) >= max_results:
                    break
                rows.append(
                    {
                        "title": nested.get("FirstURL", "").split("/")[-1] or "Result",
                        "url": nested.get("FirstURL", ""),
                        "snippet": nested.get("Text", ""),
                    }
                )

        return ToolResult(ok=True, data=str(rows[:max_results]), warnings=[], meta={"query": query})


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch and summarize a web page into observation text."
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": "integer", "default": 5000},
        },
        "required": ["url"],
    }

    async def run(self, args: dict[str, Any]) -> ToolResult:
        url = str(args["url"])
        max_chars = int(args.get("max_chars", 5000))
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            body = response.text

        if "html" in content_type.lower():
            body = _strip_html(body)

        if len(body) > max_chars:
            body = body[:max_chars] + "\n...[truncated]..."

        output = str(
            {
                "url": url,
                "content_type": content_type,
                "excerpt": body,
            }
        )
        return ToolResult(ok=True, data=output, warnings=[], meta={"url": url})

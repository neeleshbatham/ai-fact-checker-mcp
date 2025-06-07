from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("Wikipedia MCP Server")

@mcp.tool()
def search_wikipedia(query: str) -> list[str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "utf8": "",
        "format": "json"
    }
    url = "https://en.wikipedia.org/w/api.php"
    resp = requests.get(url, params=params)
    print("======================Wikipedia MCP Server======================")
    print(resp.json())
    data = resp.json()
    evidence = []
    for item in data.get("query", {}).get("search", [])[:3]:
        snippet = item.get("snippet", "")
        title = item.get("title", "")
        link = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        clean_snippet = snippet.replace('<span class="searchmatch">','').replace('</span>','')
        evidence.append(f"{clean_snippet} (Source: {link})")
    if not evidence:
        evidence = ["No relevant Wikipedia evidence found."]
    return evidence

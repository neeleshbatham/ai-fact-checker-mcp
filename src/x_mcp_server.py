from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("X MCP Server")

X_BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAD1D2QEAAAAA19eeBD1%2Bvk3%2FMOqh7H1pxRMTOww%3DLt5zwNMWYLGN0Hz69Wi7XHXtfnplWdGLbpzAG42gr0L4aEXrp9"

@mcp.tool()
def search_x(query: str) -> list[str]:
    """Search X (Twitter) for a given query and return recent posts."""
    if X_BEARER_TOKEN == "AAAAAAAAAAAAAAAAAAAAAD1D2QEAAAAA19eeBD1%2Bvk3%2FMOqh7H1pxRMTOww%3DLt5zwNMWYLGN0Hz69Wi7XHXtfnplWdGLbpzAG42gr0L4aEXrp9":
        return [f"No X API key configured. Would search for: {query}"]
    headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
    params = {"query": query, "max_results": 5}
    resp = requests.get("https://api.twitter.com/2/tweets/search/recent", headers=headers, params=params)
    if resp.status_code == 200:
        data = resp.json()
        tweets = []
        for t in data.get("data", []):
            text = t.get("text", "")
            tweets.append(f"{text} (X Post)")
        return tweets if tweets else ["No relevant X posts found."]
    else:
        return [f"X API error: {resp.status_code}"]

# Again, no .serve() and no __main__ block!

"""Exposes the adaptive retrieval agent as MCP tools, so any MCP-compatible
client (Claude Desktop, Claude Code, another agent) can call it directly
rather than treating this as a closed application."""
from mcp.server.mcpserver import MCPServer

from src.agent.graph import ask

mcp = MCPServer("adaptive-retrieval-agent")


@mcp.tool()
def query_knowledge_base(question: str) -> dict:
    """Answer a question against the indexed knowledge base. Retrieval quality
    is self-graded; the agent reformulates and retries on low confidence, and
    explicitly abstains rather than guessing if it still can't find enough
    context after retrying."""
    result = ask(question)
    return {
        "answer": result["answer"],
        "abstained": result["abstained"],
        "retries_used": result["retry_count"],
        "sources": sorted({c.source for c in result.get("retrieved_chunks", [])}),
    }


if __name__ == "__main__":
    mcp.run()

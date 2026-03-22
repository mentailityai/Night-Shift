# =============================================================================
# Night-Shift — MCP Server
# =============================================================================
# Initialises and runs the Model Context Protocol (MCP) server that
# exposes the ``search_active_rules`` tool to primary AI drafting agents.
#
# The transport layer is configurable via the ``MCP_TRANSPORT`` environment
# variable:
#   - "stdio"  → Standard I/O transport (default, for local tool-use)
#   - "sse"    → Server-Sent Events over HTTP (for network-accessible agents)
#
# Run directly:
#   python -m app.mcp.server
# =============================================================================

from __future__ import annotations

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.mcp.tools import search_active_rules

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# MCP Server Instance
# ---------------------------------------------------------------------------
server = Server("nightshift-mcp")


# ---------------------------------------------------------------------------
# Tool Registration
# ---------------------------------------------------------------------------
@server.list_tools()
async def list_tools() -> list[Tool]:
    """
    Advertise the available MCP tools to connected clients.

    Currently exposes a single tool: ``search_active_rules``.
    """
    return [
        Tool(
            name="search_active_rules",
            description=(
                "Search the Night-Shift preference database for relevant "
                "user formatting rules and style preferences.  Call this "
                "before generating any text to ensure your output matches "
                "the user's established preferences."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A natural-language description of the current "
                            "drafting task (e.g., 'drafting an indemnification "
                            "clause for a biotech license agreement')."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    Handle incoming tool calls from MCP clients.

    Parameters
    ----------
    name : str
        The name of the tool being called.
    arguments : dict
        The arguments passed to the tool.

    Returns
    -------
    list[TextContent]
        The tool's response formatted as MCP TextContent blocks.
    """
    if name == "search_active_rules":
        query = arguments.get("query", "")
        if not query:
            return [TextContent(
                type="text",
                text="Error: 'query' argument is required.",
            )]

        results = await search_active_rules(query)

        if not results:
            return [TextContent(
                type="text",
                text="No matching preference rules found.",
            )]

        # Format the results into a readable text block for the agent
        import json
        formatted = json.dumps(results, indent=2)
        return [TextContent(type="text", text=formatted)]

    return [TextContent(
        type="text",
        text=f"Error: Unknown tool '{name}'.",
    )]


# ---------------------------------------------------------------------------
# Server Runner
# ---------------------------------------------------------------------------
async def run_stdio() -> None:
    """Run the MCP server using stdio transport."""
    logger.info("mcp_server_starting", transport="stdio")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


async def run_sse() -> None:
    """
    Run the MCP server using SSE (Server-Sent Events) transport.

    This makes the MCP server accessible over HTTP for remote agents.
    """
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Route, Mount
    import uvicorn

    settings = get_settings()
    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0],
                streams[1],
                server.create_initialization_options(),
            )

    starlette_app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse_transport.handle_post_message),
        ],
    )

    logger.info(
        "mcp_server_starting",
        transport="sse",
        host=settings.mcp_sse_host,
        port=settings.mcp_sse_port,
    )

    config = uvicorn.Config(
        starlette_app,
        host=settings.mcp_sse_host,
        port=settings.mcp_sse_port,
        log_level="info",
    )
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
async def main() -> None:
    """
    Start the MCP server using the transport configured in ``.env``.
    """
    setup_logging()
    settings = get_settings()

    if settings.mcp_transport == "sse":
        await run_sse()
    else:
        await run_stdio()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

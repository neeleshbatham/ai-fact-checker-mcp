"""
Main entry point for the Wikipedia MCP server.
"""

import argparse
import logging
import sys 
import os
from dotenv import load_dotenv

from server import create_server

# Load environment variables from .env file if it exists
load_dotenv()

def main():
    """Run the Wikipedia MCP server."""
    
    # parser = argparse.ArgumentParser(description="Wikipedia MCP Server")
    # parser.add_argument(
    #     "--log-level", 
    #     type=str, 
    #     default="INFO",
    #     choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    #     help="Logging level"
    # )
    # parser.add_argument(
    #     "--transport",
    #     type=str,
    #     default="stdio",
    #     choices=["stdio", "sse"],
    #     help="Transport protocol for MCP communication (stdio for Claude Desktop, sse for HTTP streaming)"
    # )
    # args = parser.parse_args()

    # # Configure logging - use basicConfig for simplicity but ensure it goes to stderr
    # logging.basicConfig(
    #     level=getattr(logging, args.log_level.upper()),
    #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    #     stream=sys.stderr,
    #     force=True  # Override any existing basicConfig
    # )
    

    # Create and start the server
    server = create_server()
    
    # Log startup information using our configured logger
    # logger.info("Starting Wikipedia MCP server with %s transport", args.transport)
    
    # if args.transport != "stdio":
    #     config_template = """
    #     {
    #       "mcpServers": {
    #         "wikipedia": {
    #           "command": "wikipedia-mcp"
    #         }
    #       }
    #     }
    #     """
    #     logger.info("To use with Claude Desktop, configure claude_desktop_config.json with:%s", config_template)
    # else:
    #     logger.info("Using stdio transport - suppressing direct stdout messages for MCP communication.")
    #     logger.info("To use with Claude Desktop, ensure 'wikipedia-mcp' command is in your claude_desktop_config.json.")
    
    server.run(transport="streamable-http")
if __name__ == "__main__":
    main()
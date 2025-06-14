"""
Main entry point for the Twitter MCP server.
"""

import logging
import sys
from dotenv import load_dotenv
import uvicorn

from server import app

# Load environment variables from .env file if it exists
load_dotenv()

def main():
    """Run the Twitter MCP server."""
    uvicorn.run(app, host="0.0.0.0", port=8002)

if __name__ == "__main__":
    main() 
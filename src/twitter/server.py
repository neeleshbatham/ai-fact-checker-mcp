"""
Twitter MCP server implementation.
"""

import logging
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP
from twitter_client import TwitterClient

logger = logging.getLogger(__name__)

def create_server() -> FastMCP:
    """Create and configure the Twitter MCP server."""
    server = FastMCP(
        name="Twitter",
        description="Retrieve information from Twitter to provide context to LLMs."
    )

    # Load environment variables
    load_dotenv()
    
    # Get Twitter API bearer token from environment
    X_BEARER_TOKEN = "<YOUR_BEARER_TOKEN>"
    bearer_token = X_BEARER_TOKEN
    if not bearer_token:
        logger.warning("No Twitter API bearer token found in environment variables")
        bearer_token = "dummy_token"  # Will cause API calls to fail gracefully

    # Initialize Twitter client
    twitter_client = TwitterClient(bearer_token=bearer_token)

    # Register tools
    @server.tool()
    def search_tweets(query: str, limit: int = 10) -> Dict[str, Any]:
        """Search Twitter for tweets matching a query."""
        logger.info(f"Tool: Searching Twitter for: {query}")
        results = twitter_client.search_tweets(query, limit=limit)
        return {
            "query": query,
            "results": results
        }

    @server.tool()
    def get_user_tweets(username: str, limit: int = 10) -> Dict[str, Any]:
        """Get recent tweets from a specific user."""
        logger.info(f"Tool: Getting tweets for user: {username}")
        tweets = twitter_client.get_user_tweets(username, limit=limit)
        return {
            "username": username,
            "tweets": tweets
        }

    @server.tool()
    def get_trending_topics(woeid: int = 1) -> Dict[str, Any]:
        """Get trending topics for a specific location."""
        logger.info(f"Tool: Getting trending topics for WOEID: {woeid}")
        topics = twitter_client.get_trending_topics(woeid=woeid)
        return {
            "woeid": woeid,
            "topics": topics
        }

    @server.tool()
    def get_tweet_details(tweet_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific tweet."""
        logger.info(f"Tool: Getting details for tweet: {tweet_id}")
        tweet = twitter_client.get_tweet_details(tweet_id)
        return {
            "tweet_id": tweet_id,
            "tweet": tweet
        }

    @server.tool()
    def get_user_profile(username: str) -> Dict[str, Any]:
        """Get detailed information about a Twitter user."""
        logger.info(f"Tool: Getting profile for user: {username}")
        profile = twitter_client.get_user_profile(username)
        return {
            "username": username,
            "profile": profile
        }

    # Register resources
    @server.resource("/search/{query}")
    def search(query: str) -> Dict[str, Any]:
        """Search Twitter for tweets matching a query."""
        logger.info(f"Searching Twitter for: {query}")
        results = twitter_client.search_tweets(query, limit=10)
        return {
            "query": query,
            "results": results
        }

    @server.resource("/user/{username}/tweets")
    def user_tweets(username: str) -> Dict[str, Any]:
        """Get recent tweets from a specific user."""
        logger.info(f"Getting tweets for user: {username}")
        tweets = twitter_client.get_user_tweets(username, limit=10)
        return {
            "username": username,
            "tweets": tweets
        }

    @server.resource("/trends/{woeid}")
    def trends(woeid: int) -> Dict[str, Any]:
        """Get trending topics for a specific location."""
        logger.info(f"Getting trending topics for WOEID: {woeid}")
        topics = twitter_client.get_trending_topics(woeid=woeid)
        return {
            "woeid": woeid,
            "topics": topics
        }

    @server.resource("/tweet/{tweet_id}")
    def tweet_details(tweet_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific tweet."""
        logger.info(f"Getting details for tweet: {tweet_id}")
        tweet = twitter_client.get_tweet_details(tweet_id)
        return {
            "tweet_id": tweet_id,
            "tweet": tweet
        }

    @server.resource("/user/{username}/profile")
    def user_profile(username: str) -> Dict[str, Any]:
        """Get detailed information about a Twitter user."""
        logger.info(f"Getting profile for user: {username}")
        profile = twitter_client.get_user_profile(username)
        return {
            "username": username,
            "profile": profile
        }

    return server

# Create the FastAPI app for running on a different port
app = create_server().streamable_http_app() 
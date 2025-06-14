"""
Twitter API client implementation.
"""
import logging
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class TwitterClient:
    """Client for interacting with the Twitter API."""

    def __init__(self, bearer_token: str):
        """Initialize the Twitter client.
        
        Args:
            bearer_token: The Twitter API bearer token for authentication.
        """
        self.bearer_token = bearer_token
        self.base_url = "https://api.twitter.com/2"
        self.headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }

    def search_tweets(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search Twitter for tweets matching a query.
        
        Args:
            query: The search query.
            limit: Maximum number of results to return.
            
        Returns:
            A list of tweet results.
        """
        try:
            url = f"{self.base_url}/tweets/search/recent"
            params = {
                "query": query,
                "max_results": limit,
                "tweet.fields": "created_at,public_metrics,author_id,entities",
                "expansions": "author_id,referenced_tweets.id",
                "user.fields": "name,username,description"
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if "data" in data:
                for tweet in data["data"]:
                    # Get author info from includes
                    author = next(
                        (user for user in data.get("includes", {}).get("users", [])
                         if user["id"] == tweet["author_id"]),
                        None
                    )
                    
                    tweet_data = {
                        "id": tweet["id"],
                        "text": tweet["text"],
                        "created_at": tweet["created_at"],
                        "metrics": tweet.get("public_metrics", {}),
                        "author": {
                            "name": author["name"] if author else "Unknown",
                            "username": author["username"] if author else "Unknown",
                            "description": author.get("description", "")
                        } if author else None
                    }
                    results.append(tweet_data)
            
            return results
        except Exception as e:
            logger.error(f"Error searching Twitter: {e}")
            return []

    def get_user_tweets(self, username: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent tweets from a specific user.
        
        Args:
            username: The Twitter username.
            limit: Maximum number of tweets to return.
            
        Returns:
            A list of tweets.
        """
        try:
            # First get user ID
            user_url = f"{self.base_url}/users/by/username/{username}"
            user_response = requests.get(user_url, headers=self.headers)
            user_response.raise_for_status()
            user_data = user_response.json()
            
            if "data" not in user_data:
                return []
            
            user_id = user_data["data"]["id"]
            
            # Then get user's tweets
            tweets_url = f"{self.base_url}/users/{user_id}/tweets"
            params = {
                "max_results": limit,
                "tweet.fields": "created_at,public_metrics,entities",
                "expansions": "referenced_tweets.id"
            }
            
            tweets_response = requests.get(tweets_url, headers=self.headers, params=params)
            tweets_response.raise_for_status()
            tweets_data = tweets_response.json()
            
            results = []
            if "data" in tweets_data:
                for tweet in tweets_data["data"]:
                    tweet_data = {
                        "id": tweet["id"],
                        "text": tweet["text"],
                        "created_at": tweet["created_at"],
                        "metrics": tweet.get("public_metrics", {})
                    }
                    results.append(tweet_data)
            
            return results
        except Exception as e:
            logger.error(f"Error getting user tweets: {e}")
            return []

    def get_trending_topics(self, woeid: int = 1) -> List[Dict[str, Any]]:
        """Get trending topics for a specific location.
        
        Args:
            woeid: Where On Earth ID for the location (1 for worldwide).
            
        Returns:
            A list of trending topics.
        """
        try:
            url = f"{self.base_url}/trends/place"
            params = {"id": woeid}
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if isinstance(data, list) and len(data) > 0:
                trends = data[0].get("trends", [])
                for trend in trends:
                    trend_data = {
                        "name": trend["name"],
                        "url": trend["url"],
                        "tweet_volume": trend.get("tweet_volume"),
                        "query": trend["query"]
                    }
                    results.append(trend_data)
            
            return results
        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return []

    def get_tweet_details(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific tweet.
        
        Args:
            tweet_id: The ID of the tweet.
            
        Returns:
            Detailed tweet information or None if not found.
        """
        try:
            url = f"{self.base_url}/tweets/{tweet_id}"
            params = {
                "tweet.fields": "created_at,public_metrics,author_id,entities",
                "expansions": "author_id,referenced_tweets.id",
                "user.fields": "name,username,description"
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "data" in data:
                tweet = data["data"]
                author = next(
                    (user for user in data.get("includes", {}).get("users", [])
                     if user["id"] == tweet["author_id"]),
                    None
                )
                
                return {
                    "id": tweet["id"],
                    "text": tweet["text"],
                    "created_at": tweet["created_at"],
                    "metrics": tweet.get("public_metrics", {}),
                    "author": {
                        "name": author["name"] if author else "Unknown",
                        "username": author["username"] if author else "Unknown",
                        "description": author.get("description", "")
                    } if author else None,
                    "entities": tweet.get("entities", {})
                }
            
            return None
        except Exception as e:
            logger.error(f"Error getting tweet details: {e}")
            return None

    def get_user_profile(self, username: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a Twitter user.
        
        Args:
            username: The Twitter username.
            
        Returns:
            User profile information or None if not found.
        """
        try:
            url = f"{self.base_url}/users/by/username/{username}"
            params = {
                "user.fields": "created_at,description,public_metrics,profile_image_url,verified"
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "data" in data:
                user = data["data"]
                return {
                    "id": user["id"],
                    "name": user["name"],
                    "username": user["username"],
                    "description": user.get("description", ""),
                    "created_at": user.get("created_at"),
                    "metrics": user.get("public_metrics", {}),
                    "profile_image_url": user.get("profile_image_url"),
                    "verified": user.get("verified", False)
                }
            
            return None
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None 
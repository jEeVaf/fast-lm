"""Asynchronous HTTP request handler using httpx for fast-lm gateway.

This module provides async HTTP functionality for routing and managing
language model requests with support for connection pooling, retries,
and streaming responses.
"""

import logging
from typing import Any, Dict, Optional, AsyncIterator
import httpx

logger = logging.getLogger(__name__)


class AsyncHttpHandler:
    """Async HTTP handler for making requests to language model APIs."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        pool_limits: Optional[httpx.Limits] = None,
    ):
        """Initialize the async HTTP handler.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            pool_limits: Connection pool limits configuration
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.pool_limits = pool_limits or httpx.Limits(
            max_connections=100, max_keepalive_connections=20
        )
        self.client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.client = httpx.AsyncClient(
            limits=self.pool_limits, timeout=self.timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    async def get(
        self, url: str, headers: Optional[Dict[str, str]] = None, **kwargs
    ) -> httpx.Response:
        """Make an async GET request.

        Args:
            url: The URL to request
            headers: Optional request headers
            **kwargs: Additional arguments to pass to httpx

        Returns:
            Response object

        Raises:
            httpx.RequestError: If request fails after retries
        """
        if not self.client:
            raise RuntimeError("AsyncHttpHandler not initialized. Use 'async with' context manager.")
        
        return await self.client.get(url, headers=headers, **kwargs)

    async def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> httpx.Response:
        """Make an async POST request.

        Args:
            url: The URL to request
            data: Form data to send
            json: JSON data to send
            headers: Optional request headers
            **kwargs: Additional arguments to pass to httpx

        Returns:
            Response object

        Raises:
            httpx.RequestError: If request fails after retries
        """
        if not self.client:
            raise RuntimeError("AsyncHttpHandler not initialized. Use 'async with' context manager.")
        
        return await self.client.post(
            url, data=data, json=json, headers=headers, **kwargs
        )

    async def stream(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> AsyncIterator[bytes]:
        """Make an async streaming request.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: The URL to request
            headers: Optional request headers
            **kwargs: Additional arguments to pass to httpx

        Yields:
            Response chunks

        Raises:
            httpx.RequestError: If request fails
        """
        if not self.client:
            raise RuntimeError("AsyncHttpHandler not initialized. Use 'async with' context manager.")
        
        async with self.client.stream(method, url, headers=headers, **kwargs) as response:
            async for chunk in response.aiter_bytes():
                yield chunk

    async def close(self):
        """Close the underlying httpx client."""
        if self.client:
            await self.client.aclose()

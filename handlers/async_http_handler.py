
import asyncio
import os
import ssl
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import httpx
from aiohttp import ClientSession, TCPConnector
from httpx import USE_CLIENT_DEFAULT, AsyncHTTPTransport
from httpx._types import RequestFiles, VerifyTypes

# -----------------------------------------------------------------------------
# NOTE: The following imports/constants are assumed to exist in your broader 
# project scope (e.g., litellm). Ensure they are imported correctly in your file.
# -----------------------------------------------------------------------------
# import litellm
# from litellm.utils import HTTPHandler, track_llm_api_timing, LiteLLMLoggingObject
# from litellm.litellm_core_utils.logging_utils import _prepare_request_data_and_content, _raise_masked_async_error
# from litellm.secret_managers.main import str_to_bool
# from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
# from litellm._logging import verbose_logger
# Constants: AIOHTTP_KEEPALIVE_TIMEOUT, AIOHTTP_TTL_DNS_CACHE, AIOHTTP_NEEDS_CLEANUP_CLOSED, 
#            AIOHTTP_CONNECTOR_LIMIT, AIOHTTP_CONNECTOR_LIMIT_PER_HOST, _DEFAULT_TIMEOUT, etc.


class AsyncHTTPHandler:
    def __init__(
        self,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        event_hooks: Optional[Mapping[str, List[Callable[..., Any]]]] = None,
        concurrent_limit=None,  # Kept for backward compatibility, but ignored
        client_alias: Optional[str] = None,
        ssl_verify: Optional[VerifyTypes] = None,
        shared_session: Optional[ClientSession] = None,
    ):
        self.timeout = timeout
        self.event_hooks = event_hooks
        self.client_alias = client_alias
        self.client = self.create_client(
            timeout=timeout,
            event_hooks=event_hooks,
            ssl_verify=ssl_verify,
            shared_session=shared_session,
        )

    def create_client(
        self,
        timeout: Optional[Union[float, httpx.Timeout]],
        event_hooks: Optional[Mapping[str, List[Callable[..., Any]]]],
        ssl_verify: Optional[VerifyTypes] = None,
        shared_session: Optional[ClientSession] = None,
    ) -> httpx.AsyncClient:
        ssl_config = get_ssl_configuration(ssl_verify)
        cert = os.getenv("SSL_CERTIFICATE", litellm.ssl_certificate)

        if timeout is None:
            timeout = _DEFAULT_TIMEOUT

        transport = self._create_async_transport(
            ssl_context=ssl_config if isinstance(ssl_config, ssl.SSLContext) else None,
            ssl_verify=ssl_config if isinstance(ssl_config, bool) else None,
            shared_session=shared_session,
        )

        return httpx.AsyncClient(
            transport=transport,
            event_hooks=event_hooks,
            timeout=timeout,
            verify=ssl_config,
            cert=cert,
            headers=get_default_headers(),
            follow_redirects=True,
        )

    async def close(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    # =========================================================================
    # HTTP METHODS
    # =========================================================================

    async def get(
        self,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        follow_redirects: Optional[bool] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ):
        _follow_redirects = follow_redirects if follow_redirects is not None else USE_CLIENT_DEFAULT
        params = params or {}
        params.update(HTTPHandler.extract_query_params(url))

        return await self.client.get(
            url,
            params=params,
            headers=headers,
            follow_redirects=_follow_redirects,
            timeout=timeout if timeout is not None else USE_CLIENT_DEFAULT,
        )

    @track_llm_api_timing()
    async def post(
        self, url: str, data=None, json=None, params=None, headers=None, 
        timeout=None, stream=False, logging_obj: Optional[LiteLLMLoggingObject] = None, 
        files=None, content=None
    ):
        return await self._execute_request("POST", url, data, json, params, headers, timeout, stream, files, content)

    async def put(
        self, url: str, data=None, json=None, params=None, headers=None, 
        timeout=None, stream=False, content=None
    ):
        return await self._execute_request("PUT", url, data, json, params, headers, timeout, stream, content=content)

    async def patch(
        self, url: str, data=None, json=None, params=None, headers=None, 
        timeout=None, stream=False, content=None
    ):
        return await self._execute_request("PATCH", url, data, json, params, headers, timeout, stream, content=content)

    async def delete(
        self, url: str, data=None, json=None, params=None, headers=None, 
        timeout=None, stream=False, content=None
    ):
        return await self._execute_request("DELETE", url, data, json, params, headers, timeout, stream, content=content)

    # =========================================================================
    # CORE REQUEST & RETRY LOGIC
    # =========================================================================

    async def _execute_request(
        self,
        method: str,
        url: str,
        data: Optional[Union[dict, str, bytes]] = None,
        json: Optional[dict] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        stream: bool = False,
        files: Optional[RequestFiles] = None,
        content: Any = None,
    ) -> httpx.Response:
        """Centralized request handler with built-in retry and error mapping."""
        start_time = time.time()
        if timeout is None:
            timeout = self.timeout

        request_data, request_content = _prepare_request_data_and_content(data, content)

        try:
            req = self.client.build_request(
                method, url, data=request_data, json=json, params=params,
                headers=headers, timeout=timeout, files=files, content=request_content,
            )
            response = await self.client.send(req, stream=stream)
            response.raise_for_status()
            return response

        except (httpx.RemoteProtocolError, httpx.ConnectError):
            # Retry with a fresh client on connection errors
            new_client = self.create_client(timeout=timeout, event_hooks=self.event_hooks)
            try:
                return await self._single_connection_request(
                    method=method, url=url, client=new_client, data=data, json=json,
                    params=params, headers=headers, stream=stream, files=files, content=content,
                )
            finally:
                await new_client.aclose()

        except httpx.TimeoutException as e:
            time_delta = round(time.time() - start_time, 3)
            error_headers = {}
            if getattr(e, "response", None) is not None:
                error_headers = {f"response_headers-{k}": v for k, v in e.response.headers.items()}
            
            raise litellm.Timeout(
                message=f"Connection timed out. Timeout passed={timeout}, time taken={time_delta} seconds",
                model="default-model-name",
                llm_provider="litellm-httpx-handler",
                headers=error_headers,
            )
            
        except httpx.HTTPStatusError as e:
            await _raise_masked_async_error(e, stream)
            
        except Exception as e:
            raise e

    async def _single_connection_request(
        self, method: str, url: str, client: httpx.AsyncClient, 
        data=None, json=None, params=None, headers=None, 
        stream=False, files=None, content=None
    ):
        """Executes a request using a single-use client (used for connection retries)."""
        request_data, request_content = _prepare_request_data_and_content(data, content)
        req = client.build_request(
            method, url, data=request_data, json=json, params=params, 
            headers=headers, files=files, content=request_content
        )
        response = await client.send(req, stream=stream)
        response.raise_for_status()
        return response

    def __del__(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(self.close())
        except RuntimeError:
            pass

    # =========================================================================
    # TRANSPORT CREATION LOGIC
    # =========================================================================

    @staticmethod
    def _create_async_transport(
        ssl_context: Optional[ssl.SSLContext] = None,
        ssl_verify: Optional[bool] = None,
        shared_session: Optional[ClientSession] = None,
    ) -> Optional[Union["LiteLLMAiohttpTransport", AsyncHTTPTransport]]:
        if AsyncHTTPHandler._should_use_aiohttp_transport():
            return AsyncHTTPHandler._create_aiohttp_transport(
                ssl_context=ssl_context, ssl_verify=ssl_verify, shared_session=shared_session
            )
        return AsyncHTTPHandler._create_httpx_transport()

    @staticmethod
    def _should_use_aiohttp_transport() -> bool:
        from litellm.secret_managers.main import str_to_bool
        if litellm.disable_aiohttp_transport is True or str_to_bool(os.getenv("DISABLE_AIOHTTP_TRANSPORT", "False")) is True:
            return False
        verbose_logger.debug("Using AiohttpTransport...")
        return True

    @staticmethod
    def _get_ssl_connector_kwargs(
        ssl_verify: Optional[bool] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
    ) -> Dict[str, Any]:
        connector_kwargs: Dict[str, Any] = {
            "local_addr": ("0.0.0.0", 0) if litellm.force_ipv4 else None,
        }
        if ssl_context is not None:
            connector_kwargs["ssl"] = ssl_context
        elif ssl_verify is False:
            connector_kwargs["ssl"] = False
        return connector_kwargs

    @staticmethod
    def _create_aiohttp_transport(
        ssl_verify: Optional[bool] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
        shared_session: Optional[ClientSession] = None,
    ) -> "LiteLLMAiohttpTransport":
        from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
        from litellm.secret_managers.main import str_to_bool

        connector_kwargs = AsyncHTTPHandler._get_ssl_connector_kwargs(ssl_verify, ssl_context)
        
        trust_env = litellm.aiohttp_trust_env
        if str_to_bool(os.getenv("AIOHTTP_TRUST_ENV", "False")) is True:
            trust_env = True

        ssl_for_transport: Optional[Union[bool, ssl.SSLContext]] = ssl_context if ssl_context is not None else (False if ssl_verify is False else None)
        verbose_logger.debug("Creating AiohttpTransport...")

        if shared_session is not None and not shared_session.closed:
            verbose_logger.debug(f"SHARED SESSION: Reusing existing ClientSession (ID: {id(shared_session)})")
            return LiteLLMAiohttpTransport(client=shared_session, ssl_verify=ssl_for_transport, owns_session=False)

        verbose_logger.debug("NEW SESSION: Creating new ClientSession (no shared session provided)")
        transport_connector_kwargs = {
            "keepalive_timeout": AIOHTTP_KEEPALIVE_TIMEOUT,
            "ttl_dns_cache": AIOHTTP_TTL_DNS_CACHE,
            **connector_kwargs,
        }
        if AIOHTTP_NEEDS_CLEANUP_CLOSED:
            transport_connector_kwargs["enable_cleanup_closed"] = True
        if AIOHTTP_CONNECTOR_LIMIT > 0:
            transport_connector_kwargs["limit"] = AIOHTTP_CONNECTOR_LIMIT
        if AIOHTTP_CONNECTOR_LIMIT_PER_HOST > 0:
            transport_connector_kwargs["limit_per_host"] = AIOHTTP_CONNECTOR_LIMIT_PER_HOST

        socket_factory = _build_aiohttp_keepalive_socket_factory()
        if socket_factory is not None:
            transport_connector_kwargs["socket_factory"] = socket_factory

        return LiteLLMAiohttpTransport(
            client=lambda: ClientSession(connector=TCPConnector(**transport_connector_kwargs), trust_env=trust_env),
            ssl_verify=ssl_for_transport,
        )

    @staticmethod
    def _create_httpx_transport() -> Optional[AsyncHTTPTransport]:
        if litellm.force_ipv4:
            return AsyncHTTPTransport(local_address="0.0.0.0")
        return None
```

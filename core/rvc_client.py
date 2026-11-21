from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from websockets.asyncio.client import connect

_WEBSOCKET_LOGGER = logging.getLogger("astrbot_plugin_music_cover.rvc_websocket")
_WEBSOCKET_LOGGER.setLevel(logging.WARNING)


class RvcClient:
    """Call the legacy Gradio API exposed by RVC WebUI."""

    REQUIRED_ENDPOINTS = {
        "/infer_refresh",
        "/infer_change_voice",
        "/infer_convert",
        "/uvr_convert",
    }

    def __init__(self, base_url: str, timeout: int):
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"无效的 RVC WebUI 地址: {base_url}")
        path = f"{parsed.path.rstrip('/')}/"
        self.base_url = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        self.ws_url = urlunsplit(
            (ws_scheme, parsed.netloc, f"{path}queue/join", "", "")
        )
        self.timeout = timeout
        self.models: list[str] = []
        self.separator_models: list[str] = []
        self._fn_indices: dict[str, int] = {}
        self._http = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=min(timeout, 30),
            trust_env=False,
        )

    async def initialize(self) -> None:
        """Load endpoint metadata and refresh available RVC models.

        Raises:
            RuntimeError: If the WebUI does not expose the required endpoints or models.
        """
        response = await self._http.get("config")
        response.raise_for_status()
        config = response.json()
        dependencies = config.get("dependencies")
        components = config.get("components")
        if not isinstance(dependencies, list) or not isinstance(components, list):
            raise RuntimeError("RVC WebUI 返回了无效的 Gradio 配置")

        self._fn_indices = {
            f"/{api_name.lstrip('/')}": index
            for index, dependency in enumerate(dependencies)
            if isinstance(dependency, dict) and (api_name := dependency.get("api_name"))
        }
        missing = self.REQUIRED_ENDPOINTS - self._fn_indices.keys()
        if missing:
            raise RuntimeError(f"RVC WebUI 缺少接口: {', '.join(sorted(missing))}")

        refresh_result = await self.call("/infer_refresh")
        model_update = refresh_result[0] if isinstance(refresh_result, tuple) else None
        choices = (
            model_update.get("choices") if isinstance(model_update, dict) else None
        )
        self.models = [name for item in choices or [] if (name := str(item).strip())]
        if not self.models:
            raise RuntimeError("RVC WebUI 没有可用的音色模型")

        uvr_dependency = dependencies[self._fn_indices["/uvr_convert"]]
        input_ids = uvr_dependency.get("inputs") or []
        separator_component = next(
            (
                component
                for component in components
                if input_ids and component.get("id") == input_ids[0]
            ),
            None,
        )
        separator_choices = (
            separator_component.get("props", {}).get("choices", [])
            if isinstance(separator_component, dict)
            else []
        )
        self.separator_models = [
            name
            for item in separator_choices
            if (name := str(item).strip()) and "HP" in name.upper()
        ]
        if not self.separator_models:
            raise RuntimeError("RVC WebUI 没有可用的人声分离模型")

    async def call(self, api_name: str, *arguments: Any) -> Any:
        """Execute one named Gradio endpoint through its legacy queue protocol.

        Args:
            api_name: Named Gradio endpoint including the leading slash.
            *arguments: Values matching the endpoint's input component order.

        Returns:
            A scalar for one output or a tuple for multiple outputs.

        Raises:
            RuntimeError: If the endpoint is missing or reports a failed job.
            TimeoutError: If the operation exceeds the configured timeout.
        """
        fn_index = self._fn_indices.get(api_name)
        if fn_index is None:
            raise RuntimeError(f"RVC WebUI 缺少接口: {api_name}")
        session_hash = uuid.uuid4().hex

        async def exchange() -> Any:
            async with connect(
                self.ws_url,
                proxy=None,
                open_timeout=min(self.timeout, 10),
                close_timeout=5,
                max_size=1024 * 1024 * 1024,
                logger=_WEBSOCKET_LOGGER,
            ) as websocket:
                async for raw_message in websocket:
                    message = json.loads(raw_message)
                    event = message.get("msg")
                    if event == "send_hash":
                        await websocket.send(
                            json.dumps(
                                {"fn_index": fn_index, "session_hash": session_hash}
                            )
                        )
                    elif event == "send_data":
                        await websocket.send(
                            json.dumps(
                                {
                                    "data": list(arguments),
                                    "event_data": None,
                                    "fn_index": fn_index,
                                    "session_hash": session_hash,
                                }
                            )
                        )
                    elif event == "queue_full":
                        raise RuntimeError("RVC WebUI 推理队列已满")
                    elif event == "process_completed":
                        output = message.get("output") or {}
                        if not message.get("success", True) or output.get("error"):
                            raise RuntimeError(
                                str(output.get("error") or "RVC WebUI 推理失败")
                            )
                        data = output.get("data")
                        if not isinstance(data, list):
                            raise RuntimeError("RVC WebUI 返回了无效结果")
                        if len(data) == 1:
                            return data[0]
                        return tuple(data)
            raise RuntimeError("RVC WebUI 在返回结果前断开连接")

        return await asyncio.wait_for(exchange(), timeout=self.timeout)

    async def close(self) -> None:
        """Close the HTTP client used for endpoint discovery."""
        await self._http.aclose()

# llm.py
"""
LLM 调用层 —— 零依赖实现（仅用 urllib.request 和 json）
兼容任何 OpenAI API 格式的服务（OpenAI / Azure / 本地 Ollama 等）
"""
from curses import raw
import urllib.request
import urllib.error
import json
from typing import Any

class LLMClient:
    """
    OpenAI兼容的最小LLM客户端

    params:
        base_url: str API基础URL，例如 "https://api.openai.com/v1"
        api_key: str API密钥，例如 "sk-xxxx"
        model: str 模型名称，例如 "gpt-3.5-turbo"
        timeout: int 请求超时时间，默认60秒
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-3.5-turbo",
        timeout: int = 60,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _parse_response(self, raw: dict) -> dict:
        message = raw["choices"][0]["message"]
        finish_reason = raw["choices"][0].get("finish_reason", "stop")

        if finish_reason == "tool_calls" or message.get("tool_calls"):
            parsed_tool_calls = []
            for tc in (message.get("tool_calls") or []):
                fn = tc.get("function", {})
                try:
                    arguments = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    arguments = {"_raw": fn.get("arguments", "")}
                parsed_tool_calls.append({
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": arguments,
                })
            return {"type": "tool_call", "content": None, "tool_calls": parsed_tool_calls, "raw": raw}

        return {"type": "text", "content": message.get("content", ""), "tool_calls": [], "raw": raw}

    def _blocking_request(self, request: urllib.request.Request) -> dict:
        """
        阻塞式请求，返回 JSON 解析后的响应
        """
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw_bytes = response.read()
        raw = json.loads(raw_bytes.decode("utf-8"))
        return self._parse_response(raw)

    def _stream_request(self, request: urllib.request.Request) -> dict:
        """
        读取 SSE 流，实时打印内容，最终返回拼接后的完整响应字典。
        OpenAI 流式格式：每行以 "data: " 开头，最后一行为 "data: [DONE]"。
        """
        collected_text = []
        collected_tool_calls: dict[int, dict] = {}  # index -> tool_call dict

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or line == "data: [DONE]":
                    if line == "data: [DONE]":
                        break
                    continue
                if not line.startswith("data: "):
                    continue
                try:
                    chunk = json.loads(line[len("data: "):])
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})

                if "content" in delta and delta["content"]:
                    collected_text.append(delta["content"])
                    print(delta["content"], end="", flush=True)

                # 流式 tool_call 按 index 合并片段
                for tc_delta in delta.get("tool_calls", []):
                    idx = tc_delta["index"]
                    if idx not in collected_tool_calls:
                        collected_tool_calls[idx] = {
                            "id": "", "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    tc = collected_tool_calls[idx]
                    if tc_delta.get("id"):
                        tc["id"] += tc_delta["id"]
                    fn = tc_delta.get("function", {})
                    if fn.get("name"):
                        tc["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        tc["function"]["arguments"] += fn["arguments"]

        if collected_text:
            print()

        fake_raw: dict[str, Any] = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "".join(collected_text) or None,
                    "tool_calls": list(collected_tool_calls.values()) if collected_tool_calls else None,
                },
                "finish_reason": "tool_calls" if collected_tool_calls else "stop",
            }]
        }
        return self._parse_response(fake_raw)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict:
        """
        向 /chat/completions 发送请求，返回统一格式的响应字典。

        返回格式：
        {
          "type": "text" | "tool_call",
          "content": str,           # type=="text" 时的文本内容
          "tool_calls": [           # type=="tool_call" 时的工具调用列表
            {"id": str, "name": str, "arguments": dict}
          ],
          "raw": dict,              # 原始 API 响应
        }
        """

        payload = dict(str, Any) = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        url = self.base_url + "/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")

        try:
            if stream:
                return self._stream_request(request)
            else:
                return self._blocking_request(request)
        except urllib.error.HTTPError as exc:
            # 处理 HTTP 错误，例如重试或记录日志
            # 这里简单抛出异常
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API error: [{exc.code}] {error_body}") from exc


       

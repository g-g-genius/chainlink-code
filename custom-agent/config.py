import os
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("DASHSCOPE_API_KEY")
BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.5-plus")

def call_llm(messages, tools=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 4096,
    }
    
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    
    url = BASE_URL + "/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = json.loads(response.read().decode("utf-8"))
            return raw
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API error: [{exc.code}] {error_body}") from exc

def parse_response(response):
    message = response["choices"][0]["message"]
    finish_reason = response["choices"][0].get("finish_reason", "stop")
    
    if finish_reason == "tool_calls" or message.get("tool_calls"):
        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                arguments = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {"_raw": fn.get("arguments", "")}
            tool_calls.append({
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": arguments,
            })
        return {"type": "tool_call", "tool_calls": tool_calls, "message": message}
    
    return {"type": "text", "content": message.get("content", ""), "message": message}
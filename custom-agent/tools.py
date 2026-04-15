import subprocess
import os
import glob

DEFAULT_MAX_TOKENS = {
    "read_file": 8000,
    "run_command": 2000,
    "search_code": 3000,
    "list_directory": 1000,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取指定路径的文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "将内容写入指定路径的文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "文件内容"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "在文件中查找一段精确匹配的文本，替换为新文本。old_text 必须与文件中的内容完全一致。",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "old_text": {"type": "string", "description": "要被替换的原文，必须精确匹配"},
                    "new_text": {"type": "string", "description": "替换后的新内容"}
                },
                "required": ["path", "old_text", "new_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "在项目根目录执行 shell 命令，返回 stdout 和 stderr",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "在项目中搜索包含指定文本的文件和行号。返回每个匹配的文件路径、行号和该行内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "搜索的文本"},
                    "file_pattern": {"type": "string", "description": "限制搜索的文件类型，如 '*.py'", "default": ""}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "列出指定目录的文件和子目录结构（最多2层深度）",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径", "default": "."}
                },
                "required": []
            }
        }
    }
]

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * 3
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...内容过长已截断]"

def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return truncate_to_tokens(f.read(), DEFAULT_MAX_TOKENS["read_file"])
    except FileNotFoundError:
        return f"错误：文件 {path} 不存在"
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="gbk") as f:
                return truncate_to_tokens(f.read(), DEFAULT_MAX_TOKENS["read_file"])
        except Exception as e:
            return f"错误：无法解码文件 {path}: {str(e)}"
    except Exception as e:
        return f"错误：{str(e)}"

def write_file(path: str, content: str) -> str:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"已写入 {path}"
    except Exception as e:
        return f"错误：{str(e)}"

def edit_file(path: str, old_text: str, new_text: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return f"错误：文件 {path} 不存在"
    
    if old_text not in content:
        return f"错误：在 {path} 中未找到要替换的文本。请用 read_file 查看文件当前内容后重试。"
    
    count = content.count(old_text)
    if count > 1:
        return f"错误：在 {path} 中找到 {count} 处匹配。请提供更多上下文使匹配唯一。"
    
    content = content.replace(old_text, new_text, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已在 {path} 中完成替换"

def run_command(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True,
            capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace"
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr
        output += f"\n退出码: {result.returncode}"
        return truncate_to_tokens(output, DEFAULT_MAX_TOKENS["run_command"])
    except subprocess.TimeoutExpired:
        return "错误：命令执行超时（60秒）"
    except Exception as e:
        return f"错误：{str(e)}"

def search_code(pattern: str, file_pattern: str = "") -> str:
    results = []
    search_path = os.getcwd()
    
    if file_pattern:
        files = glob.glob(os.path.join(search_path, "**", file_pattern), recursive=True)
    else:
        files = glob.glob(os.path.join(search_path, "**", "*"), recursive=True)
    
    ignore_dirs = {"node_modules", "__pycache__", ".git", "venv", ".venv", "dist", "build"}
    
    for filepath in files:
        if any(ignore_dir in filepath for ignore_dir in ignore_dirs):
            continue
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    if pattern in line:
                        rel_path = os.path.relpath(filepath, search_path)
                        results.append(f"{rel_path}:{line_num}: {line.rstrip()}")
                        if len(results) >= 100:
                            break
        except Exception:
            continue
        if len(results) >= 100:
            break
    
    if results:
        return truncate_to_tokens("\n".join(results), DEFAULT_MAX_TOKENS["search_code"])
    return "未找到匹配结果"

def list_directory(path: str = ".") -> str:
    results = []
    ignore_dirs = {"node_modules", "__pycache__", ".git", "venv", ".venv", "dist", "build"}
    
    def walk_dir(current_path: str, depth: int):
        if depth > 2:
            return
        try:
            items = sorted(os.listdir(current_path))
        except PermissionError:
            return
        
        for item in items:
            full_path = os.path.join(current_path, item)
            rel_path = os.path.relpath(full_path, path)
            
            if item.startswith(".") or item in ignore_dirs:
                continue
            
            if os.path.isdir(full_path):
                if item not in ignore_dirs:
                    results.append(f"{rel_path}/")
                    walk_dir(full_path, depth + 1)
            else:
                results.append(rel_path)
            
            if len(results) >= 100:
                return
    
    walk_dir(path, 0)
    
    if results:
        return truncate_to_tokens("\n".join(results), DEFAULT_MAX_TOKENS["list_directory"])
    return f"目录 {path} 为空或不存在"

def execute_tool(name: str, args: dict) -> str:
    tool_handlers = {
        "read_file": lambda a: read_file(a["path"]),
        "write_file": lambda a: write_file(a["path"], a["content"]),
        "edit_file": lambda a: edit_file(a["path"], a["old_text"], a["new_text"]),
        "run_command": lambda a: run_command(a["command"]),
        "search_code": lambda a: search_code(a.get("pattern", ""), a.get("file_pattern", "")),
        "list_directory": lambda a: list_directory(a.get("path", ".")),
    }
    
    handler = tool_handlers.get(name)
    if handler:
        return handler(args)
    return f"未知工具: {name}"
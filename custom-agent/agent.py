import os
import json
from tools import tools, execute_tool
from config import call_llm, parse_response

def load_memory(project_root: str) -> str:
    memory_path = os.path.join(project_root, ".agent", "memory.md")
    if os.path.exists(memory_path):
        with open(memory_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def load_rules(project_root: str) -> str:
    rules_path = os.path.join(project_root, "AGENT.md")
    if os.path.exists(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def load_relevant_skills(task: str, skills_dir: str) -> str:
    if not os.path.exists(skills_dir):
        return ""
    
    loaded = []
    for skill_file in os.listdir(skills_dir):
        if not skill_file.endswith(".md"):
            continue
        skill_path = os.path.join(skills_dir, skill_file)
        try:
            with open(skill_path, "r", encoding="utf-8") as f:
                content = f.read()
            trigger_keywords = extract_trigger_keywords(content)
            if any(kw.lower() in task.lower() for kw in trigger_keywords):
                loaded.append(content)
        except Exception:
            continue
    
    return "\n\n---\n\n".join(loaded)

def extract_trigger_keywords(content: str) -> list:
    keywords = []
    in_trigger_section = False
    for line in content.split("\n"):
        if line.strip().startswith("## 触发条件"):
            in_trigger_section = True
            continue
        if in_trigger_section:
            if line.strip().startswith("##"):
                break
            if line.strip().startswith("- ") or line.strip().startswith("* "):
                keywords.append(line.strip()[2:].strip())
    return keywords

def run_agent(task: str, project_root: str = ".", max_steps: int = 25):
    rules = load_rules(project_root)
    skills = load_relevant_skills(task, os.path.join(project_root, ".agent", "skills"))
    memory = load_memory(project_root)
    
    system_prompt = f"""你是一个在本地代码仓库中工作的开发 Agent。
你可以读文件、写文件、执行命令。

## 项目规则
{rules}

## 相关技能
{skills}

## 项目记忆
{memory}

## 工作方式
1. 先理解任务，读取相关代码
2. 制定修改计划
3. 实施修改
4. 运行测试验证
5. 如果测试失败，分析原因并修正
6. 确认完成后，用普通文本回复总结你做了什么

遇到不确定的情况就停下来说明原因，不要猜。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task}
    ]

    for step in range(max_steps):
        print(f"\n--- 步骤 {step + 1} ---")
        
        response = call_llm(messages, tools)
        parsed = parse_response(response)
        
        if parsed["type"] == "text":
            print(f"\n任务完成：{parsed['content']}")
            return parsed["content"]
        
        messages.append(parsed["message"])
        
        tool_results = []
        for tc in parsed["tool_calls"]:
            print(f"  调用工具: {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)[:100]})")
            result = execute_tool(tc["name"], tc["arguments"])
            print(f"  结果: {result[:200]}...")
            tool_results.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result
            })
        
        messages.append({"role": "user", "content": json.dumps(tool_results, ensure_ascii=False)})

    print("\n达到最大步数限制，强制终止")
    return None
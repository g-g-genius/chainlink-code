#该模块提供了与操作系统交互的核心功能，用于执行文件、目录操作，管理进程及访问环境变量等
import os
#该模块允许你生成新的进程、连接到它们的输入/输出/错误管道,并获得它们的返回码,它让你可以从 Python 脚本中运行外部命令或程序
import subprocess
import json

# 读取 .env 文件中的键值对，把它们加载到操作系统的环境变量（os.environ）中，以便后续代码通过 os.getenv() 能访问到这些配置
from dotenv import load_dotenv
# 用于调用 Ollama (OpenAI 兼容接口) 的库
from openai import OpenAI

# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量中读取，如果读取不到则使用第二个参数作为默认值
base_url = os.getenv("CHIANLINK_BASE_URL")
api_key = os.getenv("CHIANLINK_API_KEY")
MODEL = os.getenv("MODEL_NAME")

# 初始化客户端
client = OpenAI(
    base_url = base_url,
    api_key = api_key,
)


SYSTEM_PROMPT = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

TOOLS = [{
    'name': 'bash',
    'description': 'Run a shell command.',
    'input_schema': {
        'type': 'object',
        'properties': {
            'command': {
                'type': 'string'
            }
        },
    }
}]

# 定义函数 run_bash，接收一个字符串类型的 command（要执行的命令）
# 返回值类型为字符串 (-> str)
def run_bash(command: str) -> str:
    # 1. 安全过滤：定义一个包含危险关键词的列表
    # 防止执行如“删除根目录”、“关机”或“静默覆盖文件”等破坏性操作
    dangerous = ['rm -rf /', 'sudo', 'shutdown', 'reboot', '> /dev/']
    # 2. 风险检查：遍历危险列表，如果用户输入的命令中包含任何一个危险词
    if any(d in command for d in dangerous):
        return 'Error: Dangerous command blocked'
    # 3. 尝试执行命令
    try:
        # 使用 subprocess.run 发起系统调用
        # shell=True: 允许通过 shell 解释器执行（支持管道、通配符等）
        # cwd=os.getcwd(): 在当前 Python 脚本运行的工作目录下执行
        # capture_output=True: 抓取命令的标准输出(stdout)和错误输出(stderr)
        # text=True: 将输出结果以字符串形式返回（而非字节流）
        # timeout=120: 设置硬性超时限制为 120 秒，防止命令死循环或阻塞
        r = subprocess.run(command,
                           shell = True,
                           cwd = os.getcwd(),
                           capture_output = True,
                           text = True,
                           timeout = 120)
        # 4. 结果处理：将标准输出和错误输出合并，并去除首尾多余的空格/换行
        out = (r.stdout + r.stderr).strip()
        # 5. 返回输出：
        # 如果有输出内容，则返回前 50000 个字符（防止数据量过大撑爆模型上下文）
        # 如果没有任何输出，则返回字符串 '(no output)'
        return out[:50000] if out else '(no output)'
    except subprocess.TimeoutExpired:
        return 'Error: Timeout (120s)'

def agent_loop(messages: list):
    while True:
        # model	        str	 必填 指定使用的模型名称（如 qwen2.5）。
        # messages	    list 必填 对话历史列表，包含 role 和 content。
        # max_tokens	int	     最大生成长度。限制模型输出的 Token 数量。
        # temperature	float	 采样温度。控制随机性。0 趋于确定，1 趋于发散。
        # stream	    bool	 流式传输。设置为 True 时，像打字机一样逐字返回。
        # tools	        list	 工具定义。定义模型可以调用的函数列表。
        # tool_choice	str/obj	 控制模型如何使用工具（如 "auto", "none"）。
        # stop	        str/list 停止符。模型遇到这些字符时会提前停止生成。
        response = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                        tools=TOOLS,
                        tool_choice="auto",
                        max_tokens=2048,
                        temperature=0.1
                    )
        # 1. 获取模型返回的消息对象
        # 在 OpenAI 格式中，消息在 response.choices[0].message
        msg = response.choices[0].message

        # 2. 将模型回复存入对话历史
        # 注意：必须把整个消息对象存进去，因为它可能包含 tool_calls
        messages.append(msg)

        # 3. 检查模型是否想要调用工具
        # OpenAI 格式中使用 tool_calls 列表，而不是 stop_reason
        if not msg.tool_calls:
            return  # 如果没有工具调用，说明任务完成，退出循环

        # 4. 处理工具调用
        for tool_call in msg.tool_calls:
            # 确认调用的函数名是否为 'bash'
            if tool_call.function.name == "bash":
                # 解析参数：模型返回的是 JSON 字符串，需要转成 Python 字典
                args = json.loads(tool_call.function.arguments)
                command = args.get("command")

                # 打印并执行命令
                print(f"\033[33m$ {command}\033[0m")
                output = run_bash(command)
                print(output[:200]) # 预览部分结果

                # 5. 将执行结果反馈给模型
                # 注意：role 必须是 'tool'，且必须提供 tool_call_id
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "bash",
                    "content": output
                })

if __name__ == "__main__":
    history = [] # 存储对话上下文
    while True:
        try:
            # 获取用户输入
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        # 1. 记录用户的问题
        history.append({"role": "user", "content": query})
        # 2. 进入 Agent 递归循环（处理 思考 -> 执行 -> 再思考 的逻辑）
        agent_loop(history)
        # 3. 打印模型最终给出的回答
        # 在 OpenAI 格式中，最后的回答在 history 最后一个元素的 .content 里
        final_msg = history[-1]
        # 如果 content 是对象属性（来自 API 返回），或者是字典（来自我们手动 append）
        if hasattr(final_msg, 'content'):
            content = final_msg.content
        else:
            content = final_msg.get('content')

        if content:
            print(f"\n{content}\n")

import sys
from agent import run_agent

sys.stdout.reconfigure(encoding='utf-8')

if __name__ == "__main__":
    print("开发 Agent 已启动")
    print("输入任务描述，Agent 将自动完成代码修改、测试等工作")
    print("输入 'quit' 或 'exit' 退出")
    print("-" * 50)
    
    while True:
        try:
            task = input("\n请输入任务: ").strip()
            if not task:
                continue
            if task.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            run_agent(task)
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"错误: {e}")
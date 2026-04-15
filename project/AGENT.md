# 项目规则

## 技术栈
- Python 3.11, FastAPI, SQLAlchemy, Pydantic v2
- 测试框架: pytest, 运行命令: `pytest tests/ -v`
- 代码格式: `ruff check --fix .`

## 目录结构
- src/api/ — FastAPI 路由
- src/models/ — SQLAlchemy 模型
- src/services/ — 业务逻辑
- src/utils/ — 工具函数
- tests/ — 测试文件，命名 test_<模块名>.py

## 代码约定
- 所有函数必须有 type hints
- 所有公共函数必须有 docstring
- import 排序: 标准库 → 第三方 → 项目内部
- 错误处理用自定义异常，定义在 src/exceptions.py

## 行为边界
- 不要修改 alembic/ 下的迁移文件
- 不要直接修改 .env 文件
- 数据库 schema 变更需要新建迁移，不要手改现有迁移
- 遇到涉及支付逻辑的修改，停下来说明情况
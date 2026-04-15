# 项目记忆

## 架构决策
- 2024-11: 从 REST 迁移到 GraphQL 的计划已取消，继续用 REST
- 2025-01: 选用 Celery 做异步任务，broker 用 Redis
- 2025-03: 认证从自建 JWT 迁移到 Auth0

## 已知坑
- UserService.get_by_email() 在邮箱大小写不一致时会返回 None，已有 issue #234 跟踪
- tests/test_payment.py 中有两个跳过的测试，原因是 Stripe sandbox 限流，不要试图修复它们
- docker-compose 启动顺序有时导致 db 还没 ready 就连接失败，重试一次就行

## 最近的变更
- 2025-03-20: 重构了 src/services/order.py 的退款逻辑，原来的三个方法合并成了一个
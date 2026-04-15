import re


def validate_email(email: str) -> bool:
    """
    使用正则表达式检查邮箱格式是否合法。
    
    参数:
        email: 待验证的邮箱字符串
        
    返回:
        bool: 邮箱格式合法返回 True，否则返回 False
    """
    # 邮箱正则表达式模式
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

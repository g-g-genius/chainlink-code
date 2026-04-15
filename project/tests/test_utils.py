import sys
sys.path.insert(0, 'src')

import unittest
from utils import validate_email


class TestValidateEmail(unittest.TestCase):
    
    def test_valid_email(self):
        """测试合法邮箱格式"""
        self.assertTrue(validate_email("test@example.com"))
        self.assertTrue(validate_email("user.name@domain.org"))
        self.assertTrue(validate_email("test+tag@sub.domain.co.uk"))
    
    def test_missing_at_symbol(self):
        """测试缺少@符号的情况"""
        self.assertFalse(validate_email("testexample.com"))
        self.assertFalse(validate_email("user.name.domain.org"))
    
    def test_missing_domain(self):
        """测试缺少域名的情况"""
        self.assertFalse(validate_email("test@"))
        self.assertFalse(validate_email("user@.com"))
        self.assertFalse(validate_email("test@domain"))


if __name__ == '__main__':
    unittest.main()

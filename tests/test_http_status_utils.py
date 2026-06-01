import unittest

from app.shared.utils.http_status_utils import (
    is_http_ok,
    is_http_success,
    is_retryable_server_error,
)


class HttpStatusUtilsTest(unittest.TestCase):
    def test_is_http_ok_only_accepts_200(self):
        self.assertTrue(is_http_ok(200))
        self.assertFalse(is_http_ok(201))

    def test_is_http_success_accepts_2xx_only(self):
        for status_code in [200, 201, 204]:
            self.assertTrue(is_http_success(status_code))

        for status_code in [300, 400, 500]:
            self.assertFalse(is_http_success(status_code))

    def test_is_retryable_server_error_accepts_5xx_only(self):
        self.assertTrue(is_retryable_server_error(500))
        self.assertTrue(is_retryable_server_error(599))

        for status_code in [400, 499, 600]:
            self.assertFalse(is_retryable_server_error(status_code))


if __name__ == "__main__":
    unittest.main()

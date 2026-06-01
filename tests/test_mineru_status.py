import unittest

from app.rag.import_.mineru_status import (
    MinerUExtractState,
    get_error_message,
    is_running_state,
    is_success_code,
    parse_extract_state,
)


class MinerUStatusTest(unittest.TestCase):
    def test_is_success_code_accepts_int_and_string_zero(self):
        self.assertTrue(is_success_code(0))
        self.assertTrue(is_success_code("0"))

    def test_get_error_message_uses_known_error_codes(self):
        token_error = get_error_message("A0202")
        parse_error = get_error_message(-60010)

        self.assertIn("Token 错误", token_error)
        self.assertIn("解析失败", parse_error)

    def test_running_states_are_explicitly_allowed(self):
        for state_value in ["waiting-file", "pending", "running", "converting"]:
            state = parse_extract_state(state_value)
            self.assertTrue(is_running_state(state))

    def test_done_failed_and_unknown_states_are_explicit(self):
        self.assertEqual(parse_extract_state("done"), MinerUExtractState.DONE)
        self.assertEqual(parse_extract_state("failed"), MinerUExtractState.FAILED)
        self.assertFalse(is_running_state(MinerUExtractState.DONE))
        self.assertFalse(is_running_state(MinerUExtractState.FAILED))

        with self.assertRaisesRegex(ValueError, "未知 MinerU 解析状态"):
            parse_extract_state("unknown")


if __name__ == "__main__":
    unittest.main()

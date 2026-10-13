import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch
from rubric_rl import verl_reward as reward


class AuxiliaryJudgeTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(
            os.environ,
            {
                "HEALTHBENCH_JUDGE_API_KEY": "primary-test-key",
                "HEALTHBENCH_JUDGE_MODEL": "primary-test-model",
                "HEALTHBENCH_JUDGE_MAX_TOKENS": "128",
                "AUXILIARY_JUDGE_ENABLED": "1",
                "AUXILIARY_JUDGE_API_BASE": "https://aux.invalid/v1",
                "AUXILIARY_JUDGE_API_KEY": "aux-test-key",
                "AUXILIARY_JUDGE_MODEL": "aux-test-model",
            },
        )
        base = patch.object(
            reward, "CANONICAL_JUDGE_API_BASE", "https://primary.invalid/v1"
        )
        base.start()
        self.addCleanup(base.stop)
        for name, value in {
            "CANONICAL_JUDGE_MODEL": "primary-test-model",
            "_JUDGE_BASE_ALLOWED": frozenset({"https://primary.invalid/v1"}),
        }.items():
            setting = patch.object(reward, name, value)
            setting.start()
            self.addCleanup(setting.stop)
        self.env.start()
        self.addCleanup(self.env.stop)
        reward._AUXILIARY_COOLDOWN_UNTIL = 0.0
        self.requests = []

    @staticmethod
    def reply(content='{"1":"PRESENT"}'):
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": content}}
                    ]
                }
            ).encode()
        )

    def score(self):
        return reward._request_judge_once(
            "frozen prompt", rubric_count=1, base_url=reward.CANONICAL_JUDGE_API_BASE
        )

    def test_auxiliary_uses_its_own_key_model_and_preserves_prompt(self):

        def send(request, **kwargs):
            self.requests.append(request)
            return self.reply()

        with patch.object(reward.urllib.request, "urlopen", side_effect=send):
            self.assertEqual(self.score(), '{"1":"PRESENT"}')
        request = self.requests[0]
        self.assertEqual(request.full_url, "https://aux.invalid/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer aux-test-key")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "aux-test-model")
        self.assertEqual(payload["messages"][0]["content"], "frozen prompt")
        self.assertEqual(payload["max_tokens"], 512)

    def test_http_failure_falls_back_with_original_credentials_and_cools_down(self):

        def send(request, **kwargs):
            self.requests.append(request)
            if request.full_url.startswith("https://aux.invalid"):
                raise urllib.error.HTTPError(
                    request.full_url, 429, "rate limit", {}, None
                )
            return self.reply()

        with patch.object(reward.urllib.request, "urlopen", side_effect=send):
            self.score()
            self.score()
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(
            self.requests[1].get_header("Authorization"), "Bearer primary-test-key"
        )
        self.assertEqual(
            json.loads(self.requests[1].data)["model"], "primary-test-model"
        )
        self.assertTrue(
            self.requests[2].full_url.startswith(reward.CANONICAL_JUDGE_API_BASE)
        )

    def test_invalid_auxiliary_json_falls_back_instead_of_becoming_zero_scores(self):

        def send(request, **kwargs):
            self.requests.append(request)
            return self.reply(
                "{}"
                if request.full_url.startswith("https://aux.invalid")
                else '{"1":"PRESENT"}'
            )

        with patch.object(reward.urllib.request, "urlopen", side_effect=send):
            self.assertEqual(self.score(), '{"1":"PRESENT"}')
        self.assertEqual(len(self.requests), 2)

    def test_disabled_preserves_original_route(self):
        with (
            patch.dict(os.environ, {"AUXILIARY_JUDGE_ENABLED": "0"}),
            patch.object(
                reward.urllib.request, "urlopen", return_value=self.reply()
            ) as send,
        ):
            self.score()
        self.assertEqual(
            send.call_args.args[0].full_url,
            reward.CANONICAL_JUDGE_API_BASE + "/chat/completions",
        )


if __name__ == "__main__":
    unittest.main()

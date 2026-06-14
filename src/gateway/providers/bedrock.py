"""AWS Bedrock provider using the Converse API.

Credentials and region are intentionally not configured here. boto3 resolves
them through the standard AWS chain (AWS_PROFILE, AWS_REGION, instance
roles, etc.). This keeps the repo environment-agnostic and avoids leaking
local profile/account names into code.
"""

from __future__ import annotations

from .base import Completion, ProviderError


class BedrockProvider:
    name = "bedrock"

    def __init__(self, model: str, client=None):
        if not model:
            raise ValueError("model id is required")
        self._model = model
        if client is None:
            import boto3  # lazy import so tests without bedrock do not need credentials

            client = boto3.client("bedrock-runtime")
        self._client = client

    def complete(self, messages: list[dict]) -> Completion:
        if not messages:
            raise ProviderError("no messages provided")

        try:
            response = self._client.converse(
                modelId=self._model,
                messages=[self._to_bedrock_message(message) for message in messages
                          if message.get("role") != "system"],
                system=[self._to_system_message(message) for message in messages
                        if message.get("role") == "system"],
                inferenceConfig={"maxTokens": 512, "temperature": 0.0},
            )
        except Exception as exc:  # boto3 raises a family of service/client errors
            raise ProviderError(f"bedrock call failed: {exc}") from exc

        try:
            text = response["output"]["message"]["content"][0]["text"]
            usage = response["usage"]
            input_tokens = usage["inputTokens"]
            output_tokens = usage["outputTokens"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"bedrock response missing expected fields: {response!r}") from exc

        return Completion(
            text=text,
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    @staticmethod
    def _to_bedrock_message(message: dict) -> dict:
        return {
            "role": BedrockProvider._to_bedrock_role(message),
            "content": [{"text": message["content"]}],
        }

    @staticmethod
    def _to_system_message(message: dict) -> dict:
        return {"text": message["content"]}

    @staticmethod
    def _to_bedrock_role(message: dict) -> str:
        role = message.get("role")
        if role in ("user", "assistant"):
            return role
        # Converse supports system separately; other roles are normalized to user
        # rather than dropped, so the model still sees the content.
        return "user"

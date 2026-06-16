"""Build configured gateway components from Settings."""

from __future__ import annotations

from .config import Settings
from .ledger import Ledger
from .pricing import PricingTable
from .providers import EchoProvider, Provider
from .router import Router


def build_providers(settings: Settings) -> list[Provider]:
    providers: list[Provider] = []
    for name in settings.providers:
        if name == "echo":
            providers.append(EchoProvider(model=settings.echo_model))
        elif name == "bedrock":
            from .providers.bedrock import BedrockProvider

            providers.append(BedrockProvider(model=settings.bedrock_model))
        elif name == "openai":
            from .providers.openai import OpenAIProvider

            providers.append(OpenAIProvider(model=settings.openai_model))
        elif name == "gemini":
            from .providers.gemini import GeminiProvider

            providers.append(GeminiProvider(model=settings.gemini_model))
    return providers


def build_router(settings: Settings, ledger: Ledger | None = None) -> Router:
    pricing = PricingTable.from_file(settings.pricing_file)
    active_ledger = ledger or Ledger(settings.ledger_path)
    return Router(build_providers(settings), pricing, active_ledger)

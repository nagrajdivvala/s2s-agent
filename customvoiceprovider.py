import os
from openai import AsyncAzureOpenAI
from agents import set_default_openai_client
from agents.voice.models.openai_model_provider import OpenAIVoiceModelProvider
from agents.voice.models.openai_stt import OpenAISTTModel
from agents.voice.models.openai_tts import OpenAITTSModel

class CustomVoiceProvider(OpenAIVoiceModelProvider):
    """
    A drop‑in replacement for OpenAIVoiceModelProvider that lets you
    specify separate deployments for chat, STT and TTS—and routes
    everything through your Azure Private Link.
    """
    def __init__(
        self,
        *,
        azure_endpoint: str,
        api_key: str,
        api_version: str,
        websocket_base_url: str,
        chat_deployment: str,
        stt_deployment: str,
        tts_deployment: str,
    ):
        # 1. Build an AsyncAzureOpenAI client that knows:
        #    • your HTTP endpoint & api-version
        #    • your default chat deployment
        #    • your Private‑Link WebSocket base URL
        client = AsyncAzureOpenAI(
            azure_endpoint     = azure_endpoint,
            api_key            = api_key,
            api_version        = api_version,
            websocket_base_url = websocket_base_url,
            azure_deployment   = chat_deployment,    # for text/chat
        )

        # 2. Make it the default for the Agent SDK
        set_default_openai_client(client)

        # 3. Initialize the base provider with our client
        super().__init__(openai_client=client)

        # 4. Remember our STT/TTS deployment names
        self._stt_deployment = stt_deployment
        self._tts_deployment = tts_deployment

    def get_stt_model(self, model_name: str | None = None):
        # Ignore the passed model_name: always use our STT deployment
        return OpenAISTTModel(self._stt_deployment, self._get_client())

    def get_tts_model(self, model_name: str | None = None):
        # Ignore the passed model_name: always use our TTS deployment
        return OpenAITTSModel(self._tts_deployment, self._get_client())

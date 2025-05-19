# my_app/voice_provider.py

from openai import AsyncAzureOpenAI, AsyncOpenAI
from agents import set_default_openai_client
from agents.voice.models.openai_model_provider import OpenAIVoiceModelProvider
from agents.voice.models.openai_stt import OpenAISTTModel
from agents.voice.models.openai_tts import OpenAITTSModel

class CustomVoiceProvider(OpenAIVoiceModelProvider):
    """
    A drop‑in replacement for OpenAIVoiceModelProvider that:
     • Lets you pass in an existing OpenAI client (AsyncAzureOpenAI or AsyncOpenAI)
     • Or, if you prefer, builds one from endpoint/api_key/api_version
     • Uses separate deployments for chat, STT and TTS
    """
    def __init__(
        self,
        *,
        openai_client: AsyncAzureOpenAI | AsyncOpenAI | None = None,
        azure_endpoint: str | None    = None,
        api_key: str | None           = None,
        api_version: str | None       = None,
        websocket_base_url: str | None= None,
        chat_deployment: str,
        stt_deployment: str,
        tts_deployment: str,
    ):
        # 1. Choose your client: use the passed one, or build a new AsyncAzureOpenAI
        if openai_client is None:
            if not (azure_endpoint and api_key and api_version):
                raise ValueError("Either openai_client or (azure_endpoint, api_key, api_version) must be provided")
            openai_client = AsyncAzureOpenAI(
                azure_endpoint     = azure_endpoint,
                api_key            = api_key,
                api_version        = api_version,
                websocket_base_url = websocket_base_url,
                azure_deployment   = chat_deployment,    # used for text/chat
            )

        # 2. Make it the default for the Agents SDK
        set_default_openai_client(openai_client)

        # 3. Initialize base provider
        super().__init__(openai_client=openai_client)

        # 4. Store deployments
        self._stt_deployment  = stt_deployment
        self._tts_deployment  = tts_deployment

    def get_stt_model(self, model_name: str | None = None):
        # Force our STT deployment
        return OpenAISTTModel(self._stt_deployment, self._get_client())

    def get_tts_model(self, model_name: str | None = None):
        # Force our TTS deployment
        return OpenAITTSModel(self._tts_deployment, self._get_client())

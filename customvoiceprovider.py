# my_app/voice_provider.py
import os
import websockets

from openai import AsyncAzureOpenAI
from agents import set_default_openai_client
from agents.voice.models.openai_model_provider import OpenAIVoiceModelProvider
from agents.voice.models.openai_stt import OpenAISTTTranscriptionSession, OpenAISTTModel
from agents.voice.models.openai_tts import OpenAITTSModel

class CustomVoiceProvider(OpenAIVoiceModelProvider):
    def __init__(
        self,
        *,
        azure_endpoint: str,
        api_key: str,
        api_version: str,
        chat_deployment: str,
        stt_deployment: str,
        tts_deployment: str,
    ):
        ws_base = azure_endpoint.replace("https://", "wss://")
        client = AsyncAzureOpenAI(
            azure_endpoint     = azure_endpoint,
            api_key            = api_key,
            api_version        = api_version,
            websocket_base_url = f"{ws_base}/openai",
            azure_deployment   = chat_deployment,
        )
        set_default_openai_client(client)
        super().__init__(openai_client=client)

        # store your deployments
        self._stt_deployment = stt_deployment
        self._tts_deployment = tts_deployment

        # patch STT
        def _stt_ws(self):
            uri = (
                f"{self._client.websocket_base_url}/realtime"
                f"?api-version={self._client.api_version}"
                f"&deployment={self._model}"
            )
            return websockets.connect(uri, extra_headers=[("api-key", api_key)])
        OpenAISTTTranscriptionSession._process_websocket_connection = _stt_ws

        # patch TTS
        def _tts_ws(self):
            uri = (
                f"{self._client.websocket_base_url}/realtime"
                f"?api-version={self._client.api_version}"
                f"&deployment={self._model}"
            )
            return websockets.connect(uri, extra_headers=[("api-key", api_key)])
        OpenAITTSModel._process_websocket_connection = _tts_ws

    def get_stt_model(self, model_name=None):
        return OpenAISTTModel(self._stt_deployment, self._get_client())

    def get_tts_model(self, model_name=None):
        return OpenAITTSModel(self._tts_deployment, self._get_client())

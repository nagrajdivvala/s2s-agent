import json
import os
import logging
import asyncio
import base64
import numpy as np
import uvicorn

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
# OpenAI Agents SDK imports
from agents import Agent, function_tool, enable_verbose_stdout_logging
from agents.voice import (
    StreamedAudioInput,
    SingleAgentVoiceWorkflow,
    VoicePipeline,
    TTSModelSettings,
    STTModelSettings,
    VoicePipelineConfig,
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from dotenv import load_dotenv
load_dotenv('.env')

# Enable verbose logging
enable_verbose_stdout_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

ws_url = os.getenv("WS_URL", "wss://your-public-url/ws")

# Azure embeddings & search setup
azureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model="text-embedding-3-small",
)
azureSearch = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_AI_SEARCH_KEY"),
    index_name="vector-1727345921686",
    embedding_function=azureOpenAIEmbeddings.embed_query,
)

# Sample tools
@function_tool
def get_weather(city: str) -> str:
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {np.random.choice(choices)}."

# Agent definitions omitted for brevity...
agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions(
        "Start by greeting the user. If GLP1 questions, hand off to glp1_agent."
    ),
    model="gpt-4o-mini",
)

# Utility to serialize numpy audio to ACS format
def serialize_audio_to_acs(audio_np: np.ndarray, subscription_id: str) -> dict:
    payload = base64.b64encode(audio_np.tobytes()).decode("utf-8")
    return {
        "kind": "AudioData",
        "audioData": {"data": payload, "encoding": "base64", "sampleRate": 24000},
        "streamSid": subscription_id,
    }

# Forward pipeline events to ACS, with client-side barge-in support
async def send_pipeline_events(websocket: WebSocket, subscription: dict, pipeline_result, state: dict):
    async for event in pipeline_result.stream():
        if event.type == "voice_stream_event_audio":
            state["is_agent_speaking"] = True
            sid = subscription.get("id", "unknown")
            packet = serialize_audio_to_acs(event.data, sid)
            await websocket.send_json(packet)

        elif event.type == "voice_stream_event_lifecycle":
            # Detect when user starts speaking mid-response
            if event.event == "turn_started" and state.get("is_agent_speaking"):
                logger.info("Barge-in detected: cancelling TTS and truncating response")
                # Cancel in-flight TTS/text generation
                if pipeline_result.text_generation_task and not pipeline_result.text_generation_task.done():
                    pipeline_result.text_generation_task.cancel()
                if pipeline_result._dispatcher_task and not pipeline_result._dispatcher_task.done():
                    pipeline_result._dispatcher_task.cancel()
                # Truncate partial assistant message (if supported)
                try:
                    await pipeline_result.truncate_conversation_item(event.item_id)
                except Exception:
                    logger.warning("truncate_conversation_item unavailable, ensure SDK version supports it")
                state["is_agent_speaking"] = False

            elif event.event == "turn_ended":
                state["is_agent_speaking"] = False

@app.post("/start_call")
async def start_call():
    return {"websocketUrl": ws_url}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")
    state = {"is_agent_speaking": False}
    subscription = {"id": None}
    streamed_input = StreamedAudioInput()

    # TTS & STT settings
    custom_tts = TTSModelSettings(
        instructions="Friendly guide voice.", voice="shimmer"
    )
    custom_stt = STTModelSettings(
        turn_detection={
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 200,
            # no interrupt_response here, using client-side cancel
        }
    )
    pipeline_config = VoicePipelineConfig(tts_settings=custom_tts, stt_settings=custom_stt)
    pipeline = await VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(agent), config=pipeline_config
    ).run(streamed_input)

    send_task = asyncio.create_task(
        send_pipeline_events(websocket, subscription, pipeline, state)
    )

    try:
        async for message in websocket.iter_text():
            packet = json.loads(message)
            kind = packet.get("kind")
            if kind == "AudioMetadata":
                subscription_id = packet.get("streamSid") or packet.get("audioMetadata", {}).get("subscriptionId")
                if subscription_id:
                    subscription["id"] = subscription_id
            elif kind == "AudioData":
                audio = base64.b64decode(packet["audioData"]["data"])
                audio_np = np.frombuffer(audio, dtype=np.int16)
                await streamed_input.add_audio(audio_np)
    finally:
        send_task.cancel()
        logger.info("WebSocket closed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)

import json
import os
import logging
import asyncio
import base64
import numpy as np
import uvicorn

from fastapi import FastAPI, WebSocket, Response
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from agents import Agent, function_tool, enable_verbose_stdout_logging
from agents.voice import (
    AudioInput,
    StreamedAudioInput,
    SingleAgentVoiceWorkflow,
    VoicePipeline,
    TTSModelSettings,
    STTModelSettings,
    VoicePipelineConfig
)
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from dotenv import load_dotenv

load_dotenv(".env")
enable_verbose_stdout_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ACS WebSocket URL
ws_url = os.getenv("WS_URL", "wss://your-public-url/ws")

# ────────────── Azure Search Setup ──────────────
azure_embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model="text-embedding-3-small",
)
azure_search = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_AI_SEARCH_KEY"),
    index_name="healthcare-intents-index",
    embedding_function=azure_embeddings.embed_query
)

# Load the 200 healthcare intents from JSON
with open("intents.json", "r") as f:
    INTENTS = json.load(f)["intents"]


# ────────────── Custom Tools ──────────────
@function_tool
async def authenticate_caller(member_id: str, dob: str) -> str:
    """
    Validate caller via ID + DOB.
    Return 'authenticated' or 'authentication_failed'.
    """
    # TODO: replace with real lookup
    valid = (member_id == "12345" and dob == "1970-01-01")
    return "authenticated" if valid else "authentication_failed"

@function_tool
async def retrieve_intent_candidates(query: str, k: int = 6) -> list[str]:
    """
    Use Azure Search vector similarity to return top-k intent names.
    """
    docs = await azure_search.asimilarity_search(query=query, k=k, search_type="similarity")
    return [doc.metadata["name"] for doc in docs]


# ────────────── Intent Classification Agent ──────────────
intent_instructions = """
You are a healthcare insurance intent classifier. 
1. Authenticate the caller with authenticate_caller(member_id, dob).
2. Retrieve top-6 candidate intents via retrieve_intent_candidates(query).
3. From those candidates, choose the single best intent from the predefined list.
4. Respond with only the intent name.
"""

intent_agent = Agent(
    name="HealthcareIntentClassifier",
    instructions=intent_instructions,
    model="gpt-4o-mini",
    tools=[authenticate_caller, retrieve_intent_candidates]
)


# ────────────── Weather Sample Tool (kept for demo) ──────────────
@function_tool
def get_weather(city: str) -> str:
    logger.info(f"get_weather called: {city}")
    return f"The weather in {city} is {np.random.choice(['sunny','cloudy','rainy','snowy'])}."


# ────────────── Voice Pipeline & Workflow ──────────────
custom_tts = TTSModelSettings(
    instructions=(
        "Personality: upbeat, friendly guide. Tone: clear and reassuring. "
        "Pronunciation: articulate. Tempo: moderately fast with pauses. "
        "Emotion: warm and supportive."
    ),
    voice="shimmer"
)

custom_stt = STTModelSettings(
    turn_detection={
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 500,
        "silence_duration_ms": 700,
    }
)

voice_config = VoicePipelineConfig(tts_settings=custom_tts, stt_settings=custom_stt)

# Use SingleAgentVoiceWorkflow with intent_agent
pipeline = VoicePipeline(
    workflow=SingleAgentVoiceWorkflow(intent_agent),
    config=voice_config
)


def serialize_audio_to_acs(audio_np: np.ndarray, subscription_id: str) -> dict:
    payload = base64.b64encode(audio_np.tobytes()).decode("utf-8")
    return {
        "kind": "AudioData",
        "audioData": {"data": payload, "encoding": "base64", "sampleRate": 24000},
        "streamSid": subscription_id,
    }


async def send_pipeline_events(ws: WebSocket, subscription: dict, result, state: dict):
    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            state["is_agent_speaking"] = True
            sid = subscription.get("id", "unknown")
            pkt = serialize_audio_to_acs(event.data, sid)
            await ws.send_json(pkt)
            logger.info(f"Sent audio for sid={sid}")


# ────────────── FastAPI Endpoints ──────────────
@app.post("/start_call")
async def start_call():
    return {"websocketUrl": ws_url}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket accepted")
    subscription = {"id": None}
    state = {"is_agent_speaking": False}
    streamed_input = StreamedAudioInput()

    # Kick off the voice pipeline
    pipeline_result = await pipeline.run(streamed_input)
    send_task = asyncio.create_task(send_pipeline_events(websocket, subscription, pipeline_result, state))

    try:
        async for msg in websocket.iter_text():
            data = json.loads(msg)
            kind = data.get("kind")

            if kind == "AudioMetadata":
                sid = data.get("streamSid") or data.get("audioMetadata", {}).get("subscriptionId")
                if sid:
                    subscription["id"] = sid
                    logger.info(f"Subscription ID set to {sid}")

            elif kind == "AudioData":
                audio = data.get("audioData", {})
                if audio.get("silent", False):
                    continue
                raw = base64.b64decode(audio["data"])
                audio_np = np.frombuffer(raw, dtype=np.int16)
                await streamed_input.add_audio(audio_np)

            else:
                logger.warning(f"Unknown kind: {kind}")

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        send_task.cancel()
        logger.info("WebSocket closed.")


# ────────────── Run Uvicorn ──────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)

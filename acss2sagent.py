import json
import os
import logging
import asyncio
import base64
import numpy as np
import uvicorn

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# OpenAI Agents SDK imports
from agents import Agent, function_tool, enable_verbose_stdout_logging
from agents.voice import AudioInput, StreamedAudioInput, SingleAgentVoiceWorkflow, VoicePipeline,TTSModelSettings,VoicePipelineConfig
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

# Enable verbose logging for debugging purposes.
enable_verbose_stdout_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS as needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variable for the ACS WebSocket URL.
ws_url = os.getenv("WS_URL", "wss://your-public-url/ws")


###############################################################################
#                         Agent and Voice Pipeline Setup                        #
###############################################################################

@function_tool
def get_weather(city: str) -> str:
    """A sample tool function that returns a random weather condition."""
    logger.info(f"get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {np.random.choice(choices)}."

# Define a Spanish-speaking agent for handoffs.
spanish_agent = Agent(
    name="Spanish",
    handoff_description="A Spanish speaking agent.",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. Speak in Spanish."
    ),
    model="gpt-4o-mini",
)

# Define the main agent with instructions to greet the user.
agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions(
        "Start by greeting the user with 'Hello, how can I help you?'. Then listen for the user's query and respond politely and concisely. "
        "ALWAYS speak in english unless the user speaks in spanish"
        "ONLY If the user speaks in Spanish, hand off to the Spanish agent otherwise always speak in english."
    ),
    model="gpt-4o-mini",
    handoffs=[spanish_agent],
    tools=[get_weather],
)


################################################################################
#           Helper Functions for ACS Serialization of Audio                   #
###############################################################################

def serialize_audio_to_acs(audio_np: np.ndarray, subscription_id: str) -> dict:
    """
    Converts a NumPy array (PCM int16) into an ACS-compatible JSON message.
    This represents the "deserialization" step of the voice pipeline output.
    """
    payload = base64.b64encode(audio_np.tobytes()).decode("utf-8")
    return {
        "kind": "AudioData",
        "audioData": {
            "data": payload,
            "encoding": "base64",
            "sampleRate": 24000,
        },
        "streamSid": subscription_id,
    }

async def send_pipeline_events(websocket: WebSocket, subscription: dict, pipeline_result):
    """
    Listens to events from the voice pipeline and sends each audio event (after converting
    it to an ACS message) back over the WebSocket.
    """
    async for event in pipeline_result.stream():
        if event.type == "voice_stream_event_audio":
            sid = subscription.get("id", "unknown")
            response_packet = serialize_audio_to_acs(event.data, sid)
            await websocket.send_json(response_packet)
            logger.info(f"Sent processed audio for subscriptionId: {sid}")


###############################################################################
#                           ACS WebSocket Endpoints                           #
###############################################################################

@app.post("/start_call")
async def start_call():
    """
    Endpoint for ACS to retrieve the WebSocket URL used for audio streaming.
    """
    return {"websocketUrl": ws_url}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for ACS bidirectional audio streaming.
    
    This endpoint handles two types of messages:
    
      - 'AudioMetadata': to update the subscription ID.
      - 'AudioData': which contains base64-encoded PCM audio (expected as int16).
      
    Incoming audio chunks are decoded and added to a StreamedAudioInput instance,
    which is then processed by the voice pipeline. Output audio events are serialized
    to ACS format and sent back.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted.")

    # A dictionary to hold the current subscription ID.
    subscription = {"id": None}

    # Create an instance of StreamedAudioInput for streaming audio.
    streamed_input = StreamedAudioInput()

    # Define custom TTS model settings with the desired instructions
    custom_tts_settings = TTSModelSettings(
        instructions="Personality: upbeat, friendly, persuasive guide"
        "Tone: Friendly, clear, and reassuring, creating a calm atmosphere and making the listener feel confident and comfortable."
        "Pronunciation: Clear, articulate, and steady, ensuring each instruction is easily understood while maintaining a natural, conversational flow."
        "Tempo: Speak relatively fast, include brief pauses and after before questions"
        "Emotion: Warm and supportive, conveying empathy and care, ensuring the listener feels guided and safe throughout the journey.",
        voice="shimmer"
    )
    voice_pipeline_config = VoicePipelineConfig(tts_settings=custom_tts_settings)
    # Initialize the voice pipeline with the desired workflow.
    pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent),config=voice_pipeline_config)
    # Run the pipeline using the streaming input.
    pipeline_result = await pipeline.run(streamed_input)

    # Start a background task to forward pipeline events to ACS.
    send_task = asyncio.create_task(send_pipeline_events(websocket, subscription, pipeline_result))

    try:
        async for message in websocket.iter_text():
            logger.debug(f"Received message: {message}")
            try:
                packet = json.loads(message)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                continue

            kind = packet.get("kind")
            if kind == "AudioMetadata":
                # Update the subscription id based on incoming metadata.
                sid = packet.get("streamSid") or packet.get("audioMetadata", {}).get("subscriptionId")
                if sid:
                    subscription["id"] = sid
                    logger.info(f"Updated subscription id to: {sid}")
            elif kind == "AudioData":
                audio_data = packet.get("audioData", {})
                if audio_data.get("silent"):
                    logger.info("Received silent audio packet.")
                    continue

                raw_audio_base64 = audio_data.get("data")
                if not raw_audio_base64:
                    logger.warning("Missing audio data in packet.")
                    continue

                try:
                    raw_bytes = base64.b64decode(raw_audio_base64)
                except Exception as e:
                    logger.error(f"Error decoding base64 audio: {e}")
                    continue

                # Convert the raw bytes to a NumPy array (PCM int16)
                audio_np = np.frombuffer(raw_bytes, dtype=np.int16)

                # Add the decoded audio to the streamed input queue.
                await streamed_input.add_audio(audio_np)
            else:
                logger.warning(f"Unknown message kind received: {kind}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        send_task.cancel()
        logger.info("WebSocket connection closed.")


###############################################################################
#                                Uvicorn Run                                  #
###############################################################################

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)

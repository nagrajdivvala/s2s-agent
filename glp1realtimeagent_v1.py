import json
import os
import logging
import asyncio
import base64
import numpy as np
import uvicorn

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import from the Agents SDK.
from agents import Agent, function_tool, enable_verbose_stdout_logging
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

# Load environment variables and configure logging.
load_dotenv('.env')
enable_verbose_stdout_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

azureOpenAIEmbeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            model="text-embedding-3-small",
        )
azureSearch = AzureSearch(
            azure_search_endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
            azure_search_key=os.getenv("AZURE_AI_SEARCH_KEY"),
            index_name="vector-1727345921686",
            embedding_function=azureOpenAIEmbeddings.embed_query
        )
# ACS WebSocket URL (for ACS audio integration)
ws_url = os.getenv("WS_URL", "wss://your-public-url/ws")

# Set your default OpenAI API key.
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type="openai"

###########################################################################
#                         Define Agent Tools                               #
###########################################################################

#@function_tool
def get_weather(city: str) -> str:
    """
    Returns a random weather condition for the specified city.
    """
    logger.info(f"get_weather called with city: {city}")
    conditions = ["sunny", "cloudy", "rainy", "snowy"]
    answer=f"The weather in {city} is {np.random.choice(conditions)}."
    output=[answer]
    return output

#@function_tool
async def answer_glp1(question: str) -> str:
    """ A search tool to answer glp1 questions"""
    logger.info(f"answer_glp1 tool is invoked for query - {question}")
    try:
        docs = await azureSearch.asimilarity_search(
            query=question, k=6, search_type="similarity"
         )
       # print("DOCUMENTS::",docs)
        faq_context = ""
        for doc in docs:
            #faq_context += f"Q: {doc.page_content}\nA: {doc.metadata['chunk']}\n\n"
            faq_context += f"{doc.page_content}"

        return f"Answer the user's question using the following information from our FAQ. If you don't have enough information, respond with 'I'm sorry, I don't have that information.\n\n{faq_context}",

    except Exception as e:
        logger.info("error in glp1 search tool - {e}",exc_info=True)
        return [{"role": "system", "content": "I'm sorry, I can't answer that."}]

# Define a Spanish-speaking handoff agent for GLP1 questions.
glp1_agent = Agent(
    name="glp1",
    handoff_description="Agent that answers GLP1 drugs related questions.",
    instructions=prompt_with_handoff_instructions(
        """
        ## Identity
        You are a helpful assistant specialized in GLP1 drugs.
        Answer only GLP1-related questions and advise professional help if needed.

        ## Instructions
        ALWAYS use answer_glp1 for GLP1 queries.
        """
    ),
    model="gpt-4o-mini",
    tools=[answer_glp1]
)

# Define the main agent.
agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions("""
        Greet the user politely. For weather queries, use get_weather; for GLP1 queries, hand off to glp1.
        Personality: upbeat, friendly, persuasive guide
        Tone: Friendly, clear, and reassuring, creating a calm atmosphere and making the listener feel confident and comfortable.
        Pronunciation: Clear, articulate, and steady, ensuring each instruction is easily understood while maintaining a natural, conversational flow.
        Tempo: Speak relatively fast, including brief pauses before and after questions.
        Emotion: Warm and supportive, conveying empathy and care, ensuring the listener feels guided and safe throughout the journey."""

    ),
    model="gpt-4o-mini",
    handoffs=[glp1_agent],
    tools=[get_weather]
)

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "A tool that returns the current weather condition for a given city. It should be used when the user asks for weather information.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city (e.g., 'San Francisco')"
                }
            },
            "required": ["city"]
        }
    },
    {
        "type": "function",
        "name": "answer_glp1",
        "description": (
            "A tool to answer questions about medicare prescription payment plan and GLP1 drugs used for weight loss and Type 2 Diabetes. "
            "It should only be used when the query relates specifically to GLP1 drugs or medicare prescription payment plan and should advise users to consult a healthcare professional for "
            "medical advice if appropriate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question about medicare prescription payment plan and GLP1 drugs that the user has asked."
                }
            },
            "required": ["question"]
        }
    }
]


###########################################################################
#           Helper Function: ACS Audio Serialization                       #
###########################################################################

def serialize_audio_to_acs(audio_np: np.ndarray, subscription_id: str) -> dict:
    """
    Converts a NumPy array (PCM int16) into an ACS-compatible JSON message.
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

###########################################################################
#       Forward Realtime API Events Back to ACS (Async Version)             #
###########################################################################

async def send_pipeline_events(websocket: WebSocket, subscription: dict, realtime_conn, state: dict):
    """
    Forwards events from the realtime API to ACS.
    Handles audio delta events and function call events.
    """
    await realtime_conn.response.create(response={
                        "instructions": "Greet the caller and ask how can i help you ? ",
                        # You can include additional inference parameters (e.g., temperature) if needed.
    })
    async for event in realtime_conn:
        if event.type == "response.audio.delta":
            audio_bytes = base64.b64decode(event.delta)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            response_packet = serialize_audio_to_acs(audio_np, subscription.get("id", "unknown"))
            await websocket.send_json(response_packet)
            logger.info(f"Sent {len(audio_bytes)} bytes of audio to ACS.")
        elif event.type == "error":
            print(event.error.type)
            print(event.error.code)
            print(event.error.event_id)
            print(event.error.message)
        # ──────────────────────────────────────────────────────────────────────────
    # NEW: When server VAD detects that the user has started speaking,
    #      clear ACS/Twilio's queued audio so that barge-in is immediate
        elif event.type == "input_audio_buffer.speech_started":
            sid = subscription.get("id", "unknown")

            # 2) send StopAudio control if your ACS client supports it
            stop_audio = {
                "Kind": "StopAudio",
                "AudioData": None,
                "StopAudio": {}
            }
            await websocket.send_json(stop_audio)
            logger.info(f"Sent StopAudio to ACS (streamSid={sid})")

        elif event.type == "response.done":
            #logger.info("Realtime response completed.")
                        # Check if the done event includes a function call in the output.
            output = getattr(event.response, "output", [])

            for item in output:
                if item.type == "function_call":
                    logger.info(f"Function call from response.done: {item}")
                    try:
                        args = json.loads(item.arguments if item.arguments is not None else "{}")
                    except Exception as e:
                        logger.error(f"Error parsing function arguments: {e}")
                        args = {}
                    # Retrieve the function name.
                    func_name = getattr(item, "name", "")

                    if func_name == "get_weather":
                        result_text = get_weather(args.get("city", "unknown"))
                        logger.info(f"get_weather output: {result_text}")
                    elif func_name == "answer_glp1":
                        result_text = await answer_glp1(args.get("question", "unknown"))
                        logger.info(f"answer_glp1 output: {result_text}")
                    else:
                        result_text = "No matching tool found."
                    await realtime_conn.conversation.item.create(item={
                        "type": "function_call_output",
                        "output": result_text[0],
                        #"content": [{"type": "output_text", "text": result_text}],
                        "call_id": getattr(item, "call_id", "unknown")
                    })
                    # Emit response.create to trigger a new model response.
                    await realtime_conn.response.create(response={
                        "instructions": agent.instructions,
                        # You can include additional inference parameters (e.g., temperature) if needed.
                    })
            logger.info("Response done for this turn; waiting for next input.")
            #break

###########################################################################
#                      ACS WebSocket Endpoints                             #
###########################################################################

@app.post("/start_call")
async def start_call():
    """
    Returns the ACS WebSocket URL.
    """
    return {"websocketUrl": ws_url}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    ACS WebSocket endpoint for bidirectional audio streaming.
    It receives ACS messages and forwards audio to the realtime API,
    then sends realtime API responses back to ACS.
    """
    await websocket.accept()
    logger.info("ACS WebSocket connection accepted.")
    state = {"is_agent_speaking": False}
    subscription = {"id": None}

    # rtp=openai.beta.realtime.connect(model="gpt-4o-realtime-preview")
    # rtp.__connection.session.update
    # Use the asynchronous realtime connection manager.
    client = openai.AsyncOpenAI()
    async with client.beta.realtime.connect(model="gpt-4o-realtime-preview") as realtime_conn:
        # Send a session.update event to configure the realtime session.
        await realtime_conn.session.update(session={
            "modalities": ["audio","text"],
            "instructions": agent.instructions,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "turn_detection": {                    # Added turn_detection to trigger response generation
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 500,
                "silence_duration_ms": 700,
                "interrupt_response": True,
                "create_response" : True,
            },
            "tools": tools,
            "tool_choice":"auto",
            "voice":"coral"
        })
        logger.info("Realtime session updated with audio modalities and tool definitions.")
        # async for event in realtime_conn:
        #  if event.type == 'error':
        #     print(event.error.type)
        #     print(event.error.code)
        #     print(event.error.event_id)
        #     print(event.error.message)

        # Start a background task to forward realtime events to ACS.
        send_task = asyncio.create_task(send_pipeline_events(websocket, subscription, realtime_conn, state))
        try:
            async for message in websocket.iter_text():
                logger.debug(f"Received ACS message: {message}")
                try:
                    packet = json.loads(message)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                    continue

                kind = packet.get("kind")
                if kind == "AudioMetadata":
                    sid = packet.get("streamSid") or packet.get("audioMetadata", {}).get("subscriptionId")
                    if sid:
                        subscription["id"] = sid
                        logger.info(f"Subscription ID updated: {sid}")
                elif kind == "AudioData":
                    audio_data = packet.get("audioData", {})
                    if audio_data.get("silent"):
                        logger.info("Received silent audio; skipping.")
                        continue
                    raw_audio_base64 = audio_data.get("data")
                    if not raw_audio_base64:
                        logger.warning("Missing audio data in ACS packet.")
                        continue
                    try:
                        raw_bytes = base64.b64decode(raw_audio_base64)
                    except Exception as e:
                        logger.error(f"Error decoding ACS audio: {e}")
                        continue
                    audio_np = np.frombuffer(raw_bytes, dtype=np.int16)
                    # Forward the ACS audio (re-encoded) to the realtime API.
                    await realtime_conn.send({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(audio_np.tobytes()).decode("utf-8")
                    })
                    logger.debug("Forwarded ACS audio to realtime API.")
                else:
                    logger.warning(f"Unknown ACS message kind: {kind}")
        except Exception as e:
            logger.error(f"ACS WebSocket error: {e}")
        finally:
            send_task.cancel()
            logger.info("Closing ACS WebSocket connection.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)

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
from agents.voice import AudioInput, StreamedAudioInput, SingleAgentVoiceWorkflow, VoicePipeline,TTSModelSettings,VoicePipelineConfig
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from dotenv import load_dotenv
load_dotenv('.env')

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

###############################################################################
#                         Agent and Voice Pipeline Setup                        #
###############################################################################

@function_tool
def get_weather(city: str) -> str:
    """A sample tool function that returns a random weather condition."""
    logger.info(f"get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {np.random.choice(choices)}."

@function_tool
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

        return f"Answer the user's question using the following information from our FAQ. If you don't have enough informat
ion, respond with 'I'm sorry, I don't have that information.\n\n{faq_context}'",

    except Exception as e:
        logger.info("error in glp1 search tool - {e}",exc_info=True)
        return [{"role": "system", "content": "I'm sorry, I can't answer that."}]
    

# Define a Spanish-speaking agent for handoffs.
glp1_agent = Agent(
    name="glp1",
    handoff_description="Agent that answers GLP1 drugs related questions.",
    instructions=prompt_with_handoff_instructions(
        """
         ## Identity
                    You are a helpful and knowledgeable assistant for a healthcare company. You only answer questions relat
ed to GLP1 drugs for treating weight loss and Type 2 Diabetes. The names for these medications include: Semaglutide, Ozempi
c, Wegovy, Mounjaro, Saxenda, Zepbound.
                    You're not a medical professional or pharmacist, so you shouldn't provide any counseling advice. If the
 caller asks for medical advice, you should ask them to consult with a healthcare professional.

                    ## Style
                    - Be informative and comprehensive, and use language that is easy-to-understand.
                    - Maintain a professional and polite tone at all times.
                    - You are able to respond in multiple non-English languages if asked. Do your best to accommodate such 
requests.
                    - Be as concise as possible, as you are currently operating in a voice-only channel.
                    - Do not use any kind of text highlighting or special characters such as parentheses, asterisks, or pou
nd signs, since the text will be converted to audio.

                    ## Response Guideline
                    - ONLY answer questions about GLP1 drugs for weight loss and Type 2 Diabetes, and their associated cond
itions and treatments.
                    - You can provide general helpful information on basic medical conditions and terminology, but avoid of
fering any medical diagnosis or clinical guidance.
                    - Never engage in any discussion about your origin or OpenAI. If asked, respond with "I'm sorry, I can'
t answer that question."
                    - For any medical emergency, direct the user to hang up and seek immediate help.
                    - For all other healthcare related questions, please ask them to consult with a healthcare professional
.
                    
                    ## Instructions
                    - ALWAYS use answer_glp tool to answer glp1 questions 
                """
    ),
    model="gpt-4o-mini",
    tools=[answer_glp1]
)

# Define the main agent with instructions to greet the user.
agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions(
        "Start by greeting the user with 'Hello, how can I help you?'. Then listen for the user's query and respond politely and concisely. "
        "ONLY If the user asks about GLP1 drugs, hand off to the glp1_agent otherwise answer what you know."
    ),
    model="gpt-4o-mini",
    handoffs=[glp1_agent],
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
        # elif event.type == "voice_stream_event_lifecycle":
        #     # Handle interruption/lifecycle events here.
        #     sid = subscription.get("id", "unknown")
        #     if event.event == "turn_started":
        #         control_msg = {"kind": "AudioLifecycle", "event": "turn_started", "streamSid": sid}
        #         await websocket.send_json(control_msg)
        #         logger.info("Turn started event received: muting microphone.")
        #     elif event.event == "turn_ended":
        #         control_msg = {"kind": "AudioLifecycle", "event": "turn_ended", "streamSid": sid}
        #         await websocket.send_json(control_msg)
        #         logger.info("Turn ended event received: unmuting microphone.")


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
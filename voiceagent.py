import asyncio
import random
import numpy as np
import sounddevice as sd
import speech_recognition as sr  # Library for speech-to-text

from agents import Agent, function_tool, enable_verbose_stdout_logging
from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions

# Enable verbose logging for debugging purposes
enable_verbose_stdout_logging()

@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    print(f"[debug] get_weather called with city: {city}")
    choices = ["sunny", "cloudy", "rainy", "snowy"]
    return f"The weather in {city} is {random.choice(choices)}."

# Define a Spanish-speaking agent for handoff in Spanish conversations
spanish_agent = Agent(
    name="Spanish",
    handoff_description="A Spanish speaking agent.",
    instructions=prompt_with_handoff_instructions(
        "You're speaking to a human, so be polite and concise. Speak in Spanish."
    ),
    model="gpt-4o-mini",
)

# Define the main agent and update its instructions to welcome the user.
agent = Agent(
    name="Assistant",
    instructions=prompt_with_handoff_instructions(
        "Start by greeting the user with 'Hello, how can I help you?'. "
        "Then wait for the user's query and answer politely and concisely. "
        "ONLY If the user speaks in Spanish, hand off to the Spanish agent otherwise always speak in english."
    ),
    model="gpt-4o-mini",
    handoffs=[spanish_agent],
    tools=[get_weather],
)

async def conversation_loop(pipeline: VoicePipeline):
    """
    Continuously record audio from the microphone, transcribe the user’s input
    to check for an exit keyword ("bye"), and then process the audio through the
    voice pipeline for multi-turn conversation.
    """
    # Start an output audio stream to play the agent's responses
    player = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
    player.start()

    # Initialize the recognizer for transcription
    recognizer = sr.Recognizer()

    print("Multi-turn conversation started. You may speak after the prompt.")
    while True:
        try:
            # Record a 3-second audio clip from the default microphone
            print("Listening... Please speak now.")
            duration = 3  # seconds
            rec_buffer = sd.rec(int(24000 * duration), samplerate=24000, channels=1, dtype=np.int16)
            sd.wait()  # Wait for recording to complete
            audio_data = rec_buffer.flatten()

            # Convert recorded audio to bytes and perform speech-to-text transcription
            try:
                # For speech_recognition, create an AudioData object.
                audio_bytes = audio_data.tobytes()
                user_audio = sr.AudioData(audio_bytes, 24000, 2)
                user_text = recognizer.recognize_google(user_audio)
                print("User said (transcribed):", user_text)
            except Exception as transcribe_error:
                print(f"[Transcription Error] {transcribe_error}")
                user_text = ""

            # Check if the transcription contains the exit keyword "bye"
            if "bye" in user_text.lower():
                print("Exit keyword detected. Ending conversation. Goodbye!")
                break

            # Wrap the recorded audio into an AudioInput object for the pipeline
            audio_input = AudioInput(buffer=audio_data)
            result = await pipeline.run(audio_input)

            # Stream and play the agent's response audio events in real time
            async for event in result.stream():
                if event.type == "voice_stream_event_audio":
                    player.write(event.data)
        except Exception as e:
            print(f"[error] An error occurred during the conversation loop: {e}")
            break

async def main():
    # Initialize the voice pipeline with the single-agent workflow
    pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))

    # Trigger an initial (welcome) response.
    # A short period of silence (1 second) helps to signal that it's the initial turn.
    print("Starting conversation with a welcome message...")
    initial_buffer = np.zeros(24000 * 1, dtype=np.int16)  # 1 second of silence
    initial_audio_input = AudioInput(buffer=initial_buffer)
    initial_result = await pipeline.run(initial_audio_input)

    # Create an output stream to play the welcome message
    player = sd.OutputStream(samplerate=24000, channels=1, dtype=np.int16)
    player.start()

    async for event in initial_result.stream():
        if event.type == "voice_stream_event_audio":
            player.write(event.data)

    # After playing the welcome message, start the multi-turn conversation loop.
    await conversation_loop(pipeline)

if __name__ == "__main__":
    asyncio.run(main())
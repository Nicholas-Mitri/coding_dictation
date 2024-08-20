import time
from openai import OpenAI


def send_system_and_user_prompts(system_prompt, user_prompt):
    """
    Send system and user prompts to the OpenAI API and get the response.

    Parameters:
    system_prompt (str): The prompt for the system role.
    user_prompt (str): The prompt for the user role.

    Returns:
    str: The response content from the OpenAI model.
    """
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return completion.choices[0].message.content


def transcribe_audio(audio_file_path):
    """
    Transcribe an audio file using OpenAI's Whisper model.

    Parameters:
    audio_file_path (str): The file path to the audio file.

    Returns:
    str: The transcribed text from the audio file.
    """
    client = OpenAI()
    with open(audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
    return transcription.text


if __name__ == "__main__":
    # Define your system and user prompts
    system_prompt = "You are a helpful assistant."
    user_prompt = "Can you explain the theory of relativity in simple terms?"

    # Get the response from the model
    time_start = time.perf_counter()
    response_text = send_system_and_user_prompts(system_prompt, user_prompt)
    time_stop = time.perf_counter()
    print(f"GPT-4 Response in {time_stop - time_start:.2f} seconds:", response_text)

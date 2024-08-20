from pynput import keyboard
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import gpt_fn, time, pyperclip, sys, os
import pygame

# Global variable to control the recording state
is_recording = False


def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()


# Code execution continues while the audio is playing
def record_audio():
    global is_recording
    is_recording = True

    # Define the audio settings
    fs = 44100  # Sample rate
    channels = 1  # Number of channels

    # List to store the recorded chunks
    recorded_chunks = []

    def callback(indata, frames, time, status):
        if is_recording:
            recorded_chunks.append(indata.copy())
        else:
            raise sd.CallbackStop

    # Start recording
    with sd.InputStream(samplerate=fs, channels=channels, callback=callback):
        while is_recording:
            sd.sleep(100)

    # Concatenate all recorded chunks
    recording = np.concatenate(recorded_chunks, axis=0)

    # Save the recording to a file
    filename = "./audio files/recording.wav"
    sf.write(filename, recording, fs)


def stop_recording():
    global is_recording
    is_recording = False
    # wait for the recording to stop
    time.sleep(1)


def on_activate_start():
    print("Recording started")

    #  play a local sound to indicate the recording has started
    play_audio("./audio files/chime.mp3")
    t = threading.Thread(target=record_audio)
    t.start()


def on_activate_stop():
    print("Recording stopped")

    #  play a sound to indicate the recording has stopped
    play_audio("./audio files/chime.mp3")

    stop_recording()

    # send audio to openAI whisper model to transcribe
    recording_transcipt = gpt_fn.transcribe_audio("./audio files/recording.wav")
    system_prompt = "You are a helpful assistant."

    print("Transcription:", recording_transcipt, end="\n\n")

    time_start = time.perf_counter()
    try:
        with open("system_prompt.txt", "r") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        print("System prompt file not found. Please make sure the file exists.")

    response_text = gpt_fn.send_system_and_user_prompts(
        system_prompt, recording_transcipt
    )
    time_stop = time.perf_counter()

    # save the response to clipboard
    pyperclip.copy(response_text)
    play_audio("./audio files/start.mp3")

    # send response to keyboard to write it out
    keyboard_controller = keyboard.Controller()
    keyboard_controller.press(keyboard.Key.cmd)
    keyboard_controller.press("v")
    keyboard_controller.release("v")
    keyboard_controller.release(keyboard.Key.cmd)

    # delete the recording file
    os.remove("audio files/recording.wav")

    # print(
    #     f"GPT-4 Response in {time_stop - time_start:.2f} seconds:",
    #     response_text,
    # )


def on_activate_exit():
    print("Exiting...")
    sys.exit()


def run_listen_up():
    print("Listening for keyboard shortcuts...")
    # Define the keyboard shortcuts
    start_recording = "<f3>"
    stop_recording = "<f4>"
    exit_app = "<cmd>+<shift>+<ctrl>+<alt>+q"

    # Create the keyboard listener
    with keyboard.GlobalHotKeys(
        {
            start_recording: on_activate_start,
            stop_recording: on_activate_stop,
            exit_app: on_activate_exit,
        }
    ) as h:
        h.join()


if __name__ == "__main__":
    run_listen_up()

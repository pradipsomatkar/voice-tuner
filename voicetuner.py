import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly, lfilter
import threading
import queue
import os
import pyrubberband as rb

# Parameters
SAMPLE_RATE = 44100  # Sample rate in Hz
BLOCK_SIZE = 1024    # Number of samples per block
RECORDING_FOLDER = "recordings"  # Folder to save recordings

# Reverb and echo buffers
reverb_buffer = np.zeros(int(SAMPLE_RATE * 0.5))  # 0.5-second reverb buffer
echo_buffer = np.zeros(int(SAMPLE_RATE * 1.0))    # 1.0-second echo buffer

# Queue for recording
recording_queue = queue.Queue()

# Global variables for effects
pitch_shift = 0
reverb_decay = 0.5
echo_delay = 0.3
echo_feedback = 0.5
auto_tune_enabled = False

# Create recordings folder if it doesn't exist
if not os.path.exists(RECORDING_FOLDER):
    os.makedirs(RECORDING_FOLDER)

def pitch_shift_audio(input_signal, pitch_shift):
    """
    Shift the pitch of the input signal using pyrubberband.
    """
    try:
        return rb.pitch_shift(input_signal, SAMPLE_RATE, pitch_shift)
    except Exception as e:
        print(f"Error in pitch_shift_audio: {e}")
        return input_signal

def apply_reverb(input_signal, decay):
    """
    Apply a simple reverb effect using a feedback delay.
    """
    try:
        global reverb_buffer
        reverb_signal = np.zeros_like(input_signal)
        for i in range(len(input_signal)):
            reverb_signal[i] = input_signal[i] + reverb_buffer[-1] * decay
            reverb_buffer = np.roll(reverb_buffer, 1)
            reverb_buffer[0] = reverb_signal[i]
        return reverb_signal
    except Exception as e:
        print(f"Error in apply_reverb: {e}")
        return input_signal

def apply_echo(input_signal, delay, feedback):
    """
    Apply an echo effect using a delay buffer.
    """
    try:
        global echo_buffer
        echo_signal = np.zeros_like(input_signal)
        delay_samples = int(delay * SAMPLE_RATE)
        for i in range(len(input_signal)):
            echo_signal[i] = input_signal[i] + echo_buffer[-delay_samples] * feedback
            echo_buffer = np.roll(echo_buffer, 1)
            echo_buffer[0] = echo_signal[i]
        return echo_signal
    except Exception as e:
        print(f"Error in apply_echo: {e}")
        return input_signal

def auto_tune(input_signal):
    """
    Apply pitch correction using pyrubberband.
    """
    try:
        # Detect the pitch of the input signal
        pitch = rb.detect_pitch(input_signal, SAMPLE_RATE)
        # Correct the pitch to the nearest semitone
        corrected_pitch = round(pitch)
        # Shift the pitch to the corrected pitch
        return rb.pitch_shift(input_signal, SAMPLE_RATE, corrected_pitch - pitch)
    except Exception as e:
        print(f"Error in auto_tune: {e}")
        return input_signal

def audio_callback(indata, outdata, frames, time, status):
    """
    Callback function for real-time audio processing.
    """
    try:
        if status:
            print(f"Audio stream status: {status}")

        # Extract mono audio
        input_signal = indata[:, 0]

        # Apply pitch shifting
        pitched_audio = pitch_shift_audio(input_signal, pitch_shift)

        # Apply auto-tune
        if auto_tune_enabled:
            pitched_audio = auto_tune(pitched_audio)

        # Apply reverb
        reverbed_audio = apply_reverb(pitched_audio, reverb_decay)

        # Apply echo
        echoed_audio = apply_echo(reverbed_audio, echo_delay, echo_feedback)

        # Normalize audio to prevent clipping
        output_signal = echoed_audio / np.max(np.abs(echoed_audio))

        # Output the processed audio
        outdata[:] = output_signal.reshape(-1, 1)

        # Add the processed audio to the recording queue
        if not recording_queue.full():
            recording_queue.put(output_signal)
    except Exception as e:
        print(f"Error in audio_callback: {e}")

def record_audio(track_name):
    """
    Record the processed audio to a file.
    """
    try:
        print(f"Recording started for track: {track_name}")
        recorded_audio = []
        while True:
            data = recording_queue.get()
            if data is None:  # Stop signal
                break
            recorded_audio.append(data)
        recorded_audio = np.concatenate(recorded_audio)
        recording_file = os.path.join(RECORDING_FOLDER, f"{track_name}.wav")
        sf.write(recording_file, recorded_audio, SAMPLE_RATE)
        print(f"Recording saved to {recording_file}")
    except Exception as e:
        print(f"Error in record_audio: {e}")

def start_recording():
    """
    Start recording a new track.
    """
    try:
        track_name = track_name_entry.get()
        if not track_name:
            print("Please enter a track name.")
            return
        recording_thread = threading.Thread(target=record_audio, args=(track_name,))
        recording_thread.start()
    except Exception as e:
        print(f"Error in start_recording: {e}")

def stop_recording():
    """
    Stop recording the current track.
    """
    try:
        recording_queue.put(None)
    except Exception as e:
        print(f"Error in stop_recording: {e}")

def update_pitch_shift(value):
    """
    Update the pitch shift value.
    """
    global pitch_shift
    try:
        pitch_shift = float(value)
    except Exception as e:
        print(f"Error in update_pitch_shift: {e}")

def update_reverb_decay(value):
    """
    Update the reverb decay value.
    """
    global reverb_decay
    try:
        reverb_decay = float(value)
    except Exception as e:
        print(f"Error in update_reverb_decay: {e}")

def update_echo_delay(value):
    """
    Update the echo delay value.
    """
    global echo_delay
    try:
        echo_delay = float(value)
    except Exception as e:
        print(f"Error in update_echo_delay: {e}")

def update_echo_feedback(value):
    """
    Update the echo feedback value.
    """
    global echo_feedback
    try:
        echo_feedback = float(value)
    except Exception as e:
        print(f"Error in update_echo_feedback: {e}")

def toggle_auto_tune():
    """
    Toggle auto-tune on/off.
    """
    global auto_tune_enabled
    try:
        auto_tune_enabled = not auto_tune_enabled
    except Exception as e:
        print(f"Error in toggle_auto_tune: {e}")

# Create the main window
root = tk.Tk()
root.title("Advanced Voice Tuner")

# Track name entry
tk.Label(root, text="Track Name:").grid(row=0, column=0, padx=10, pady=10)
track_name_entry = tk.Entry(root)
track_name_entry.grid(row=0, column=1, padx=10, pady=10)

# Start/stop recording buttons
start_button = tk.Button(root, text="Start Recording", command=start_recording)
start_button.grid(row=1, column=0, padx=10, pady=10)
stop_button = tk.Button(root, text="Stop Recording", command=stop_recording)
stop_button.grid(row=1, column=1, padx=10, pady=10)

# Pitch shift slider
tk.Label(root, text="Pitch Shift:").grid(row=2, column=0, padx=10, pady=10)
pitch_shift_slider = ttk.Scale(root, from_=-12, to=12, command=update_pitch_shift)
pitch_shift_slider.set(0)
pitch_shift_slider.grid(row=2, column=1, padx=10, pady=10)

# Reverb decay slider
tk.Label(root, text="Reverb Decay:").grid(row=3, column=0, padx=10, pady=10)
reverb_decay_slider = ttk.Scale(root, from_=0, to=1, command=update_reverb_decay)
reverb_decay_slider.set(0.5)
reverb_decay_slider.grid(row=3, column=1, padx=10, pady=10)

# Echo delay slider
tk.Label(root, text="Echo Delay:").grid(row=4, column=0, padx=10, pady=10)
echo_delay_slider = ttk.Scale(root, from_=0, to=1, command=update_echo_delay)
echo_delay_slider.set(0.3)
echo_delay_slider.grid(row=4, column=1, padx=10, pady=10)

# Echo feedback slider
tk.Label(root, text="Echo Feedback:").grid(row=5, column=0, padx=10, pady=10)
echo_feedback_slider = ttk.Scale(root, from_=0, to=1, command=update_echo_feedback)
echo_feedback_slider.set(0.5)
echo_feedback_slider.grid(row=5, column=1, padx=10, pady=10)

# Auto-tune toggle button
auto_tune_button = tk.Button(root, text="Toggle Auto-Tune", command=toggle_auto_tune)
auto_tune_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

try:
    # Start the audio stream
    stream = sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,  # Mono audio
        dtype=np.float32,
        callback=audio_callback
    )
    stream.start()
except Exception as e:
    print(f"Error starting audio stream: {e}")

# Run the GUI
root.mainloop()

# Stop the audio stream when the GUI is closed
try:
    stream.stop()
except Exception as e:
    print(f"Error stopping audio stream: {e}")
    
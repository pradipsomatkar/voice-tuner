# === Voice Refiner App (Safe for WAV/with FFmpeg for M4A/MP3) ===

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import os
import shutil

# --- Compatibility patch for librosa ---
if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "complex"):
    np.complex = np.complex128

# --- Output directory (fixed path) ---
OUTPUT_DIR = r"D:\SP\tune"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Processing Function ---
def process_audio(input_path, output_path, use_autotune=True):
    ext = os.path.splitext(input_path)[1].lower()

    # If input is not WAV, convert using ffmpeg via pydub
    if ext != ".wav":
        try:
            audio = AudioSegment.from_file(input_path)
            temp_wav_path = "temp.wav"
            audio.export(temp_wav_path, format="wav")
            load_path = temp_wav_path
        except Exception as e:
            raise RuntimeError("FFmpeg is required for non-WAV files. Please install FFmpeg.") from e
    else:
        load_path = input_path

    # Load audio
    y, sr = librosa.load(load_path, sr=None)

    # Noise reduction
    stft = librosa.stft(y)
    magnitude, phase = librosa.magphase(stft)
    noise_profile = np.median(magnitude, axis=1, keepdims=True)
    mask = magnitude > noise_profile * 1.5
    stft_denoised = stft * mask
    y_denoised = librosa.istft(stft_denoised)

    # EQ enhancement
    y_fft = np.fft.rfft(y_denoised)
    freqs = np.fft.rfftfreq(len(y_denoised), 1/sr)
    eq_boost = np.ones_like(y_fft)
    eq_boost[(freqs > 150) & (freqs < 300)] *= 1.2   # Warmth
    eq_boost[(freqs > 3000) & (freqs < 5000)] *= 1.3 # Clarity
    y_eq = np.fft.irfft(y_fft * eq_boost)

    # Pitch correction
    if use_autotune:
        y_corrected = librosa.effects.pitch_shift(y_eq, sr=sr, n_steps=0.5)
    else:
        y_corrected = librosa.effects.pitch_shift(y_eq, sr=sr, n_steps=0.2)

    # Normalize
    y_normalized = librosa.util.normalize(y_corrected)

    # Save output
    sf.write(output_path, y_normalized, sr)


# === GUI Application ===
class VoiceRefineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 Voice Refiner App")
        self.root.geometry("420x250")

        self.file_path = None
        self.use_autotune = tk.BooleanVar(value=True)

        tk.Label(root, text="Voice Refiner", font=("Arial", 16, "bold")).pack(pady=10)
        tk.Button(root, text="Select Audio File", command=self.select_file).pack(pady=5)
        tk.Checkbutton(root, text="Enable Auto-Tune (robotic effect)", variable=self.use_autotune).pack(pady=5)
        tk.Button(root, text="Process Audio", command=self.run_processing).pack(pady=15)

    def select_file(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.m4a *.wav *.mp3")]
        )
        if self.file_path:
            messagebox.showinfo("File Selected", f"Selected:\n{self.file_path}")

    def run_processing(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select an audio file first.")
            return

        base_name = os.path.basename(self.file_path)
        name, _ = os.path.splitext(base_name)
        output_path = os.path.join(OUTPUT_DIR, f"{name}_refined.wav")

        try:
            process_audio(self.file_path, output_path, self.use_autotune.get())
            messagebox.showinfo("Success", f"Processed file saved at:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")


# === Run App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceRefineApp(root)
    root.mainloop()





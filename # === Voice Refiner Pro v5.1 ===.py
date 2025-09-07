# === Voice Refiner Pro v5.1 ===
# FIXED: Added a main scrollbar to ensure all content is visible on any screen size.

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range
import os
import threading
import json
from scipy.signal import convolve
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

plt.style.use('dark_background')

# --- Compatibility patch ---
if not hasattr(np, "float"): np.float = np.float64
if not hasattr(np, "complex"): np.complex = np.complex128

# === Core Audio Processing Engine (Unchanged) ===
def process_audio(y, sr, params):
    output_path = params['output_path']
    ir_path = params.get('ir_path')
    stft = librosa.stft(y)
    magnitude, phase = librosa.magphase(stft)
    noise_profile = np.median(magnitude, axis=1, keepdims=True)
    mask = magnitude > noise_profile * params['noise_level']
    y_denoised = librosa.istft(stft * mask, length=len(y))
    y_int = np.int16(y_denoised / np.max(np.abs(y_denoised)) * 32767)
    audio_segment = AudioSegment(y_int.tobytes(), frame_rate=sr, sample_width=2, channels=1)
    audio_segment = compress_dynamic_range(
        audio_segment, threshold=params['comp_thresh'], ratio=params['comp_ratio'], attack=5.0, release=50.0
    )
    y_compressed = np.array(audio_segment.get_array_of_samples(), dtype=np.float64) / 32767.0
    y_fft = np.fft.rfft(y_compressed)
    freqs = np.fft.rfftfreq(len(y_compressed), 1 / sr)
    eq_boost = np.ones_like(y_fft, dtype=np.complex128)
    eq_boost[(freqs > 150) & (freqs < 300)] *= params['warmth_boost']
    eq_boost[(freqs > 3000) & (freqs < 5000)] *= params['clarity_boost']
    y_eq = np.fft.irfft(y_fft * eq_boost)
    drive = params['saturation_drive']
    y_saturated = np.tanh(y_eq * drive) / np.tanh(drive) if drive > 1.0 else y_eq
    pitch_steps = params['pitch_steps']
    y_corrected = librosa.effects.pitch_shift(y=y_saturated, sr=sr, n_steps=pitch_steps) if pitch_steps != 0 else y_saturated
    y_normalized = librosa.util.normalize(y_corrected)
    y_final = y_normalized
    if ir_path and os.path.exists(ir_path):
        ir, ir_sr = librosa.load(ir_path, sr=sr, mono=True)
        reverb_signal = convolve(y_normalized, ir, mode='full')
        reverb_signal = reverb_signal[:len(y_normalized) + sr // 2]
        reverb_signal = librosa.util.normalize(reverb_signal) * 0.7
        mix = params['reverb_mix'] / 100.0
        y_final = ((1.0 - mix) * y_normalized) + (mix * reverb_signal[:len(y_normalized)])
    y_final = librosa.util.normalize(y_final)
    sf.write(output_path, y_final, sr)

# === GUI Application ===
class VoiceRefinerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎵 Voice Refiner Pro v5.1 (Scrollable)")
        self.root.geometry("620x900") # Initial size, but now resizable
        self.root.minsize(550, 700) # Minimum practical size

        self.file_path = None
        self.ir_path = None
        self.audio_data = None
        self.sample_rate = None
        self.controls = {}

        # --- Create a main container frame ---
        main_container = ttk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True)

        # --- Create a canvas and a scrollbar ---
        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        
        # --- Create the frame to hold all the content (this is what will scroll) ---
        self.scrollable_frame = ttk.Frame(canvas)

        # --- Configure the canvas to use the scrollbar ---
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # --- Pack the scrollbar and canvas into the main container ---
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # --- Place the scrollable frame inside the canvas ---
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # --- Bind the frame's size to the canvas's scroll region ---
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.scrollable_frame.bind("<Configure>", on_frame_configure)
        
        # --- ALL WIDGETS NOW GO INSIDE self.scrollable_frame ---
        
        # --- Waveform Display Frame ---
        # Note: The parent is now self.scrollable_frame
        plot_frame = ttk.Frame(self.scrollable_frame) 
        plot_frame.pack(fill=tk.X, pady=(15, 10), padx=15)
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor('#2E2E2E')
        self.ax.set_facecolor('#2E2E2E')
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # --- File IO & Main Actions ---
        actions_frame = ttk.LabelFrame(self.scrollable_frame, text="Main Actions", padding=10)
        actions_frame.pack(fill=tk.X, pady=5, padx=15)
        self.select_button = ttk.Button(actions_frame, text="1. Select Audio File", command=self.select_file)
        self.select_button.pack(fill=tk.X, pady=2)
        self.selected_file_label = ttk.Label(actions_frame, text="No audio file selected.", wraplength=450)
        self.selected_file_label.pack(fill=tk.X, pady=2, ipady=5)
        self.magic_enhance_button = ttk.Button(actions_frame, text="✨ 2. Magic Vocal Chain (Smart Enhance)", command=self.apply_magic_enhance)
        self.magic_enhance_button.pack(fill=tk.X, pady=(10, 5), ipady=5)

        # --- Settings Notebook ---
        notebook = ttk.Notebook(self.scrollable_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10, padx=15)
        
        tab_names = ['Tuning & EQ', 'Dynamics & Color', 'Ambience (Reverb)', 'Presets']
        tabs = {name: ttk.Frame(notebook, padding=10) for name in tab_names}
        for name, tab_frame in tabs.items():
            notebook.add(tab_frame, text=name)
        
        # --- Controls ---
        self.controls['noise_level'] = self._create_slider(tabs['Tuning & EQ'], "Noise Gate", 1.0, 5.0, 1.5)
        self.controls['pitch_steps'] = self._create_slider(tabs['Tuning & EQ'], "Pitch Shift", -2.0, 2.0, 0.0)
        self.controls['warmth_boost'] = self._create_slider(tabs['Tuning & EQ'], "Warmth Boost", 1.0, 2.0, 1.0)
        self.controls['clarity_boost'] = self._create_slider(tabs['Tuning & EQ'], "Clarity Boost", 1.0, 2.0, 1.0)
        self.controls['comp_thresh'] = self._create_slider(tabs['Dynamics & Color'], "Comp Threshold", -60.0, 0.0, 0.0)
        self.controls['comp_ratio'] = self._create_slider(tabs['Dynamics & Color'], "Comp Ratio", 1.0, 10.0, 1.0)
        self.controls['saturation_drive'] = self._create_slider(tabs['Dynamics & Color'], "Saturation Drive", 1.0, 5.0, 1.0)
        self.select_ir_button = ttk.Button(tabs['Ambience (Reverb)'], text="Select Reverb IR File (.wav)", command=self.select_ir_file)
        self.select_ir_button.pack(fill=tk.X, pady=5)
        self.selected_ir_label = ttk.Label(tabs['Ambience (Reverb)'], text="No IR file selected.", wraplength=450)
        self.selected_ir_label.pack(fill=tk.X, pady=5)
        self.controls['reverb_mix'] = self._create_slider(tabs['Ambience (Reverb)'], "Reverb Mix (%)", 0.0, 100.0, 0.0)
        ttk.Button(tabs['Presets'], text="Save Current Settings as Preset", command=self.save_preset).pack(pady=10, ipady=5, fill=tk.X)
        ttk.Button(tabs['Presets'], text="Load Settings from Preset", command=self.load_preset).pack(pady=10, ipady=5, fill=tk.X)

        self.process_button = ttk.Button(self.scrollable_frame, text="▶ 3. START PROCESSING", command=self.start_processing_thread)
        self.process_button.pack(pady=10, ipady=10, fill=tk.X, padx=15)
        self.status_label = ttk.Label(self.scrollable_frame, text="Ready", anchor=tk.CENTER)
        self.status_label.pack(fill=tk.X, padx=15, pady=(0,15))

    def _create_slider(self, parent, label, from_, to, initial_val):
        # (This function is unchanged)
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=7)
        ttk.Label(frame, text=f"{label}:").pack(side=tk.LEFT, padx=5)
        value_label = ttk.Label(frame, text=f"{initial_val:.2f}", width=6)
        value_label.pack(side=tk.RIGHT, padx=5)
        slider = ttk.Scale(frame, from_=from_, to=to, orient=tk.HORIZONTAL)
        slider.set(initial_val)
        slider.configure(command=lambda val, label=value_label: label.config(text=f"{float(val):.2f}"))
        slider.pack(fill=tk.X, expand=True, padx=5)
        return slider

    def _plot_waveform(self):
        # (This function is unchanged)
        if self.audio_data is None: return
        self.ax.clear()
        librosa.display.waveshow(self.audio_data, sr=self.sample_rate, ax=self.ax, color='#1E90FF')
        self.ax.set_xlabel("Time (s)", color='white')
        self.ax.set_ylabel("Amplitude", color='white')
        self.ax.margins(x=0)
        self.fig.tight_layout()
        self.canvas.draw()
        
    def select_file(self):
        # (This function is unchanged)
        path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.m4a *.wav *.mp3")])
        if not path: return
        self.file_path = path
        try:
            self.audio_data, self.sample_rate = librosa.load(path, sr=None, mono=True)
            self._plot_waveform()
            self.selected_file_label.config(text=f"Audio: ...{os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error Loading File", f"Could not load the audio file.\n{e}")

    def apply_magic_enhance(self):
        # (This function is unchanged)
        if self.audio_data is None:
            messagebox.showerror("Error", "Please select an audio file first.")
            return
        y_trimmed, _ = librosa.effects.trim(self.audio_data, top_db=20)
        rms = librosa.feature.rms(y=y_trimmed)[0]
        dynamic_range = np.max(rms) / (np.mean(rms) + 1e-6)
        comp_ratio = np.clip(1 + dynamic_range, 2.5, 6.0)
        comp_thresh = np.clip(-10 - (dynamic_range * 2), -30, -12)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y_trimmed, sr=self.sample_rate))
        norm_centroid = spectral_centroid / (self.sample_rate / 2)
        clarity_boost = 1.0 + (0.5 * (1-norm_centroid))
        warmth_boost = 1.1 + (0.3 * norm_centroid)
        magic_settings = {
            'noise_level': 2.2, 'pitch_steps': 0.05,
            'warmth_boost': np.clip(warmth_boost, 1.0, 1.4), 'clarity_boost': np.clip(clarity_boost, 1.1, 1.6),
            'comp_thresh': comp_thresh, 'comp_ratio': comp_ratio, 'saturation_drive': 1.4, 'reverb_mix': 15.0
        }
        for key, value in magic_settings.items():
            if key in self.controls:
                self.controls[key].set(value)
        messagebox.showinfo("Magic Vocal Chain", "Smart preset applied based on audio analysis!")

    def select_ir_file(self, *args):
        # (This function is unchanged)
        path = filedialog.askopenfilename(filetypes=[("Impulse Response", "*.wav")])
        if path:
            self.ir_path = path
            self.selected_ir_label.config(text=f"IR: ...{os.path.basename(path)}")
            
    def get_params(self):
        # (This function is unchanged)
        params = {key: slider.get() for key, slider in self.controls.items()}
        params['ir_path'] = self.ir_path
        input_dir = os.path.dirname(self.file_path)
        base_name = os.path.basename(self.file_path)
        name, _ = os.path.splitext(base_name)
        params['output_path'] = os.path.join(input_dir, f"{name}_refined_pro.wav")
        return params

    def start_processing_thread(self):
        # (This function is unchanged)
        if self.audio_data is None:
            messagebox.showerror("Error", "Please select an audio file first.")
            return
        self.process_button.config(state=tk.DISABLED)
        self.status_label.config(text="Processing... Please wait.")
        thread = threading.Thread(target=self.run_processing, args=(self.audio_data, self.sample_rate, self.get_params()))
        thread.daemon = True
        thread.start()

    def run_processing(self, y, sr, params):
        # (This function is unchanged)
        try:
            process_audio(y, sr, params)
            self.root.after(0, lambda: messagebox.showinfo("Success", f"File saved:\n{params['output_path']}"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed:\n{e}"))
        finally:
            self.root.after(0, self.reset_ui)

    def reset_ui(self):
        # (This function is unchanged)
        self.process_button.config(state=tk.NORMAL)
        self.status_label.config(text="Ready")
        
    def save_preset(self):
        # (This function is unchanged)
        preset_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if not preset_path: return
        settings = {key: slider.get() for key, slider in self.controls.items()}
        with open(preset_path, 'w') as f:
            json.dump(settings, f, indent=4)
        messagebox.showinfo("Success", f"Preset saved to {preset_path}")

    def load_preset(self):
        # (This function is unchanged)
        preset_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not preset_path: return
        with open(preset_path, 'r') as f:
            settings = json.load(f)
        for key, value in settings.items():
            if key in self.controls:
                self.controls[key].set(value)
        messagebox.showinfo("Success", "Preset loaded.")

if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceRefinerApp(root)
    root.mainloop()
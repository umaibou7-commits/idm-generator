# IDM Generator - Python + Gradio version
# Requirements (install before running):
#   pip install gradio librosa soundfile numpy

import numpy as np
import librosa
import gradio as gr

SCALES = {
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}


def get_scale_notes(root_midi: int, scale_name: str, count: int):
    scale_intervals = SCALES.get(scale_name, SCALES["minor"])
    notes = []
    for i in range(count):
        octave = i // len(scale_intervals)
        interval_idx = i % len(scale_intervals)
        notes.append(root_midi + octave * 12 + scale_intervals[interval_idx])
    return np.array(notes, dtype=np.float32)


def apply_aphex_style(audio: np.ndarray, sr: int, tempo: float, glitch: float,
                      atmosphere: float, scale: str, rng: np.random.Generator):
    length = len(audio)
    result = np.zeros_like(audio, dtype=np.float32)

    # Bass notes
    bass_notes = get_scale_notes(36, scale, 8)
    note_length = length // len(bass_notes)

    for i, note in enumerate(bass_notes):
        start = i * note_length
        end = min((i + 1) * note_length, length)
        if start >= length:
            break
        n = end - start
        if n <= 0:
            continue
        t = np.arange(n, dtype=np.float32) / sr
        freq = 440.0 * (2.0 ** ((note - 69.0) / 12.0))
        envelope = np.exp(-2.0 * t / (note_length / sr + 1e-6))
        sine = np.sin(2.0 * np.pi * freq * t)
        sub = np.sin(2.0 * np.pi * freq * 0.5 * t)
        result[start:end] += (0.7 * sine + 0.3 * sub) * envelope * 0.15

    # Kicks
    bpm = 120.0 * tempo
    beat_samples = int(sr * 60.0 / bpm)
    if beat_samples <= 0:
        beat_samples = int(sr * 60.0 / 120.0)
    for i in range(0, length, beat_samples):
        kick_len = int(sr * 0.15)
        end = min(i + kick_len, length)
        n = end - i
        if n <= 0:
            continue
        t = np.arange(n, dtype=np.float32) / sr
        kick = np.sin(2.0 * np.pi * 50.0 * t) * np.exp(-15.0 * t)
        result[i:end] += kick * 0.4

    # Drone + original
    i_arr = np.arange(length, dtype=np.float32)
    lfo = np.sin(2.0 * np.pi * 0.5 * i_arr / sr)
    drone = np.sin(2.0 * np.pi * 110.0 * i_arr / sr) * (0.8 + 0.2 * lfo)
    result += audio * 0.5 + drone * 0.1 * (atmosphere / 10.0)

    # Glitch
    if glitch > 0.3:
        chunk_size = int(sr * 0.3)
        if chunk_size > 0:
            i = 0
            while i < length - chunk_size:
                if rng.random() < glitch * 0.3:
                    segment = result[i:i + chunk_size].copy()[::-1]
                    result[i:i + chunk_size] = segment
                i += chunk_size * 2

    return result


def apply_squarepusher_style(audio: np.ndarray, sr: int, tempo: float, bass: float,
                             complexity: int, scale: str, rng: np.random.Generator):
    length = len(audio)
    result = np.zeros_like(audio, dtype=np.float32)

    bpm = 170.0 * tempo
    beat_samples = int(sr * 60.0 / bpm)
    if beat_samples <= 0:
        beat_samples = int(sr * 60.0 / 170.0)

    pattern = [0, 0.5, 1, 1.25, 1.75, 2, 2.5, 2.75, 3, 3.75]
    num_bars = int(length / (beat_samples * 4)) + 1

    for bar in range(num_bars):
        bar_start = bar * beat_samples * 4
        if bar_start >= length:
            break

        # Kicks
        for pos in pattern:
            idx = int(bar_start + pos * beat_samples)
            if idx >= length:
                continue
            kick_len = int(sr * 0.1)
            end = min(idx + kick_len, length)
            n = end - idx
            if n <= 0:
                continue
            t = np.arange(n, dtype=np.float32) / sr
            kick = np.sin(2.0 * np.pi * 55.0 * t) * np.exp(-25.0 * t)
            result[idx:end] += kick * 0.7

        # Snares
        for pos in [0.5, 1.5, 2.5, 3.5]:
            idx = int(bar_start + pos * beat_samples)
            if idx >= length:
                continue
            snare_len = int(sr * 0.12)
            end = min(idx + snare_len, length)
            n = end - idx
            if n <= 0:
                continue
            noise = (rng.random(n).astype(np.float32) - 0.5) * 0.4
            result[idx:end] += noise

        # Hihats
        for i in range(32):
            idx = int(bar_start + i * beat_samples / 8.0)
            if idx >= length:
                continue
            hihat_len = int(sr * 0.03)
            end = min(idx + hihat_len, length)
            n = end - idx
            if n <= 0:
                continue
            noise = (rng.random(n).astype(np.float32) - 0.5) * 0.08
            result[idx:end] += noise

    # Bass line
    bass_notes = get_scale_notes(40, scale, 12)
    note_length = length // len(bass_notes)

    for i, note in enumerate(bass_notes):
        start = i * note_length
        end = min((i + 1) * note_length, length)
        if start >= length:
            break
        n = end - start
        if n <= 0:
            continue
        t = np.arange(n, dtype=np.float32) / sr
        freq = 440.0 * (2.0 ** ((note - 69.0) / 12.0))
        envelope = np.exp(-2.0 * t / (note_length / sr + 1e-6))
        saw = 2.0 * (t * freq - np.floor(t * freq + 0.5))
        sub = np.sin(2.0 * np.pi * freq * 0.5 * t)
        result[start:end] += (0.5 * saw + 0.3 * sub) * envelope * bass * 0.15

    # Optional melody
    if complexity > 5:
        melody_notes = get_scale_notes(64, scale, 16)
        melody_length = length // len(melody_notes)
        for i, note in enumerate(melody_notes):
            start = i * melody_length
            end = min((i + 1) * melody_length, length)
            if start >= length:
                break
            n = end - start
            if n <= 0:
                continue
            t = np.arange(n, dtype=np.float32) / sr
            freq = 440.0 * (2.0 ** ((note - 69.0) / 12.0))
            envelope = np.exp(-3.0 * t / (melody_length / sr + 1e-6))
            sine = np.sin(2.0 * np.pi * freq * t)
            result[start:end] += sine * envelope * 0.1

    # Add original
    result += audio * 0.25

    return result


def generate_idm(
    file_path,
    style,
    tempo,
    glitch,
    bass,
    duration,
    seed,
    scale,
    complexity,
    atmosphere,
):
    if file_path is None:
        return None

    # Seed
    seed_int = int(seed) if seed is not None else 42
    rng = np.random.default_rng(seed_int)

    # Load audio (mono)
    audio, sr = librosa.load(file_path, sr=None, mono=True)

    if len(audio) == 0:
        raise ValueError("Audio file is empty")

    # Target length (samples)
    target_length = int(sr * float(duration) * 60.0)
    if target_length <= 0:
        target_length = len(audio)

    # Adjust length (repeat or cut)
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        reps = int(np.ceil(target_length / len(audio)))
        audio = np.tile(audio, reps)[:target_length]

    audio = audio.astype(np.float32)

    # Apply style
    if style == "Aphex Twin":
        processed = apply_aphex_style(
            audio, sr, float(tempo), float(glitch),
            float(atmosphere), scale, rng
        )
    else:
        processed = apply_squarepusher_style(
            audio, sr, float(tempo), float(bass),
            int(complexity), scale, rng
        )

    # Normalize
    max_val = np.max(np.abs(processed))
    if max_val > 0:
        processed = processed * (0.95 / max_val)

    return (sr, processed)


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown(
            "# IDM Generator (Python)\nUpload an audio file and generate Aphex Twin / Squarepusher style IDM."
        )

        with gr.Row():
            with gr.Column():
                audio_in = gr.Audio(
                    label="Input audio",
                    type="filepath"
                )
                style = gr.Radio(
                    ["Aphex Twin", "Squarepusher"],
                    value="Aphex Twin",
                    label="Style"
                )
                scale = gr.Dropdown(
                    list(SCALES.keys()),
                    value="minor",
                    label="Scale"
                )
            with gr.Column():
                tempo = gr.Slider(
                    minimum=0.8,
                    maximum=2.5,
                    value=1.2,
                    step=0.1,
                    label="Tempo multiplier"
                )
                glitch = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Glitch amount"
                )
                bass = gr.Slider(
                    minimum=0.5,
                    maximum=3.0,
                    value=1.5,
                    step=0.1,
                    label="Bass boost (Squarepusher style)"
                )
                duration = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=3,
                    step=1,
                    label="Output duration (minutes)"
                )
                complexity = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Complexity (Squarepusher style)"
                )
                atmosphere = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Atmosphere (Aphex style)"
                )
                seed = gr.Number(
                    value=42,
                    label="Random seed"
                )

        generate_btn = gr.Button("Generate IDM track")
        audio_out = gr.Audio(
            label="Generated IDM",
            type="numpy"
        )

        generate_btn.click(
            fn=generate_idm,
            inputs=[
                audio_in,
                style,
                tempo,
                glitch,
                bass,
                duration,
                seed,
                scale,
                complexity,
                atmosphere,
            ],
            outputs=audio_out,
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()

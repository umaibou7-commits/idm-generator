import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io
import tempfile
import os
import glob

# =========================
# 音階定義
# =========================
SCALES = {
    "minor": [0, 2, 3,5, 7, 8, 10],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

# =========================
# DnB キック＆スネアパターン
# =========================
DNB_PATTERNS = {
    "クラシック2ステップ": {
        "kicks": [0.0, 1.5],
        "snares": [1.0, 3.0],
    },
    "ローラー": {
        "kicks": [0.0, 1.25, 2.75],
        "snares": [1.0, 3.0],
    },
    "アーメン風": {
        "kicks": [0.0, 1.5, 2.0],
        "snares": [1.0, 2.75],
    },
    "ハーフタイムドロップ": {
        "kicks": [0.0, 2.0],
        "snares": [2.5],
    },
    "マイクロステップ": {
        "kicks": [0.0, 1.75, 2.25],
        "snares": [1.0, 3.0],
    },
}


def get_scale_notes(root_midi, scale_name, count):
    scale_intervals = SCALES.get(scale_name, SCALES["minor"])
    notes = []
    for i in range(count):
        octave = i // len(scale_intervals)
        interval_idx = i % len(scale_intervals)
        notes.append(root_midi + octave * 12 + scale_intervals[interval_idx])
    return np.array(notes, dtype=np.float32)


# =========================
# ソース音声の解析（キック / 声ネタ候補）
# =========================
def analyze_source_for_percussion_and_voice(audio, sr):
    hop_length = 512
    n_fft = 2048

    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )

    if onset_frames.size == 0:
        frames = np.arange(0, int(len(audio) / hop_length))
    else:
        frames = onset_frames

    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, hop_length=hop_length, n_fft=n_fft
    )[0]
    rms = librosa.feature.rms(
        y=audio, frame_length=n_fft, hop_length=hop_length
    )[0]

    min_len = min(len(centroid), len(rms))
    centroid = centroid[:min_len]
    rms = rms[:min_len]

    energy_mean = float(np.mean(rms) + 1e-6)
    cent_mean = float(np.mean(centroid) + 1e-6)

    kick_idx = []
    voice_idx = []
    for f in frames:
        if f >= min_len:
            continue
        e = rms[f]
        c = centroid[f]
        if e > energy_mean * 1.2 and c < cent_mean * 0.7:
            kick_idx.append(f)
        elif e > energy_mean * 0.9 and c > cent_mean * 1.2:
            voice_idx.append(f)

    kick_idx = np.array(kick_idx, dtype=int)
    voice_idx = np.array(voice_idx, dtype=int)

    onset_samples_all = librosa.frames_to_samples(frames, hop_length=hop_length)
    kick_samples = (
        librosa.frames_to_samples(kick_idx, hop_length=hop_length)
        if kick_idx.size > 0
        else np.array([], dtype=int)
    )
    voice_samples = (
        librosa.frames_to_samples(voice_idx, hop_length=hop_length)
        if voice_idx.size > 0
        else np.array([], dtype=int)
    )

    return {
        "onset_samples": onset_samples_all,
        "kick_samples": kick_samples,
        "voice_samples": voice_samples,
    }


# =========================
# グラニュラー処理
# =========================
def make_granular_layer(
    audio,
    sr,
    rng,
    chop_amount=0.7,
    texture_amount=0.7,
    onset_samples=None,
):
    length = len(audio)
    if length < int(sr * 0.3):
        return np.zeros_like(audio, dtype=np.float32)

    grain_ms = 80.0 + 80.0 * float(chop_amount)
    grain_len = int(sr * grain_ms / 1000.0)
    grain_len = max(32, min(grain_len, length // 4))

    overlap = 0.3 + 0.5 * float(texture_amount)
    hop = int(grain_len * (1.0 - overlap))
    hop = max(8, hop)

    if onset_samples is not None and onset_samples.size > 0:
        candidates = onset_samples.copy()
    else:
        candidates = np.arange(0, length - grain_len, hop, dtype=int)

    candidates = candidates[candidates < length - grain_len]
    if candidates.size < 8:
        candidates = np.arange(0, length - grain_len, hop, dtype=int)

    out = np.zeros(length, dtype=np.float32)
    env = np.hanning(grain_len).astype(np.float32)

    density = 0.6 + 1.4 * float(chop_amount)
    effective_hop = max(4, int(hop / density))

    pos = 0
    while pos < length:
        src_start = int(rng.choice(candidates))
        if src_start + grain_len >= length:
            src_start = max(0, length - grain_len - 1)

        g = audio[src_start : src_start + grain_len].astype(np.float32)

        if rng.random() < 0.5 * chop_amount:
            g = g[::-1]

        if rng.random() < 0.4 * chop_amount:
            levels = int(8 + 8 * (1.0 - texture_amount))
            if levels > 1:
                g = np.round(g * levels) / levels

        gain = 0.4 + 0.8 * rng.random()
        g = g * env * gain

        end = min(pos + grain_len, length)
        out[pos:end] += g[: end - pos]

        pos += effective_hop

    return out


# =========================
# Aphex Twin 風
# =========================
def apply_aphex_style(length, sr, tempo, scale, glitch, atmosphere, rng):
    result = np.zeros(length, dtype=np.float32)
    bass_notes = get_scale_notes(36, scale, 8)
    note_length = max(1, length // len(bass_notes))

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
        result[start:end] += (0.7 * sine + 0.3 * sub) * envelope * 0.18

    bpm = 120.0 * float(tempo)
    beat_samples = int(sr * 60.0 / bpm)
    if beat_samples <= 0:
        beat_samples = int(sr * 60.0 / 120.0)

    for i in range(0, length, beat_samples):
        kick_len = int(sr * 0.16)
        end = min(i + kick_len, length)
        n = end - i
        if n <= 0:
            continue
        t = np.arange(n, dtype=np.float32) / sr
        kick = np.sin(2.0 * np.pi * 50.0 * t) * np.exp(-14.0 * t)
        result[i:end] += kick * 0.55

    i_arr = np.arange(length, dtype=np.float32)
    lfo_slow = np.sin(2.0 * np.pi * 0.25 * i_arr / sr)
    lfo_fast = np.sin(2.0 * np.pi * 0.7 * i_arr / sr)

    drone_freq = 110.0
    drone = np.sin(2.0 * np.pi * drone_freq * i_arr / sr) * (0.7 + 0.3 * lfo_slow)
    pad = np.sin(2.0 * np.pi * drone_freq * 0.5 * i_arr / sr) * (0.5 + 0.5 * lfo_fast)

    amb = drone * 0.15 + pad * 0.12
    amb_scale = 0.3 + 0.7 * (float(atmosphere) / 10.0)
    result += amb * amb_scale

    if glitch > 0.2:
        chunk_size = int(sr * (0.18 + 0.2 * glitch))
        if chunk_size > 0:
            i = 0
            while i < length - chunk_size:
                if rng.random() < glitch * 0.35:
                    seg = result[i : i + chunk_size].copy()
                    if rng.random() < 0.5:
                        seg = seg[::-1]
                    if rng.random() < 0.5:
                        seg *= 0.4 + 0.6 * rng.random()
                    result[i : i + chunk_size] = seg
                i += chunk_size

    return result


# =========================
# Squarepusher 風
# =========================
def apply_squarepusher_style(length, sr, tempo, scale, bass, complexity, rng):
    result = np.zeros(length, dtype=np.float32)

    bpm = 170.0 * float(tempo)
    beat_samples = int(sr * 60.0 / bpm)
    if beat_samples <= 0:
        beat_samples = int(sr * 60.0 / 170.0)

    pattern = [0, 0.5, 1, 1.25, 1.75, 2, 2.5, 2.75, 3, 3.75]
    num_bars = int(length / (beat_samples * 4)) + 1

    for bar in range(num_bars):
        bar_start = bar * beat_samples * 4
        if bar_start >= length:
            break

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

    bass_notes = get_scale_notes(40, scale, 12)
    note_length = max(1, length // len(bass_notes))

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
        result[start:end] += (0.5 * saw + 0.3 * sub) * envelope * float(bass) * 0.15

    if complexity > 5:
        melody_notes = get_scale_notes(64, scale, 16)
        melody_length = max(1, length // len(melody_notes))
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

    return result


# =========================
# 新しい 303 風アシッドシンセ
# =========================
def apply_acid_303(length, sr, tempo, scale, acid_amount, rng):
    if acid_amount <= 0.0:
        return np.zeros(length, dtype=np.float32)

    acid_amount = float(acid_amount)

    # 想定BPMと16分音符の長さ
    bpm = 130.0 * tempo
    seconds_per_beat = 60.0 / max(1e-3, bpm)
    samples_per_16th = int(sr * seconds_per_beat / 4.0)
    if samples_per_16th <= 0:
        samples_per_16th = int(sr * 60.0 / (130.0 * 4.0))

    total_steps = max(8, length // samples_per_16th)
    # ベースとなるスケールから 303 用の音階
    scale_notes = get_scale_notes(48, scale, 8)  # 低めのレンジ
    pattern_notes = []
    accents = []
    slides = []
    gates = []

    for step in range(total_steps):
        # ある程度同じトニック周りをウロウロさせる
        base_note = float(scale_notes[step % len(scale_notes)])
        # たまにオクターブ上
        if rng.random() < 0.25:
            base_note += 12.0
        # たまにオクターブ下
        if rng.random() < 0.1:
            base_note -= 12.0
        pattern_notes.append(base_note)

        # accent / slide / gate
        accents.append(rng.random() < 0.3)
        slides.append(rng.random() < 0.25)
        # ゲート長（1.0でフル、0.4くらいまで短く）
        gates.append(0.4 + 0.6 * rng.random())

    pattern_notes = np.array(pattern_notes, dtype=np.float32)
    accents = np.array(accents, dtype=bool)
    slides = np.array(slides, dtype=bool)
    gates = np.array(gates, dtype=np.float32)

    out = np.zeros(length, dtype=np.float32)

    phase = 0.0
    two_pi = 2.0 * np.pi

    for step in range(total_steps):
        start = step * samples_per_16th
        if start >= length:
            break
        end = min((step + 1) * samples_per_16th, length)
        n = end - start
        if n <= 0:
            continue

        gate = gates[step]
        gate_n = int(n * gate)
        if gate_n <= 0:
            continue

        # ノートの開始・終了周波数
        freq_start = 440.0 * (2.0 ** ((pattern_notes[step] - 69.0) / 12.0))
        if step < total_steps - 1 and slides[step]:
            freq_end = 440.0 * (2.0 ** ((pattern_notes[step + 1] - 69.0) / 12.0))
        else:
            freq_end = freq_start

        freqs = np.linspace(freq_start, freq_end, gate_n, dtype=np.float32)
        phase_increments = two_pi * freqs / sr

        # フェーズの累積（stepごとに連続させる）
        local_phase = phase + np.cumsum(phase_increments)
        phase = float(local_phase[-1])

        # 303は鋸歯波なので saw を生成
        saw = 2.0 * (local_phase / two_pi - np.floor(local_phase / two_pi + 0.5))

        # エンベロープ：アタック早め、リリース中程度
        t = np.linspace(0.0, 1.0, gate_n, dtype=np.float32)
        env = np.exp(-4.0 * t)
        env[: max(1, int(0.05 * gate_n))] *= np.linspace(0.0, 1.0, max(1, int(0.05 * gate_n)))

        # アクセントで音量と「歪み」を強める
        accent_gain = 1.0
        drive = 2.0 + 4.0 * acid_amount
        if accents[step]:
            accent_gain = 1.3 + 0.7 * acid_amount
            drive *= 1.3

        wave = np.tanh(saw * drive) * env * accent_gain

        # 簡易レゾナンスっぽいEQ（高域少し持ち上げる）
        hp = np.zeros_like(wave)
        hp[1:] = wave[1:] - 0.98 * wave[:-1]
        wave = 0.6 * wave + 0.7 * hp

        out[start : start + gate_n] += wave * (0.4 + 0.6 * acid_amount)

    return out


# =========================
# アシッド・ハウス寄り
# =========================
def apply_acid_house_core(length, sr, tempo, scale, bass, rng):
    result = np.zeros(length, dtype=np.float32)

    bpm = 130.0 * float(tempo)
    beat_samples = int(sr * 60.0 / bpm)
    if beat_samples <= 0:
        beat_samples = int(sr * 60.0 / 130.0)

    num_bars = int(length / (beat_samples * 4)) + 1

    for bar in range(num_bars):
        bar_start = bar * beat_samples * 4
        if bar_start >= length:
            break

        # 4つ打ちキック
        for beat in range(4):
            idx = int(bar_start + beat * beat_samples)
            if idx >= length:
                continue
            kick_len = int(sr * 0.18)
            end = min(idx + kick_len, length)
            n = end - idx
            if n <= 0:
                continue
            t = np.arange(n, dtype=np.float32) / sr
            kick = np.sin(2.0 * np.pi * 48.0 * t) * np.exp(-12.0 * t)
            result[idx:end] += kick * 0.8

        # クラップ 2拍目 & 4拍目
        for beat in [1, 3]:
            idx = int(bar_start + beat * beat_samples)
            if idx >= length:
                continue
            clap_len = int(sr * 0.12)
            end = min(idx + clap_len, length)
            n = end - idx
            if n <= 0:
                continue
            noise = (rng.random(n).astype(np.float32) - 0.5)
            env = np.exp(-15.0 * np.arange(n, dtype=np.float32) / sr)
            result[idx:end] += noise * env * 0.6

        # ハイハット（8分）
        for i in range(8):
            idx = int(bar_start + i * beat_samples / 2.0)
            if idx >= length:
                continue
            hat_len = int(sr * 0.04)
            end = min(idx + hat_len, length)
            n = end - idx
            if n <= 0:
                continue
            noise = (rng.random(n).astype(np.float32) - 0.5)
            env = np.exp(-40.0 * np.arange(n, dtype=np.float32) / sr)
            result[idx:end] += noise * env * 0.3

    # シンプルなサブベース
    bass_notes = get_scale_notes(36, scale, 4)
    bar_len = max(1, length // max(1, len(bass_notes)))
    for i, note in enumerate(bass_notes):
        start = i * bar_len
        end = min((i + 1) * bar_len, length)
        if start >= length:
            break
        n = end - start
        if n <= 0:
            continue
        t = np.arange(n, dtype=np.float32) / sr
        freq = 440.0 * (2.0 ** ((note - 69.0) / 12.0))
        env = np.exp(-2.0 * t / (bar_len / sr + 1e-6))
        sine = np.sin(2.0 * np.pi * freq * t)
        result[start:end] += sine * env * float(bass) * 0.2

    return result


# =========================
# Drum'n'Bass コア
# =========================
def apply_dnb_core(length, sr, tempo, scale, bass, complexity, pattern_name, rng):
    result = np.zeros(length, dtype=np.float32)

    bpm = 172.0 * float(tempo)
    beat_samples = int(sr * 60.0 / bpm)
    if beat_samples <= 0:
        beat_samples = int(sr * 60.0 / 172.0)

    pattern_def = DNB_PATTERNS.get(pattern_name, DNB_PATTERNS["クラシック2ステップ"])
    base_kicks = pattern_def["kicks"]
    base_snares = pattern_def["snares"]

    num_bars = int(length / (beat_samples * 4)) + 1

    for bar in range(num_bars):
        bar_start = bar * beat_samples * 4
        if bar_start >= length:
            break

        bar_offset = 0.0
        if complexity >= 7 and bar % 2 == 1:
            bar_offset = 0.125

        kicks = base_kicks
        snares = base_snares

        for pos in kicks:
            pos_eff = pos + (bar_offset if rng.random() < 0.4 else 0.0)
            idx = int(bar_start + pos_eff * beat_samples)
            if idx >= length:
                continue
            kick_len = int(sr * 0.08)
            end = min(idx + kick_len, length)
            n = end - idx
            if n <= 0:
                continue
            t = np.arange(n, dtype=np.float32) / sr
            osc = np.sin(2.0 * np.pi * 65.0 * t) * np.exp(-18.0 * t)
            click = (rng.random(n).astype(np.float32) - 0.5) * np.exp(-80.0 * t)
            kick = osc * 0.9 + click * 0.3
            result[idx:end] += kick * 0.9

        for pos in snares:
            pos_eff = pos + (bar_offset if rng.random() < 0.3 else 0.0)
            idx = int(bar_start + pos_eff * beat_samples)
            if idx >= length:
                continue
            snare_len = int(sr * 0.12)
            end = min(idx + snare_len, length)
            n = end - idx
            if n <= 0:
                continue
            t = np.arange(n, dtype=np.float32) / sr
            tone = np.sin(2.0 * np.pi * 200.0 * t) * np.exp(-20.0 * t)
            noise = (rng.random(n).astype(np.float32) - 0.5) * np.exp(-25.0 * t)
            snare = tone * 0.4 + noise * 0.9
            result[idx:end] += snare * 0.85

        if complexity >= 6:
            ghost_positions = [p + 0.25 for p in kicks] + [p - 0.25 for p in snares]
            for pos in ghost_positions:
                if pos <= 0 or pos >= 4:
                    continue
                idx = int(bar_start + pos * beat_samples)
                if idx >= length:
                    continue
                glen = int(sr * 0.05)
                end = min(idx + glen, length)
                n = end - idx
                if n <= 0:
                    continue
                t = np.arange(n, dtype=np.float32) / sr
                env = np.exp(-30.0 * t)
                noise = (rng.random(n).astype(np.float32) - 0.5) * env
                result[idx:end] += noise * 0.25

        for i in range(16):
            idx = int(bar_start + i * beat_samples / 4.0)
            if idx >= length:
                continue
            hat_len = int(sr * 0.03)
            end = min(idx + hat_len, length)
            n = end - idx
            if n <= 0:
                continue
            t = np.arange(n, dtype=np.float32) / sr
            noise = (rng.random(n).astype(np.float32) - 0.5)
            env = np.exp(-60.0 * t)
            amp = 0.25 + 0.1 * np.sin(2.0 * np.pi * i / 16.0)
            result[idx:end] += noise * env * amp

    bass_notes = get_scale_notes(38, scale, num_bars)
    bar_len = max(1, length // max(1, len(bass_notes)))
    for i, note in enumerate(bass_notes):
        start = i * bar_len
        end = min((i + 1) * bar_len, length)
        if start >= length:
            break
        n = end - start
        if n <= 0:
            continue
        t = np.arange(n, dtype=np.float32) / sr
        freq = 440.0 * (2.0 ** ((note - 69.0) / 12.0))
        detune = freq * 0.01
        osc1 = np.sin(2.0 * np.pi * (freq - detune) * t)
        osc2 = np.sin(2.0 * np.pi * (freq + detune) * t)
        env = np.exp(-1.5 * t / (bar_len / sr + 1e-6))
        bass_sig = (osc1 + osc2) * 0.5
        bass_sig = np.tanh(bass_sig * 2.0)
        result[start:end] += bass_sig * env * float(bass) * 0.22

    return result


# =========================
# 元音源からキックを抜き出す
# =========================
def make_kick_layer_from_source(length, sr, audio, kick_positions, bpm, strength, rng):
    if strength <= 0.0 or kick_positions is None or len(kick_positions) == 0:
        return np.zeros(length, dtype=np.float32)

    base_pos = int(kick_positions[0])
    sample_len = int(sr * 0.25)
    start = max(0, base_pos - int(0.02 * sr))
    end = min(len(audio), start + sample_len)
    if end - start <= 16:
        return np.zeros(length, dtype=np.float32)

    kick_sample = audio[start:end].astype(np.float32)
    env = np.hanning(len(kick_sample)).astype(np.float32)
    kick_sample = kick_sample * env

    layer = np.zeros(length, dtype=np.float32)
    beat_samples = int(sr * 60.0 / bpm)
    if beat_samples <= 0:
        beat_samples = int(sr * 60.0 / 120.0)

    pos = 0
    while pos < length:
        end_pos = min(pos + len(kick_sample), length)
        layer[pos:end_pos] += kick_sample[: end_pos - pos] * strength
        pos += beat_samples

    return layer


# =========================
# 声ネタレイヤー
# =========================
def make_voice_layer_from_source(audio, sr, length, voice_positions, level, rng):
    if level <= 0.0 or voice_positions is None or len(voice_positions) == 0:
        return np.zeros(length, dtype=np.float32)

    layer = np.zeros(length, dtype=np.float32)

    num_snippets = min(len(voice_positions), 12)
    chosen = rng.choice(voice_positions, size=num_snippets, replace=False)

    min_len = int(sr * 0.25)
    max_len = int(sr * 0.6)
    kernel = np.ones(64, dtype=np.float32) / 64.0

    for pos in chosen:
        snip_len = int(rng.integers(min_len, max_len))
        start_src = int(max(0, pos - snip_len // 4))
        end_src = int(min(len(audio), start_src + snip_len))
        snip = audio[start_src:end_src].astype(np.float32)
        if len(snip) <= 32:
            continue

        smooth = np.convolve(snip, kernel, mode="same")
        snip_hp = snip - smooth

        env = np.hanning(len(snip_hp)).astype(np.float32)
        snip_hp = snip_hp * env

        dest_pos = int(rng.integers(0, max(1, length - len(snip_hp))))
        end_dest = dest_pos + len(snip_hp)
        layer[dest_pos:end_dest] += snip_hp * level

    return layer


# =========================
# アレンジ・ミックス
# =========================
def arrange_layers(core, granular, original, acid, voice, sr, bpm_for_sidechain=None, sidechain_depth=0.0):
    length = len(core)
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)

    core_gain = np.ones(length, dtype=np.float32)
    granular_gain = np.ones(length, dtype=np.float32)
    acid_gain = np.ones(length, dtype=np.float32)
    voice_gain = np.ones(length, dtype=np.float32)

    intro = (t < 0.2)
    groove1 = (t >= 0.2) & (t < 0.5)
    breakd = (t >= 0.5) & (t < 0.7)
    groove2 = (t >= 0.7)

    core_gain[intro] *= 0.3
    granular_gain[intro] *= 0.7
    acid_gain[intro] *= 0.5
    voice_gain[intro] *= 0.3

    core_gain[groove1] *= 1.0
    granular_gain[groove1] *= 0.9
    acid_gain[groove1] *= 0.9
    voice_gain[groove1] *= 0.4

    core_gain[breakd] *= 0.3
    granular_gain[breakd] *= 0.8
    acid_gain[breakd] *= 0.6
    voice_gain[breakd] *= 1.0

    core_gain[groove2] *= 1.1
    granular_gain[groove2] *= 0.9
    acid_gain[groove2] *= 1.0
    voice_gain[groove2] *= 0.5

    mixed = (
        core * core_gain
        + granular * granular_gain
        + original
        + acid * acid_gain
        + voice * voice_gain
    )

    if bpm_for_sidechain is not None and sidechain_depth > 0.0:
        beat_samples = int(sr * 60.0 / bpm_for_sidechain)
        if beat_samples > 0:
            env_beat = 0.3 + 0.7 * (np.linspace(0.0, 1.0, beat_samples, dtype=np.float32) ** 2)
            side_env = np.tile(env_beat, int(np.ceil(length / beat_samples)))[:length]
            mixed = mixed * (1.0 - sidechain_depth + sidechain_depth * side_env)

    fade_in = np.clip(t / 0.05, 0.0, 1.0)
    fade_out = np.clip((1.0 - t) / 0.1, 0.0, 1.0)
    env_master = np.minimum(fade_in, fade_out)

    return mixed * env_master


# =========================
# メイン生成ロジック
# =========================
def generate_idm_array(
    audio,
    sr,
    style,
    tempo,
    glitch,
    bass,
    duration_min,
    seed,
    scale,
    complexity,
    atmosphere,
    idm_mode,
    acid_amount,
    voice_amount,
    dnb_pattern_name,
):
    if audio.size == 0:
        raise ValueError("音声データが空です。")

    rng = np.random.default_rng(int(seed))

    max_in = float(np.max(np.abs(audio)))
    if max_in > 0:
        audio = (audio / max_in).astype(np.float32)
    else:
        audio = audio.astype(np.float32)

    target_length = int(sr * float(duration_min) * 60.0)
    if target_length <= 0:
        target_length = len(audio)

    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        reps = int(np.ceil(target_length / len(audio)))
        audio = np.tile(audio, reps)[:target_length]

    length = len(audio)

    analysis = analyze_source_for_percussion_and_voice(audio, sr)

    tempo_eff = float(tempo)
    glitch_eff = float(glitch)
    complexity_eff = int(complexity)
    bass_eff = float(bass)
    atmosphere_eff = int(atmosphere)
    acid_eff = float(acid_amount)

    if idm_mode == "アンビエント寄り":
        tempo_eff *= 0.8
        glitch_eff *= 0.6
        complexity_eff = max(1, complexity_eff - 2)
        atmosphere_eff = min(10, atmosphere_eff + 2)
        acid_eff *= 0.7
    elif idm_mode == "ブレイクコア寄り":
        tempo_eff *= 1.4
        glitch_eff = min(1.0, glitch_eff * 1.4 + 0.1)
        complexity_eff = min(10, complexity_eff + 2)
        bass_eff *= 1.2
        acid_eff = min(1.0, acid_eff * 1.4 + 0.1)

    if style == "Aphex Twin":
        base_bpm = 120.0 * tempo_eff
    elif style == "Squarepusher":
        base_bpm = 170.0 * tempo_eff
    elif style == "Acid House":
        base_bpm = 130.0 * tempo_eff
    else:
        base_bpm = 172.0 * tempo_eff

    granular = make_granular_layer(
        audio,
        sr,
        rng,
        chop_amount=glitch_eff,
        texture_amount=float(atmosphere_eff) / 10.0,
        onset_samples=analysis["onset_samples"],
    )

    if style == "Aphex Twin":
        core = apply_aphex_style(
            length,
            sr,
            tempo=tempo_eff,
            scale=scale,
            glitch=glitch_eff,
            atmosphere=atmosphere_eff,
            rng=rng,
        )
    elif style == "Squarepusher":
        core = apply_squarepusher_style(
            length,
            sr,
            tempo=tempo_eff,
            scale=scale,
            bass=bass_eff,
            complexity=complexity_eff,
            rng=rng,
        )
    elif style == "Acid House":
        core = apply_acid_house_core(
            length,
            sr,
            tempo=tempo_eff,
            scale=scale,
            bass=bass_eff,
            rng=rng,
        )
    else:
        core = apply_dnb_core(
            length,
            sr,
            tempo=tempo_eff,
            scale=scale,
            bass=bass_eff,
            complexity=complexity_eff,
            pattern_name=dnb_pattern_name,
            rng=rng,
        )

    kick_layer = make_kick_layer_from_source(
        length,
        sr,
        audio,
        analysis["kick_samples"],
        bpm=base_bpm,
        strength=0.5 if style != "Drum'n'Bass" else 0.2,
        rng=rng,
    )
    core = core + kick_layer

    original_level = 0.08 + 0.12 * (1.0 - glitch_eff)
    original_layer = audio * original_level

    acid_layer = apply_acid_303(
        length,
        sr,
        tempo=tempo_eff,
        scale=scale,
        acid_amount=acid_eff,
        rng=rng,
    )

    voice_layer = make_voice_layer_from_source(
        audio,
        sr,
        length,
        analysis["voice_samples"],
        level=float(voice_amount),
        rng=rng,
    )

    processed = arrange_layers(
        core,
        granular,
        original_layer,
        acid_layer,
        voice_layer,
        sr,
        bpm_for_sidechain=base_bpm,
        sidechain_depth=0.25 if style in ("Acid House", "Squarepusher", "Drum'n'Bass") else 0.1,
    )

    max_val = float(np.max(np.abs(processed)))
    if max_val > 0:
        processed = processed * (0.98 / max_val)

    return sr, processed.astype(np.float32)


# =========================
# Streamlit UI
# =========================
def main():
    st.set_page_config(
        page_title="IDM Generator",
        page_icon="🎛️",
        layout="wide",
    )

    st.title("IDM Generator (Streamlit)")
    st.caption(
        "アップロードした曲や YouTube の音源を素材に、Aphex Twin / Squarepusher / Acid House / Drum'n'Bass 風のIDMトラックに自動変換します。"
    )

    # セッション状態（YouTube音源保管）
    if "yt_audio" not in st.session_state:
        st.session_state["yt_audio"] = None
        st.session_state["yt_sr"] = None
        st.session_state["yt_info"] = None

    col_left, col_right = st.columns([1, 1.4])

    with col_left:
        uploaded_file = st.file_uploader(
            "元になる音源ファイル（WAV / MP3 など）",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
        )

        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/*")

        st.markdown("### または YouTube から読み込む")

        yt_url = st.text_input(
            "YouTube の URL",
            placeholder="https://www.youtube.com/watch?v=...",
        )
        yt_col1, yt_col2 = st.columns([1, 1])
        with yt_col1:
            yt_load = st.button("YouTube から音声を取得")
        with yt_col2:
            if not YT_AVAILABLE:
                st.info("YouTube機能を使うには requirements.txt に `yt-dlp` を追加してください。")

        if yt_load:
            if not yt_url.strip():
                st.warning("YouTube の URL を入力してください。")
            elif not YT_AVAILABLE:
                st.error("yt_dlp が利用できません。`yt-dlp` をインストールしてください。")
            else:
                with st.spinner("YouTube から音声をダウンロード中..."):
                    try:
                        yt_audio, yt_sr = load_audio_from_youtube(yt_url)
                        st.session_state["yt_audio"] = yt_audio
                        st.session_state["yt_sr"] = yt_sr
                        st.session_state["yt_info"] = yt_url
                        # プレビュー用に一瞬書き出して再生
                        preview_buf = io.BytesIO()
                        sf.write(preview_buf, yt_audio[: min(len(yt_audio), yt_sr * 30)], yt_sr, format="WAV")
                        preview_buf.seek(0)
                        st.success("YouTube 音源を読み込みました。")
                        st.audio(preview_buf.read(), format="audio/wav")
                    except Exception as e:
                        st.error(f"YouTube 音源の取得に失敗しました: {e}")

        if st.session_state["yt_audio"] is not None and uploaded_file is None:
            st.markdown("✅ 現在のソース: YouTube 音源")

        style = st.radio(
            "ベースとなるスタイル",
            ["Aphex Twin", "Squarepusher", "Acid House", "Drum'n'Bass"],
        )

        idm_mode = st.selectbox(
            "IDMモード（ざっくりキャラクター）",
            ["標準", "アンビエント寄り", "ブレイクコア寄り"],
            help="標準: バランス型 / アンビエント寄り: テンポ遅め・空間多め / ブレイクコア寄り: テンポ速め・グリッチ多め",
        )

        scale = st.selectbox(
            "スケール（曲の雰囲気になる音階）",
            options=list(SCALES.keys()),
            index=0,
            help="minor: 暗め / major: 明るめ / pentatonic: 和風っぽい / それ以外は少しマニアックなモード音階です。",
        )

        seed = st.number_input(
            "ランダムシード（同じ値なら同じ展開）",
            min_value=0,
            max_value=99999,
            value=42,
            step=1,
        )

        dnb_pattern_name = st.selectbox(
            "D'n'Bキック＆スネアパターン（Drum'n'Bassスタイル時のみ有効）",
            options=list(DNB_PATTERNS.keys()),
            index=0,
            help="クラシック2ステップ: ベーシックな2ステップ / ローラー: 転がるグルーヴ / アーメン風: Break系のノリ など",
        )

    with col_right:
        tempo = st.slider(
            "テンポ（全体の速さ）",
            0.8,
            2.5,
            1.2,
            0.1,
            help="1.0 が標準。値を上げるとブレイクビートやベースが速くなります。",
        )
        glitch = st.slider(
            "サンプリングの刻み具合（グリッチ感）",
            0.0,
            1.0,
            0.6,
            0.05,
            help="原曲をどれくらい細かく切り刻んで再配置するか。高くするほどバグっぽい展開になります。",
        )
        bass = st.slider(
            "ベースの太さ",
            0.5,
            3.0,
            1.5,
            0.1,
            help="特に Squarepusher / Acid House / Drum'n'Bass スタイルの低音の主張に効きます。Aphex スタイルでも少し影響します。",
        )
        complexity = st.slider(
            "リズム / メロディの複雑さ",
            1,
            10,
            7,
            1,
            help="高くすると不規則なブレイクとメロディが増え、よりカオティックなIDMになります。",
        )
        atmosphere = st.slider(
            "空間・アンビエント感",
            1,
            10,
            6,
            1,
            help="ドローンやパッドの厚み。Aphex 系の浮遊感・奥行きに効きます。",
        )
        acid_amount = st.slider(
            "アシッドシンセの存在感（303感）",
            0.0,
            1.0,
            0.6,
            0.1,
            help="0でアシッド無し。値を上げると303っぽいウネウネしたシンセラインが前に出ます。",
        )
        voice_amount = st.slider(
            "声ネタ・ボーカルの飛び道具感",
            0.0,
            1.0,
            0.4,
            0.1,
            help="アップロードした音源の中から『声っぽい部分』を抜き出して散りばめます。0でオフ。",
        )
        duration = st.slider(
            "生成時間（分）",
            1,
            8,
            3,
            1,
            help="出力トラックのおおよその長さです。",
        )

        generate = st.button("IDMトラックを生成する")

    if generate:
        # ソースとなる音源を決定
        source_audio = None
        source_sr = None

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            if not file_bytes:
                st.error("アップロードされたファイルが空です。別のファイルを試してください。")
                return
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                source_audio, source_sr = librosa.load(tmp_path, sr=None, mono=True)
            except Exception as e:
                st.error(f"音声ファイルの読み込みに失敗しました: {e}")
                return
        elif st.session_state["yt_audio"] is not None:
            source_audio = st.session_state["yt_audio"]
            source_sr = st.session_state["yt_sr"]
        else:
            st.warning("先に音源ファイルをアップロードするか、YouTube から読み込んでください。")
            return

        with st.spinner("IDMトラックを生成中..."):
            try:
                sr_out, processed = generate_idm_array(
                    audio=source_audio,
                    sr=source_sr,
                    style=style,
                    tempo=tempo,
                    glitch=glitch,
                    bass=bass,
                    duration_min=duration,
                    seed=seed,
                    scale=scale,
                    complexity=complexity,
                    atmosphere=atmosphere,
                    idm_mode=idm_mode,
                    acid_amount=acid_amount,
                    voice_amount=voice_amount,
                    dnb_pattern_name=dnb_pattern_name,
                )
            except Exception as e:
                st.error(f"トラック生成中にエラーが発生しました: {e}")
                return

            buffer = io.BytesIO()
            sf.write(buffer, processed, sr_out, format="WAV")
            buffer.seek(0)

            st.success("生成完了！")
            audio_bytes = buffer.read()
            st.audio(audio_bytes, format="audio/wav")

            buffer = io.BytesIO(audio_bytes)
            st.download_button(
                "WAV をダウンロード",
                data=buffer,
                file_name=f"idm_{style.replace(' ', '').lower()}_{seed}.wav",
                mime="audio/wav",
            )


if __name__ == "__main__":
    main()

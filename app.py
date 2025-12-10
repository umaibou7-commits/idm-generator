import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io
import tempfile

# -------------------------
# 音階定義
# -------------------------
SCALES = {
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}


def get_scale_notes(root_midi: int, scale_name: str, count: int) -> np.ndarray:
    """スケールと回数から MIDI ノート列を生成"""
    scale_intervals = SCALES.get(scale_name, SCALES["minor"])
    notes = []
    for i in range(count):
        octave = i // len(scale_intervals)
        interval_idx = i % len(scale_intervals)
        notes.append(root_midi + octave * 12 + scale_intervals[interval_idx])
    return np.array(notes, dtype=np.float32)


def apply_aphex_style(
    audio: np.ndarray,
    sr: int,
    tempo: float,
    glitch: float,
    atmosphere: float,
    scale: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Aphex Twin っぽい処理"""
    length = len(audio)
    result = np.zeros_like(audio, dtype=np.float32)

    # ベースノート
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

    # キック
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

    # ドローン＋元音声
    i_arr = np.arange(length, dtype=np.float32)
    lfo = np.sin(2.0 * np.pi * 0.5 * i_arr / sr)
    drone = np.sin(2.0 * np.pi * 110.0 * i_arr / sr) * (0.8 + 0.2 * lfo)
    result += audio * 0.5 + drone * 0.1 * (atmosphere / 10.0)

    # グリッチ
    if glitch > 0.3:
        chunk_size = int(sr * 0.3)
        if chunk_size > 0:
            i = 0
            while i < length - chunk_size:
                if rng.random() < glitch * 0.3:
                    segment = result[i : i + chunk_size].copy()[::-1]
                    result[i : i + chunk_size] = segment
                i += chunk_size * 2

    return result


def apply_squarepusher_style(
    audio: np.ndarray,
    sr: int,
    tempo: float,
    bass: float,
    complexity: int,
    scale: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Squarepusher っぽい処理"""
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

        # キック
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

        # スネア
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

        # ハイハット
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

    # ベースライン
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

    # メロディ
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

    # 元音声を混ぜる
    result += audio * 0.25

    return result


def generate_idm_array(
    audio: np.ndarray,
    sr: int,
    style: str,
    tempo: float,
    glitch: float,
    bass: float,
    duration_min: int,
    seed: int,
    scale: str,
    complexity: int,
    atmosphere: int,
) -> tuple[int, np.ndarray]:
    """波形配列から IDM トラックを生成"""
    rng = np.random.default_rng(int(seed))

    # 目標長（サンプル数）
    target_length = int(sr * float(duration_min) * 60.0)
    if target_length <= 0:
        target_length = len(audio)

    # 長さ調整（カット or ループ）
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        reps = int(np.ceil(target_length / len(audio)))
        audio = np.tile(audio, reps)[:target_length]

    audio = audio.astype(np.float32)

    # スタイル適用
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

    # 正規化（安全な形）
    max_val = float(np.max(np.abs(processed)))
    if max_val > 0:
        processed = processed * (0.95 / max_val)

    return sr, processed.astype(np.float32)


def main():
    st.set_page_config(
        page_title="IDM Generator",
        page_icon="🎛️",
        layout="wide",
    )

    st.title("IDM Generator (Streamlit)")
    st.caption("アップロードした音声から Aphex Twin / Squarepusher 風 IDM トラックを生成します。")

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        uploaded_file = st.file_uploader(
            "音声ファイルをアップロード",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
        )

        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/*")

        style = st.radio("スタイル", ["Aphex Twin", "Squarepusher"])

        scale = st.selectbox(
            "スケール（音階）",
            options=list(SCALES.keys()),
            index=0,
        )

        seed = st.number_input(
            "シード値",
            min_value=0,
            max_value=99999,
            value=42,
            step=1,
        )

    with col_right:
        tempo = st.slider("テンポ倍率", 0.8, 2.5, 1.2, 0.1)
        glitch = st.slider("グリッチ強度（Aphex 用）", 0.0, 1.0, 0.5, 0.05)
        bass = st.slider("ベースブースト（Squarepusher 用）", 0.5, 3.0, 1.5, 0.1)
        complexity = st.slider("複雑さ（Squarepusher 用）", 1, 10, 5, 1)
        atmosphere = st.slider("アトモスフィア（Aphex 用）", 1, 10, 5, 1)
        duration = st.slider("生成時間（分）", 1, 8, 3, 1)

        generate = st.button("IDMトラックを生成する")

    if generate:
        if uploaded_file is None:
            st.warning("先に音声ファイルをアップロードしてください。")
        else:
            with st.spinner("生成中..."):
                # 一時ファイルに保存して librosa で読み込み
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                audio, sr = librosa.load(tmp_path, sr=None, mono=True)

                if audio.size == 0:
                    st.error("音声ファイルが空です。別のファイルを試してください。")
                    return

                sr_out, processed = generate_idm_array(
                    audio,
                    sr,
                    style,
                    tempo,
                    glitch,
                    bass,
                    duration,
                    seed,
                    scale,
                    complexity,
                    atmosphere,
                )

                # WAV バイナリに変換
                buffer = io.BytesIO()
                sf.write(buffer, processed, sr_out, format="WAV")
                buffer.seek(0)

                st.success("生成完了！")
                audio_bytes = buffer.read()
                st.audio(audio_bytes, format="audio/wav")

                # ダウンロード用にもう一度先頭に戻す
                buffer = io.BytesIO(audio_bytes)
                st.download_button(
                    "WAV をダウンロード",
                    data=buffer,
                    file_name=f"idm_{style.replace(' ', '').lower()}_{seed}.wav",
                    mime="audio/wav",
                )


if __name__ == "__main__":
    main()

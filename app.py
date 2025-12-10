import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import io
import tempfile

# =========================
# 音階定義
# =========================
SCALES = {
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "major": [0, 2, 4, 5, 7, 9, 11],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}


def get_scale_notes(root_midi, scale_name, count):
    """スケールと回数から MIDI ノート列を生成"""
    scale_intervals = SCALES.get(scale_name, SCALES["minor"])
    notes = []
    for i in range(count):
        octave = i // len(scale_intervals)
        interval_idx = i % len(scale_intervals)
        notes.append(root_midi + octave * 12 + scale_intervals[interval_idx])
    return np.array(notes, dtype=np.float32)


# =========================
# 原曲を細かく刻んで再構成するグラニュラー処理
# =========================
def make_granular_layer(audio, sr, rng, chop_amount=0.7, texture_amount=0.7):
    """
    原曲を細かく刻んで並び替えるレイヤーを作る。
    chop_amount: 0~1 刻みの激しさ（グリッチ感）
    texture_amount: 0~1 テクスチャの粗さ（ローファイ感）
    """
    length = len(audio)
    if length < int(sr * 0.3):
        # 短すぎるときは何もしない
        return np.zeros_like(audio, dtype=np.float32)

    # 1粒の長さ（80〜160ms）
    grain_ms = 80.0 + 80.0 * float(chop_amount)
    grain_len = int(sr * grain_ms / 1000.0)
    grain_len = max(32, min(grain_len, length // 4))

    # オーバーラップ率 30〜80%
    overlap = 0.3 + 0.5 * float(texture_amount)
    hop = int(grain_len * (1.0 - overlap))
    hop = max(8, hop)

    # オンセット位置（アタックのある場所）を候補にする
    try:
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_frames = librosa.onset.onset_frames(onset_envelope=onset_env)
        onset_samples = librosa.frames_to_samples(onset_frames)
    except Exception:
        onset_samples = np.arange(0, length - grain_len, hop)

    onset_samples = onset_samples[onset_samples < length - grain_len]
    if onset_samples.size < 8:
        onset_samples = np.arange(0, length - grain_len, hop)

    out = np.zeros(length, dtype=np.float32)
    env = np.hanning(grain_len).astype(np.float32)

    # 刻み密度（1〜2倍）
    density = 0.6 + 1.4 * float(chop_amount)
    effective_hop = max(4, int(hop / density))

    pos = 0
    while pos < length:
        src_start = int(rng.choice(onset_samples))
        if src_start + grain_len >= length:
            src_start = max(0, length - grain_len - 1)

        g = audio[src_start : src_start + grain_len].astype(np.float32)

        # ランダム反転
        if rng.random() < 0.5 * chop_amount:
            g = g[::-1]

        # ランダムなローファイ／ビットクラッシュ的処理
        if rng.random() < 0.4 * chop_amount:
            levels = int(8 + 8 * (1.0 - texture_amount))  # 8〜16段階
            if levels > 1:
                g = np.round(g * levels) / levels

        gain = 0.4 + 0.8 * rng.random()
        g = g * env * gain

        end = min(pos + grain_len, length)
        out[pos:end] += g[: end - pos]

        pos += effective_hop

    return out


# =========================
# Aphex Twin 風のシンセ／ドラム生成
# =========================
def apply_aphex_style(length, sr, tempo, scale, glitch, atmosphere, rng):
    result = np.zeros(length, dtype=np.float32)

    # ベースライン（スケール準拠）
    bass_notes = get_scale_notes(36, scale, 8)
    note_length = max(1, length // len(bass_notes))

    for i, note in enumerate(bass_notes):
        start = i * note_length
        end = min((i + 1) * note_length, length)
        n = end - start
        if n <= 0:
            continue

        t = np.arange(n, dtype=np.float32) / sr
        freq = 440.0 * (2.0 ** ((note - 69.0) / 12.0))
        envelope = np.exp(-2.0 * t / (note_length / sr + 1e-6))
        sine = np.sin(2.0 * np.pi * freq * t)
        sub = np.sin(2.0 * np.pi * freq * 0.5 * t)
        result[start:end] += (0.7 * sine + 0.3 * sub) * envelope * 0.18

    # うねるキック
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

    # アンビエントなドローン・パッド
    i_arr = np.arange(length, dtype=np.float32)
    lfo_slow = np.sin(2.0 * np.pi * 0.25 * i_arr / sr)
    lfo_fast = np.sin(2.0 * np.pi * 0.7 * i_arr / sr)

    drone_freq = 110.0
    drone = np.sin(2.0 * np.pi * drone_freq * i_arr / sr) * (0.7 + 0.3 * lfo_slow)
    pad = np.sin(2.0 * np.pi * drone_freq * 0.5 * i_arr / sr) * (0.5 + 0.5 * lfo_fast)

    amb = drone * 0.15 + pad * 0.12
    amb_scale = 0.3 + 0.7 * (float(atmosphere) / 10.0)
    result += amb * amb_scale

    # グリッチ（オーディオチャンクの反転・ゲイン変化）
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
                        seg *= (0.4 + 0.6 * rng.random())
                    result[i : i + chunk_size] = seg
                i += chunk_size

    return result


# =========================
# Squarepusher 風のブレイクビート／ベース
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

    # メロディ（複雑さが高いとき）
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
# 簡易アレンジ（フェードイン・アウト）
# =========================
def arrange_layers(core, granular, original, sr):
    length = len(core)
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)

    # 0〜5%でフェードイン、90〜100%でフェードアウト
    fade_in = np.clip(t / 0.05, 0.0, 1.0)
    fade_out = np.clip((1.0 - t) / 0.1, 0.0, 1.0)
    env_master = np.minimum(fade_in, fade_out)

    mixed = core + granular + original
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
):
    if audio.size == 0:
        raise ValueError("音声データが空です。")

    rng = np.random.default_rng(int(seed))

    # 入力を軽く正規化
    max_in = float(np.max(np.abs(audio)))
    if max_in > 0:
        audio = (audio / max_in).astype(np.float32)
    else:
        audio = audio.astype(np.float32)

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

    length = len(audio)

    # グラニュラー・レイヤー（原曲を分解して再構成）
    granular = make_granular_layer(
        audio,
        sr,
        rng,
        chop_amount=float(glitch),
        texture_amount=float(atmosphere) / 10.0,
    )

    # IDM コア（スタイル別）
    if style == "Aphex Twin":
        core = apply_aphex_style(
            length,
            sr,
            tempo=float(tempo),
            scale=scale,
            glitch=float(glitch),
            atmosphere=int(atmosphere),
            rng=rng,
        )
    else:
        core = apply_squarepusher_style(
            length,
            sr,
            tempo=float(tempo),
            scale=scale,
            bass=float(bass),
            complexity=int(complexity),
            rng=rng,
        )

    # 原曲レイヤーはかなり控えめに（ほぼテクスチャ）
    original_level = 0.08 + 0.12 * (1.0 - float(glitch))
    original_layer = audio * original_level

    # 全レイヤーを簡易アレンジ
    processed = arrange_layers(core, granular, original_layer, sr)

    # 最終正規化
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
        "アップロードした曲を素材に、Aphex Twin / Squarepusher テイストの IDM トラックに自動変換します。"
    )

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        uploaded_file = st.file_uploader(
            "元になる音源（WAV / MP3 など）",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
        )

        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/*")

        style = st.radio("スタイル", ["Aphex Twin", "Squarepusher"])

        scale = st.selectbox(
            "スケール（曲の雰囲気になる音階）",
            options=list(SCALES.keys()),
            index=0,
            help="minor: 暗め / major: 明るめ / pentatonic: 和風 / それ以外は少しマニアックなモード音階です。",
        )

        seed = st.number_input(
            "ランダムシード（同じ値なら同じ展開）",
            min_value=0,
            max_value=99999,
            value=42,
            step=1,
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
            help="特に Squarepusher スタイルの低音の主張に効きます。Aphex スタイルでも少し影響します。",
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
        if uploaded_file is None:
            st.warning("先に音声ファイルをアップロードしてください。")
        else:
            with st.spinner("生成中..."):
                try:
                    # ファイルを一度バイト列として取得してから一時ファイルに保存
                    file_bytes = uploaded_file.getvalue()
                    if not file_bytes:
                        st.error("アップロードされたファイルが空です。別のファイルを試してください。")
                        return

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
                        tmp.write(file_bytes)
                        tmp_path = tmp.name

                    try:
                        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
                    except Exception as e:
                        st.error(f"音声ファイルの読み込みに失敗しました: {e}")
                        return

                    try:
                        sr_out, processed = generate_idm_array(
                            audio=audio,
                            sr=sr,
                            style=style,
                            tempo=tempo,
                            glitch=glitch,
                            bass=bass,
                            duration_min=duration,
                            seed=seed,
                            scale=scale,
                            complexity=complexity,
                            atmosphere=atmosphere,
                        )
                    except Exception as e:
                        st.error(f"トラック生成中にエラーが発生しました: {e}")
                        return

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
                except Exception as e:
                    st.error(f"予期しないエラーが発生しました: {e}")


if __name__ == "__main__":
    main()

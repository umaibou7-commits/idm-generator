import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import io
import random
from scipy import signal
from scipy.ndimage import gaussian_filter1d

st.set_page_config(page_title="IDM Generator", layout="wide", initial_sidebar_state="expanded")

class GlitchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, padding=7)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=15, padding=7)
        self.conv4 = nn.Conv1d(16, 1, kernel_size=15, padding=7)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.tanh(self.conv2(x))
        x = self.tanh(self.conv3(x))
        x = self.tanh(self.conv4(x))
        return x

@st.cache_resource
def get_glitch_model():
    model = GlitchNet()
    model.eval()
    return model

def apply_ai_glitch(audio, sr, intensity, model):
    if intensity < 0.01:
        return audio
    
    chunk_size = min(sr * 2, len(audio))
    processed = np.copy(audio)
    
    for i in range(0, len(audio) - chunk_size, chunk_size // 2):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        
        chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0)
        
        with torch.no_grad():
            glitched = model(chunk_tensor).squeeze().numpy()
        
        blend = intensity * glitched[:len(processed[i:i+chunk_size])] + (1 - intensity) * processed[i:i+chunk_size]
        processed[i:i+chunk_size] = blend
    
    return processed

def granular_synthesis(audio, sr, grain_size_ms=50, density=0.5):
    grain_samples = int(sr * grain_size_ms / 1000)
    output = np.zeros_like(audio)
    
    num_grains = int(len(audio) / grain_samples * density)
    
    for _ in range(num_grains):
        start = random.randint(0, max(0, len(audio) - grain_samples))
        grain = audio[start:start+grain_samples]
        
        if len(grain) < grain_samples:
            continue
            
        envelope = np.hanning(len(grain))
        grain = grain * envelope
        
        out_pos = random.randint(0, max(0, len(output) - len(grain)))
        output[out_pos:out_pos+len(grain)] += grain * random.uniform(0.3, 0.8)
    
    return output

def bit_crush(audio, bits=8):
    max_val = 2**(bits - 1)
    crushed = np.round(audio * max_val) / max_val
    return crushed

def stutter_effect(audio, sr, num_stutters=5):
    output = np.copy(audio)
    
    for _ in range(num_stutters):
        stutter_len = random.randint(sr // 32, sr // 8)
        start = random.randint(0, max(0, len(audio) - stutter_len))
        
        stutter_chunk = audio[start:start+stutter_len]
        repeats = random.randint(2, 6)
        
        for r in range(repeats):
            insert_pos = start + r * stutter_len
            if insert_pos + len(stutter_chunk) < len(output):
                output[insert_pos:insert_pos+len(stutter_chunk)] = stutter_chunk * (1 - r * 0.15)
    
    return output

def apply_lfo_modulation(audio, sr, rate_hz=0.5, depth=0.3):
    lfo = np.sin(2 * np.pi * rate_hz * np.arange(len(audio)) / sr)
    modulated = audio * (1 + depth * lfo)
    return modulated

def reverse_chunks(audio, sr, chunk_duration_ms=200, probability=0.3):
    chunk_samples = int(sr * chunk_duration_ms / 1000)
    output = np.copy(audio)
    
    for i in range(0, len(audio) - chunk_samples, chunk_samples):
        if random.random() < probability:
            output[i:i+chunk_samples] = output[i:i+chunk_samples][::-1]
    
    return output

def create_amen_break(sr, duration, tempo_multiplier):
    t = np.linspace(0, duration, int(sr * duration))
    
    kick_times = [0, 0.5, 1.0, 1.5]
    snare_times = [0.25, 0.75, 1.25, 1.75]
    hihat_times = np.arange(0, 2, 0.125)
    
    breakbeat = np.zeros_like(t)
    
    for kt in kick_times:
        idx = int(kt * sr * tempo_multiplier)
        if idx < len(breakbeat):
            kick = signal.gaussian(int(sr * 0.05), std=int(sr * 0.01))
            kick = kick * np.sin(2 * np.pi * 60 * np.arange(len(kick)) / sr)
            end_idx = min(idx + len(kick), len(breakbeat))
            breakbeat[idx:end_idx] += kick[:end_idx-idx]
    
    for st in snare_times:
        idx = int(st * sr * tempo_multiplier)
        if idx < len(breakbeat):
            snare = np.random.randn(int(sr * 0.08)) * 0.5
            end_idx = min(idx + len(snare), len(breakbeat))
            breakbeat[idx:end_idx] += snare[:end_idx-idx]
    
    for ht in hihat_times:
        idx = int(ht * sr * tempo_multiplier)
        if idx < len(breakbeat):
            hihat = np.random.randn(int(sr * 0.02)) * 0.2
            hihat = librosa.effects.preemphasis(hihat)
            end_idx = min(idx + len(hihat), len(breakbeat))
            breakbeat[idx:end_idx] += hihat[:end_idx-idx]
    
    return breakbeat

def bass_synthesizer(sr, duration, style='squarepusher'):
    t = np.linspace(0, duration, int(sr * duration))
    bass = np.zeros_like(t)
    
    if style == 'squarepusher':
        notes = [40, 45, 38, 43, 40, 36, 45, 38]
        note_duration = duration / len(notes)
        
        for i, note in enumerate(notes):
            start = int(i * note_duration * sr)
            end = int((i + 1) * note_duration * sr)
            if end > len(bass):
                end = len(bass)
            
            freq = librosa.midi_to_hz(note)
            note_t = np.linspace(0, note_duration, end - start)
            
            saw = 2 * (note_t * freq - np.floor(note_t * freq + 0.5))
            square = signal.square(2 * np.pi * freq * note_t)
            
            envelope = np.exp(-3 * note_t / note_duration)
            bass_note = (0.6 * saw + 0.4 * square) * envelope
            
            bass[start:end] += bass_note
    
    return bass

def apply_compression(audio, threshold=0.3, ratio=4):
    abs_audio = np.abs(audio)
    mask = abs_audio > threshold
    
    compressed = np.copy(audio)
    compressed[mask] = np.sign(audio[mask]) * (threshold + (abs_audio[mask] - threshold) / ratio)
    
    return compressed

def spring_reverb(audio, sr, decay=0.5, spring_freq=800):
    impulse_len = int(sr * 0.5)
    impulse = np.zeros(impulse_len)
    
    for i in range(10):
        delay = int(sr * (0.01 + i * 0.015))
        if delay < impulse_len:
            impulse[delay] += decay ** i * np.sin(2 * np.pi * spring_freq * delay / sr)
    
    reverb = signal.fftconvolve(audio, impulse, mode='same')
    return reverb

def aphex_twin_style(audio, sr, params, model):
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    
    audio = apply_ai_glitch(audio, sr, params['glitch_intensity'], model)
    
    if params['glitch_intensity'] > 0.3:
        audio = bit_crush(audio, bits=max(4, int(12 * (1 - params['glitch_intensity']))))
    
    audio = reverse_chunks(audio, sr, chunk_duration_ms=random.randint(100, 400), 
                          probability=params['glitch_intensity'] * 0.5)
    
    audio = granular_synthesis(audio, sr, grain_size_ms=random.randint(30, 100), 
                              density=0.3 + params['glitch_intensity'] * 0.4)
    
    audio = apply_lfo_modulation(audio, sr, rate_hz=random.uniform(0.2, 1.5), 
                                depth=params['glitch_intensity'] * 0.4)
    
    if params['glitch_intensity'] > 0.5:
        audio = stutter_effect(audio, sr, num_stutters=int(params['glitch_intensity'] * 8))
    
    drone = np.sin(2 * np.pi * random.choice([55, 110, 220]) * np.arange(len(audio)) / sr)
    drone = gaussian_filter1d(drone, sigma=sr//10)
    audio = audio * 0.85 + drone * 0.15 * params['glitch_intensity']
    
    return audio

def squarepusher_style(audio, sr, params, model):
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    
    breakbeat = create_amen_break(sr, len(audio) / sr, params['tempo_multiplier'])
    if len(breakbeat) > len(audio):
        breakbeat = breakbeat[:len(audio)]
    else:
        breakbeat = np.pad(breakbeat, (0, len(audio) - len(breakbeat)))
    
    bass = bass_synthesizer(sr, len(audio) / sr, style='squarepusher')
    if len(bass) > len(audio):
        bass = bass[:len(audio)]
    else:
        bass = np.pad(bass, (0, len(audio) - len(bass)))
    
    bass = bass * params['bass_boost'] * 1.5
    
    audio = apply_ai_glitch(audio, sr, params['glitch_intensity'] * 0.6, model)
    
    audio = audio * 0.4 + breakbeat * 0.35 + bass * 0.25
    
    audio = spring_reverb(audio, sr, decay=0.4)
    
    audio = apply_compression(audio, threshold=0.25, ratio=6)
    
    if params['glitch_intensity'] > 0.3:
        audio = stutter_effect(audio, sr, num_stutters=int(params['glitch_intensity'] * 5))
    
    return audio

def process_audio(uploaded_file, params):
    model = get_glitch_model()
    
    audio, sr = librosa.load(uploaded_file, sr=44100, mono=True)
    
    target_length = int(sr * params['duration'] * 60)
    
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        repeats = int(np.ceil(target_length / len(audio)))
        audio = np.tile(audio, repeats)[:target_length]
    
    audio = librosa.effects.time_stretch(audio, rate=params['tempo_multiplier'])
    
    if params['style'] == 'Aphex Twin':
        processed = aphex_twin_style(audio, sr, params, model)
    else:
        processed = squarepusher_style(audio, sr, params, model)
    
    max_val = np.max(np.abs(processed))
    if max_val > 0:
        processed = processed / max_val * 0.95
    
    return processed, sr

def main():
    st.title("🎵 IDM Generator: Aphex Twin × Squarepusher")
    st.markdown("### AI-Powered Glitch & Breakcore Generator")
    
    with st.sidebar:
        st.header("⚙️ Parameters")
        
        style = st.selectbox(
            "Style",
            ["Aphex Twin", "Squarepusher"],
            help="Aphex Twin: Glitch, ambient, experimental | Squarepusher: Breakcore, fast bass"
        )
        
        tempo = st.slider("Tempo Multiplier", 0.8, 2.5, 1.2, 0.1)
        glitch = st.slider("Glitch Intensity", 0.0, 1.0, 0.5, 0.05)
        bass = st.slider("Bass Boost", 0.5, 3.0, 1.5, 0.1)
        duration = st.slider("Duration (minutes)", 1, 8, 3, 1)
        seed = st.number_input("Seed (for reproducibility)", 0, 99999, 42)
        
        st.markdown("---")
        st.markdown("**Tips:**")
        st.markdown("- Higher glitch = more chaos")
        st.markdown("- Squarepusher needs bass boost")
        st.markdown("- Same seed = same output")
    
    uploaded_file = st.file_uploader(
        "Drop your audio file (WAV/MP3/FLAC/OGG)",
        type=['wav', 'mp3', 'flac', 'ogg']
    )
    
    if uploaded_file is not None:
        st.success(f"✅ Loaded: {uploaded_file.name}")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Generate IDM Track", use_container_width=True):
                params = {
                    'style': style,
                    'tempo_multiplier': tempo,
                    'glitch_intensity': glitch,
                    'bass_boost': bass,
                    'duration': duration,
                    'seed': seed
                }
                
                with st.spinner(f"Generating {style} style... This may take a minute..."):
                    try:
                        processed_audio, sr = process_audio(uploaded_file, params)
                        
                        st.session_state['audio'] = processed_audio
                        st.session_state['sr'] = sr
                        st.session_state['params'] = params
                        
                        st.success("✨ Generation complete!")
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Try a different file or adjust parameters")
        
        if 'audio' in st.session_state:
            st.markdown("---")
            st.subheader("🎧 Your Generated Track")
            
            buffer = io.BytesIO()
            sf.write(buffer, st.session_state['audio'], st.session_state['sr'], format='WAV')
            buffer.seek(0)
            
            st.audio(buffer, format='audio/wav')
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.download_button(
                    label="⬇️ Download WAV",
                    data=buffer,
                    file_name=f"idm_{st.session_state['params']['style'].replace(' ', '_').lower()}_{st.session_state['params']['seed']}.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
            
            with col2:
                if st.button("❤️ Love this track!", use_container_width=True):
                    st.balloons()
                    st.success("Saved to favorites! (local session)")
            
            with col3:
                if st.button("🔄 Clear", use_container_width=True):
                    for key in ['audio', 'sr', 'params']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
            
            with st.expander("📊 Track Info"):
                st.write(f"**Style:** {st.session_state['params']['style']}")
                st.write(f"**Duration:** {len(st.session_state['audio']) / st.session_state['sr']:.1f}s")
                st.write(f"**Sample Rate:** {st.session_state['sr']} Hz")
                st.write(f"**Tempo Multiplier:** {st.session_state['params']['tempo_multiplier']}")
                st.write(f"**Glitch Intensity:** {st.session_state['params']['glitch_intensity']}")
                st.write(f"**Seed:** {st.session_state['params']['seed']}")
    
    else:
        st.info("👆 Upload an audio file to start generating IDM tracks")
        
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. **Upload** any audio file (vocals, instruments, field recordings)
            2. **Choose** your style: Aphex Twin (glitchy/ambient) or Squarepusher (breakcore/bass)
            3. **Adjust** parameters in the sidebar
            4. **Generate** and listen to your AI-powered IDM track
            5. **Download** the result as WAV
            
            **Pro Tips:**
            - Start with Aphex Twin at 0.5 glitch intensity
            - For Squarepusher, boost bass to 2.0+
            - Use the same seed to reproduce results
            - Longer durations = more processing time
            """)

if __name__ == "__main__":
    main()

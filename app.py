def aphex_twin_style(audio, sr, params, model):
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    
    original = np.copy(audio)
    
    melodic_bass = bass_synthesizer(sr, len(audio) / sr, style='aphex')
    if len(melodic_bass) > len(audio):
        melodic_bass = melodic_bass[:len(audio)]
    else:
        melodic_bass = np.pad(melodic_bass, (0, len(audio) - len(melodic_bass)))
    
    kick_pattern = create_kick_pattern(sr, len(audio) / sr, bpm=120 * params['tempo_multiplier'])
    if len(kick_pattern) > len(audio):
        kick_pattern = kick_pattern[:len(audio)]
    else:
        kick_pattern = np.pad(kick_pattern, (0, len(audio) - len(kick_pattern)))
    
    audio = audio * 0.6 + melodic_bass * 0.25 + kick_pattern * 0.15
    
    glitch_intensity = params['glitch_intensity'] * 0.4
    audio = apply_ai_glitch(audio, sr, glitch_intensity, model)
    
    if params['glitch_intensity'] > 0.5:
        audio = reverse_chunks(audio, sr, chunk_duration_ms=random.randint(200, 600), 
                              probability=params['glitch_intensity'] * 0.2)
    
    if params['glitch_intensity'] > 0.6:
        audio = stutter_effect(audio, sr, num_stutters=int(params['glitch_intensity'] * 3))
    
    drone = np.sin(2 * np.pi * random.choice([55, 82.5, 110]) * np.arange(len(audio)) / sr)
    drone = gaussian_filter1d(drone, sigma=sr//5)
    audio = audio * 0.9 + drone * 0.1 * (params['glitch_intensity'] * 0.5)
    
    audio = add_ambient_pad(audio, sr)
    
    return audio

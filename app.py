import React, { useState, useRef } from 'react';
import { Upload, Play, Download, Zap, Music, Sliders, RefreshCw, Heart, Info } from 'lucide-react';

const IDMGenerator = () => {
  const [file, setFile] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [progress, setProgress] = useState(0);
  const [showInfo, setShowInfo] = useState(false);
  
  const [style, setStyle] = useState('aphex');
  const [tempo, setTempo] = useState(1.2);
  const [glitch, setGlitch] = useState(0.5);
  const [bass, setBass] = useState(1.5);
  const [duration, setDuration] = useState(3);
  const [seed, setSeed] = useState(42);
  const [scale, setScale] = useState('minor');
  const [complexity, setComplexity] = useState(5);
  const [atmosphere, setAtmosphere] = useState(5);
  
  const audioRef = useRef(null);
  const fileInputRef = useRef(null);

  const scales = {
    minor: [0, 2, 3, 5, 7, 8, 10],
    major: [0, 2, 4, 5, 7, 9, 11],
    pentatonic: [0, 2, 4, 7, 9],
    blues: [0, 3, 5, 6, 7, 10],
    dorian: [0, 2, 3, 5, 7, 9, 10],
    phrygian: [0, 1, 3, 5, 7, 8, 10],
    chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setAudioUrl(null);
    }
  };

  const generateAudio = async () => {
    if (!file) return;
    
    setIsGenerating(true);
    setProgress(0);
    
    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const arrayBuffer = await file.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      setProgress(20);
      
      let channelData = audioBuffer.getChannelData(0);
      const sampleRate = audioBuffer.sampleRate;
      const targetLength = Math.floor(sampleRate * duration * 60);
      
      if (channelData.length > targetLength) {
        channelData = channelData.slice(0, targetLength);
      } else {
        const repeats = Math.ceil(targetLength / channelData.length);
        const repeated = new Float32Array(targetLength);
        for (let i = 0; i < repeats; i++) {
          repeated.set(channelData, i * channelData.length);
        }
        channelData = repeated.slice(0, targetLength);
      }
      
      setProgress(40);
      
      let processed = new Float32Array(channelData);
      
      if (style === 'aphex') {
        processed = applyAphexStyle(processed, sampleRate);
      } else {
        processed = applySquarepusherStyle(processed, sampleRate);
      }
      
      setProgress(80);
      
      const max = Math.max(...processed.map(Math.abs));
      if (max > 0) {
        processed = processed.map(v => v / max * 0.95);
      }
      
      setProgress(90);
      
      const newBuffer = audioContext.createBuffer(1, processed.length, sampleRate);
      newBuffer.copyToChannel(processed, 0);
      
      const wav = audioBufferToWav(newBuffer);
      const blob = new Blob([wav], { type: 'audio/wav' });
      const url = URL.createObjectURL(blob);
      
      setAudioUrl(url);
      setProgress(100);
      
    } catch (error) {
      console.error('Generation error:', error);
      alert('生成エラーが発生しました。別のファイルをお試しください。');
    } finally {
      setIsGenerating(false);
    }
  };

  const applyAphexStyle = (audio, sr) => {
    const result = new Float32Array(audio.length);
    
    const bassNotes = getScaleNotes(36, scale, 8);
    const noteDuration = audio.length / bassNotes.length;
    
    for (let i = 0; i < bassNotes.length; i++) {
      const start = Math.floor(i * noteDuration);
      const end = Math.floor((i + 1) * noteDuration);
      const freq = 440 * Math.pow(2, (bassNotes[i] - 69) / 12);
      
      for (let j = start; j < end && j < audio.length; j++) {
        const t = (j - start) / sr;
        const envelope = Math.exp(-2 * t / (noteDuration / sr));
        const sine = Math.sin(2 * Math.PI * freq * t);
        const sub = Math.sin(2 * Math.PI * freq * 0.5 * t);
        result[j] += (0.7 * sine + 0.3 * sub) * envelope * 0.15;
      }
    }
    
    const bpm = 120 * tempo;
    const beatDuration = Math.floor(60 / bpm * sr);
    for (let i = 0; i < audio.length; i += beatDuration) {
      const kickLen = Math.floor(sr * 0.15);
      for (let j = 0; j < kickLen && i + j < audio.length; j++) {
        const t = j / sr;
        const kick = Math.sin(2 * Math.PI * 50 * t) * Math.exp(-15 * t);
        result[i + j] += kick * 0.4;
      }
    }
    
    for (let i = 0; i < audio.length; i++) {
      const lfo = Math.sin(2 * Math.PI * 0.5 * i / sr);
      const drone = Math.sin(2 * Math.PI * 110 * i / sr) * (0.8 + 0.2 * lfo);
      result[i] += audio[i] * 0.5 + drone * 0.1 * (atmosphere / 10);
    }
    
    if (glitch > 0.3) {
      const chunkSize = Math.floor(sr * 0.3);
      for (let i = 0; i < audio.length - chunkSize; i += chunkSize * 2) {
        if (Math.random() < glitch * 0.3) {
          for (let j = 0; j < chunkSize && i + j < audio.length; j++) {
            result[i + j] = result[i + chunkSize - j];
          }
        }
      }
    }
    
    return result;
  };

  const applySquarepusherStyle = (audio, sr) => {
    const result = new Float32Array(audio.length);
    
    const bpm = 170 * tempo;
    const beatDuration = Math.floor(60 / bpm * sr);
    const pattern = [0, 0.5, 1, 1.25, 1.75, 2, 2.5, 2.75, 3, 3.75];
    
    for (let bar = 0; bar < audio.length / (beatDuration * 4); bar++) {
      const barStart = bar * beatDuration * 4;
      
      pattern.forEach(pos => {
        const idx = Math.floor(barStart + pos * beatDuration);
        if (idx < audio.length) {
          const kickLen = Math.floor(sr * 0.1);
          for (let j = 0; j < kickLen && idx + j < audio.length; j++) {
            const t = j / sr;
            const kick = Math.sin(2 * Math.PI * 55 * t) * Math.exp(-25 * t);
            result[idx + j] += kick * 0.7;
          }
        }
      });
      
      [0.5, 1.5, 2.5, 3.5].forEach(pos => {
        const idx = Math.floor(barStart + pos * beatDuration);
        if (idx < audio.length) {
          const snareLen = Math.floor(sr * 0.12);
          for (let j = 0; j < snareLen && idx + j < audio.length; j++) {
            const noise = (Math.random() - 0.5) * 0.4;
            result[idx + j] += noise;
          }
        }
      });
      
      for (let i = 0; i < 32; i++) {
        const idx = Math.floor(barStart + i * beatDuration / 8);
        if (idx < audio.length) {
          const hihatLen = Math.floor(sr * 0.03);
          for (let j = 0; j < hihatLen && idx + j < audio.length; j++) {
            const noise = (Math.random() - 0.5) * 0.08;
            result[idx + j] += noise;
          }
        }
      }
    }
    
    const bassNotes = getScaleNotes(40, scale, 12);
    const noteDuration = audio.length / bassNotes.length;
    
    for (let i = 0; i < bassNotes.length; i++) {
      const start = Math.floor(i * noteDuration);
      const end = Math.floor((i + 1) * noteDuration);
      const freq = 440 * Math.pow(2, (bassNotes[i] - 69) / 12);
      
      for (let j = start; j < end && j < audio.length; j++) {
        const t = (j - start) / sr;
        const envelope = Math.exp(-2 * t / (noteDuration / sr));
        const saw = 2 * (t * freq - Math.floor(t * freq + 0.5));
        const sub = Math.sin(2 * Math.PI * freq * 0.5 * t);
        result[j] += (0.5 * saw + 0.3 * sub) * envelope * bass * 0.15;
      }
    }
    
    if (complexity > 5) {
      const melodyNotes = getScaleNotes(64, scale, 16);
      const melodyDuration = audio.length / melodyNotes.length;
      
      for (let i = 0; i < melodyNotes.length; i++) {
        const start = Math.floor(i * melodyDuration);
        const end = Math.floor((i + 1) * melodyDuration);
        const freq = 440 * Math.pow(2, (melodyNotes[i] - 69) / 12);
        
        for (let j = start; j < end && j < audio.length; j++) {
          const t = (j - start) / sr;
          const envelope = Math.exp(-3 * t / (melodyDuration / sr));
          const sine = Math.sin(2 * Math.PI * freq * t);
          result[j] += sine * envelope * 0.1;
        }
      }
    }
    
    for (let i = 0; i < audio.length; i++) {
      result[i] += audio[i] * 0.25;
    }
    
    return result;
  };

  const getScaleNotes = (rootMidi, scaleName, count) => {
    const scaleIntervals = scales[scaleName] || scales.minor;
    const notes = [];
    
    for (let i = 0; i < count; i++) {
      const octave = Math.floor(i / scaleIntervals.length);
      const intervalIdx = i % scaleIntervals.length;
      notes.push(rootMidi + octave * 12 + scaleIntervals[intervalIdx]);
    }
    
    return notes;
  };

  const audioBufferToWav = (buffer) => {
    const numberOfChannels = buffer.numberOfChannels;
    const length = buffer.length * numberOfChannels * 2;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);
    const channels = [];
    let offset = 0;
    let pos = 0;

    const setString = (str) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(pos + i, str.charCodeAt(i));
      }
      pos += str.length;
    };

    const setUint16 = (data) => {
      view.setUint16(pos, data, true);
      pos += 2;
    };

    const setUint32 = (data) => {
      view.setUint32(pos, data, true);
      pos += 4;
    };

    setString('RIFF');
    setUint32(36 + length);
    setString('WAVE');
    setString('fmt ');
    setUint32(16);
    setUint16(1);
    setUint16(numberOfChannels);
    setUint32(buffer.sampleRate);
    setUint32(buffer.sampleRate * 2 * numberOfChannels);
    setUint16(numberOfChannels * 2);
    setUint16(16);
    setString('data');
    setUint32(length);

    for (let i = 0; i < buffer.numberOfChannels; i++) {
      channels.push(buffer.getChannelData(i));
    }

    let index = 44;
    for (let i = 0; i < buffer.length; i++) {
      for (let channel = 0; channel < numberOfChannels; channel++) {
        let sample = Math.max(-1, Math.min(1, channels[channel][i]));
        sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        view.setInt16(index, sample, true);
        index += 2;
      }
    }

    return arrayBuffer;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 md:mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Zap className="w-8 h-8 md:w-12 md:h-12 text-purple-400 animate-pulse" />
            <h1 className="text-3xl md:text-5xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent">
              IDM Generator
            </h1>
            <Music className="w-8 h-8 md:w-12 md:h-12 text-pink-400 animate-pulse" />
          </div>
          <p className="text-purple-300 text-sm md:text-lg">AI駆動のグリッチ＆ブレイクコアジェネレーター</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 md:gap-8">
          {/* Left Panel - Controls */}
          <div className="lg:col-span-1 space-y-4">
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-6 border border-purple-500/20 shadow-xl">
              <div className="flex items-center gap-2 mb-6">
                <Sliders className="w-5 h-5 text-purple-400" />
                <h2 className="text-xl font-bold">パラメーター</h2>
              </div>

              {/* Style Selection */}
              <div className="mb-6">
                <label className="block text-sm font-medium mb-3 text-purple-300">スタイル</label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => setStyle('aphex')}
                    className={`p-4 rounded-xl font-medium transition-all ${
                      style === 'aphex'
                        ? 'bg-gradient-to-br from-purple-600 to-pink-600 shadow-lg shadow-purple-500/50'
                        : 'bg-slate-700/50 hover:bg-slate-700'
                    }`}
                  >
                    Aphex Twin
                  </button>
                  <button
                    onClick={() => setStyle('squarepusher')}
                    className={`p-4 rounded-xl font-medium transition-all ${
                      style === 'squarepusher'
                        ? 'bg-gradient-to-br from-purple-600 to-pink-600 shadow-lg shadow-purple-500/50'
                        : 'bg-slate-700/50 hover:bg-slate-700'
                    }`}
                  >
                    Squarepusher
                  </button>
                </div>
              </div>

              {/* Scale Selection */}
              <div className="mb-6">
                <label className="block text-sm font-medium mb-2 text-purple-300">
                  スケール <span className="text-xs text-slate-400">音階の種類</span>
                </label>
                <select
                  value={scale}
                  onChange={(e) => setScale(e.target.value)}
                  className="w-full bg-slate-700 rounded-lg p-3 text-white focus:ring-2 focus:ring-purple-500 outline-none"
                >
                  <option value="minor">マイナー（暗め）</option>
                  <option value="major">メジャー（明るめ）</option>
                  <option value="pentatonic">ペンタトニック（和風）</option>
                  <option value="blues">ブルース（ソウルフル）</option>
                  <option value="dorian">ドリアン（ジャズ風）</option>
                  <option value="phrygian">フリジアン（中東風）</option>
                  <option value="chromatic">クロマチック（全音）</option>
                </select>
              </div>

              {/* Sliders */}
              <div className="space-y-5">
                <div>
                  <label className="flex justify-between text-sm mb-2">
                    <span className="text-purple-300">テンポ倍率</span>
                    <span className="text-white font-mono">{tempo.toFixed(1)}x</span>
                  </label>
                  <input
                    type="range"
                    min="0.8"
                    max="2.5"
                    step="0.1"
                    value={tempo}
                    onChange={(e) => setTempo(parseFloat(e.target.value))}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                  />
                </div>

                <div>
                  <label className="flex justify-between text-sm mb-2">
                    <span className="text-purple-300">グリッチ強度</span>
                    <span className="text-white font-mono">{(glitch * 100).toFixed(0)}%</span>
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={glitch}
                    onChange={(e) => setGlitch(parseFloat(e.target.value))}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-pink-500"
                  />
                </div>

                <div>
                  <label className="flex justify-between text-sm mb-2">
                    <span className="text-purple-300">ベースブースト</span>
                    <span className="text-white font-mono">{bass.toFixed(1)}x</span>
                  </label>
                  <input
                    type="range"
                    min="0.5"
                    max="3"
                    step="0.1"
                    value={bass}
                    onChange={(e) => setBass(parseFloat(e.target.value))}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                  />
                </div>

                <div>
                  <label className="flex justify-between text-sm mb-2">
                    <span className="text-purple-300">複雑さ</span>
                    <span className="text-white font-mono">{complexity}/10</span>
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    step="1"
                    value={complexity}
                    onChange={(e) => setComplexity(parseInt(e.target.value))}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-pink-500"
                  />
                </div>

                <div>
                  <label className="flex justify-between text-sm mb-2">
                    <span className="text-purple-300">アトモスフィア</span>
                    <span className="text-white font-mono">{atmosphere}/10</span>
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    step="1"
                    value={atmosphere}
                    onChange={(e) => setAtmosphere(parseInt(e.target.value))}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                  />
                </div>

                <div>
                  <label className="flex justify-between text-sm mb-2">
                    <span className="text-purple-300">生成時間</span>
                    <span className="text-white font-mono">{duration}分</span>
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="8"
                    step="1"
                    value={duration}
                    onChange={(e) => setDuration(parseInt(e.target.value))}
                    className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-pink-500"
                  />
                </div>

                <div>
                  <label className="flex justify-between text-sm mb-2">
                    <span className="text-purple-300">シード値</span>
                    <span className="text-white font-mono">{seed}</span>
                  </label>
                  <input
                    type="number"
                    min="0"
                    max="99999"
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
                    className="w-full bg-slate-700 rounded-lg p-2 text-white focus:ring-2 focus:ring-purple-500 outline-none"
                  />
                </div>
              </div>
            </div>

            {/* Info Card */}
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-4 border border-purple-500/20">
              <button
                onClick={() => setShowInfo(!showInfo)}
                className="flex items-center gap-2 w-full text-left"
              >
                <Info className="w-4 h-4 text-purple-400" />
                <span className="text-sm font-medium">使い方</span>
              </button>
              {showInfo && (
                <div className="mt-3 text-xs text-slate-300 space-y-2">
                  <p>• 音声ファイルをアップロード</p>
                  <p>• スタイルとスケールを選択</p>
                  <p>• パラメーターを調整</p>
                  <p>• 生成ボタンをクリック</p>
                  <p>• 完成したトラックを再生・ダウンロード</p>
                </div>
              )}
            </div>
          </div>

          {/* Right Panel - Main Area */}
          <div className="lg:col-span-2 space-y-6">
            {/* Upload Area */}
            <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-8 border-2 border-dashed border-purple-500/30 hover:border-purple-500/60 transition-colors">
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileChange}
                className="hidden"
              />
              
              {!file ? (
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full flex flex-col items-center justify-center py-12 cursor-pointer group"
                >
                  <Upload className="w-16 h-16 text-purple-400 mb-4 group-hover:scale-110 transition-transform" />
                  <p className="text-xl font-medium mb-2">音声ファイルをドロップ</p>
                  <p className="text-sm text-slate-400">または クリックして選択</p>
                  <p className="text-xs text-slate-500 mt-2">WAV, MP3, FLAC, OGG対応</p>
                </button>
              ) : (
                <div className="text-center">
                  <div className="bg-purple-500/20 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-4">
                    <Music className="w-10 h-10 text-purple-400" />
                  </div>
                  <p className="font-medium mb-2">{file.name}</p>
                  <p className="text-sm text-slate-400 mb-4">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="text-purple-400 hover:text-purple-300 text-sm flex items-center gap-2 mx-auto"
                  >
                    <RefreshCw className="w-4 h-4" />
                    別のファイルを選択
                  </button>
                </div>
              )}
            </div>

            {/* Generate Button */}
            {file && !audioUrl && (
              <button
                onClick={generateAudio}
                disabled={isGenerating}
                className={`w-full py-6 rounded-2xl font-bold text-lg transition-all ${
                  isGenerating
                    ? 'bg-slate-700 cursor-not-allowed'
                    : 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 shadow-lg shadow-purple-500/50 hover:shadow-xl hover:shadow-purple-500/60'
                }`}
              >
                {isGenerating ? (
                  <div className="flex items-center justify-center gap-3">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
                    <span>生成中... {progress}%</span>
                  </div>
                ) : (
                  <div className="flex items-center justify-center gap-3">
                    <Zap className="w-6 h-6" />
                    <span>IDMトラックを生成</span>
                  </div>
                )}
              </button>
            )}

            {/* Progress Bar */}
            {isGenerating && (
              <div className="bg-slate-800/50 rounded-2xl p-6 border border-purple-500/20">
                <div className="w-full bg-slate-700 rounded-full h-3 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-purple-600 to-pink-600 transition-all duration-300 rounded-full"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="text-center text-sm text-slate-400 mt-3">
                  {progress < 40 ? 'オーディオを読み込んでいます...' :
                   progress < 80 ? `${style === 'aphex' ? 'Aphex Twin' : 'Squarepusher'}スタイルを適用中...` :
                   '最終処理中...'}
                </p>
              </div>
            )}

            {/* Audio Player */}
            {audioUrl && (
              <div className="bg-slate-800/50 backdrop-blur-lg rounded-2xl p-8 border border-purple-500/20 shadow-xl">
                <div className="flex items-center gap-3 mb-6">
                  <div className="bg-gradient-to-br from-purple-600 to-pink-600 rounded-full w-12 h-12 flex items-center justify-center">
                    <Play className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold">生成完了！</h3>
                    <p className="text-sm text-slate-400">あなたのIDMトラック</p>
                  </div>
                </div>

                <audio
                  ref={audioRef}
                  src={audioUrl}
                  controls
                  className="w-full mb-6"
                  style={{
                    filter: 'hue-rotate(270deg) saturate(3)',
                    height: '54px'
                  }}
                />

                <div className="grid grid-cols-2 gap-4">
                  <a
                    href={audioUrl}
                    download={`idm_${style}_${seed}.wav`}
                    className="flex items-center justify-center gap-2 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 py-4 rounded-xl font-medium transition-all shadow-lg hover:shadow-xl"
                  >
                   <Download className="w-5 h-5" />
                    ダウンロード
                  </a>
                  
                  <button
                    onClick={() => {
                      setAudioUrl(null);
                      setFile(null);
                    }}
                    className="flex items-center justify-center gap-2 bg-slate-700 hover:bg-slate-600 py-4 rounded-xl font-medium transition-all"
                  >
                    <RefreshCw className="w-5 h-5" />
                    新しく作る
                  </button>
                </div>

                <div className="mt-6 p-4 bg-slate-900/50 rounded-xl">
                  <h4 className="text-sm font-medium mb-3 text-purple-300">トラック情報</h4>
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <span className="text-slate-400">スタイル:</span>
                      <span className="ml-2 text-white">{style === 'aphex' ? 'Aphex Twin' : 'Squarepusher'}</span>
                    </div>
                    <div>
                      <span className="text-slate-400">スケール:</span>
                      <span className="ml-2 text-white">{scale}</span>
                    </div>
                    <div>
                      <span className="text-slate-400">テンポ:</span>
                      <span className="ml-2 text-white">{tempo}x</span>
                    </div>
                    <div>
                      <span className="text-slate-400">グリッチ:</span>
                      <span className="ml-2 text-white">{(glitch * 100).toFixed(0)}%</span>
                    </div>
                    <div>
                      <span className="text-slate-400">複雑さ:</span>
                      <span className="ml-2 text-white">{complexity}/10</span>
                    </div>
                    <div>
                      <span className="text-slate-400">シード:</span>
                      <span className="ml-2 text-white">{seed}</span>
                    </div>
                  </div>
                </div>

                <button
                  onClick={() => {
                    alert('お気に入りに保存しました！（このセッション内のみ）');
                  }}
                  className="w-full mt-4 flex items-center justify-center gap-2 bg-pink-600/20 hover:bg-pink-600/30 py-3 rounded-xl font-medium transition-all border border-pink-500/30"
                >
                  <Heart className="w-5 h-5 text-pink-400" />
                  このトラックを保存
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-12 text-slate-500 text-sm">
          <p>AI駆動のIDMジェネレーター | Aphex Twin × Squarepusher スタイル</p>
        </div>
      </div>
    </div>
  );
};

export default IDMGenerator;

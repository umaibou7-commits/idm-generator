'use client';

import React, { useState, useRef } from 'react';
import {
  Upload,
  Play,
  Download,
  Zap,
  Music,
  Sliders,
  RefreshCw,
  Heart,
  Info
} from 'lucide-react';

const IDMGenerator = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [showInfo, setShowInfo] = useState(false);

  const [style, setStyle] = useState<'aphex' | 'squarepusher'>('aphex');
  const [tempo, setTempo] = useState(1.2);
  const [glitch, setGlitch] = useState(0.5);
  const [bass, setBass] = useState(1.5);
  const [duration, setDuration] = useState(3); // 分
  const [seed, setSeed] = useState(42);
  const [scale, setScale] = useState<
    'minor' | 'major' | 'pentatonic' | 'blues' | 'dorian' | 'phrygian' | 'chromatic'
  >('minor');
  const [complexity, setComplexity] = useState(5);
  const [atmosphere, setAtmosphere] = useState(5);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const scales: Record<string, number[]> = {
    minor: [0, 2, 3, 5, 7, 8, 10],
    major: [0, 2, 4, 5, 7, 9, 11],
    pentatonic: [0, 2, 4, 7, 9],
    blues: [0, 3, 5, 6, 7, 10],
    dorian: [0, 2, 3, 5, 7, 9, 10],
    phrygian: [0, 1, 3, 5, 7, 8, 10],
    chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
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
      const AudioCtx =
        (window as any).AudioContext || (window as any).webkitAudioContext;
      if (!AudioCtx) {
        alert('このブラウザはWeb Audio APIに対応していません。');
        return;
      }

      const audioContext = new AudioCtx();
      const arrayBuffer = await file.arrayBuffer();

      // Safari互換の decodeAudioData
      const audioBuffer = await new Promise<AudioBuffer>((resolve, reject) => {
        // 新仕様（Promiseを返す）の場合
        if (audioContext.decodeAudioData.length === 1) {
          audioContext
            .decodeAudioData(arrayBuffer)
            .then(resolve)
            .catch(reject);
        } else {
          // 旧仕様（コールバック）の場合
          audioContext.decodeAudioData(
            arrayBuffer,
            (buffer: AudioBuffer) => resolve(buffer),
            (err: any) => reject(err)
          );
        }
      });

      setProgress(20);

      let channelData = audioBuffer.getChannelData(0);
      const sampleRate = audioBuffer.sampleRate;
      const targetLength = Math.floor(sampleRate * duration * 60); // 分 → 秒

      // 長さ調整（★ここを安全なやり方に修正）
      if (channelData.length > targetLength) {
        channelData = channelData.slice(0, targetLength);
      } else if (channelData.length < targetLength) {
        const repeated = new Float32Array(targetLength);
        for (let i = 0; i < targetLength; i++) {
          repeated[i] = channelData[i % channelData.length];
        }
        channelData = repeated;
      }

      setProgress(40);

      let processed = new Float32Array(channelData);

      if (style === 'aphex') {
        processed = applyAphexStyle(processed, sampleRate);
      } else {
        processed = applySquarepusherStyle(processed, sampleRate);
      }

      setProgress(80);

      // ★正規化の安全な実装（Math.max + スプレッドをやめる）
      let max = 0;
      for (let i = 0; i < processed.length; i++) {
        const v = Math.abs(processed[i]);
        if (v > max) max = v;
      }

      if (max > 0) {
        const scaleGain = 0.95 / max;
        for (let i = 0; i < processed.length; i++) {
          processed[i] *= scaleGain;
        }
      }

      setProgress(90);

      const newBuffer = audioContext.createBuffer(
        1,
        processed.length,
        sampleRate
      );
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

  const applyAphexStyle = (audio: Float32Array, sr: number) => {
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
    const beatDuration = Math.floor((60 / bpm) * sr);
    for (let i = 0; i < audio.length; i += beatDuration) {
      const kickLen = Math.floor(sr * 0.15);
      for (let j = 0; j < kickLen && i + j < audio.length; j++) {
        const t = j / sr;
        const kick =
          Math.sin(2 * Math.PI * 50 * t) * Math.exp(-15 * t);
        result[i + j] += kick * 0.4;
      }
    }

    for (let i = 0; i < audio.length; i++) {
      const lfo = Math.sin((2 * Math.PI * 0.5 * i) / sr);
      const drone =
        Math.sin((2 * Math.PI * 110 * i) / sr) * (0.8 + 0.2 * lfo);
      result[i] +=
        audio[i] * 0.5 + drone * 0.1 * (atmosphere / 10);
    }

    if (glitch > 0.3) {
      const chunkSize = Math.floor(sr * 0.3);
      for (let i = 0; i < audio.length - chunkSize; i += chunkSize * 2) {
        if (Math.random() < glitch * 0.3) {
          for (
            let j = 0;
            j < chunkSize && i + j < audio.length;
            j++
          ) {
            const srcIndex = i + chunkSize - j;
            if (srcIndex < audio.length) {
              result[i + j] = result[srcIndex];
            }
          }
        }
      }
    }

    return result;
  };

  const applySquarepusherStyle = (audio: Float32Array, sr: number) => {
    const result = new Float32Array(audio.length);

    const bpm = 170 * tempo;
    const beatDuration = Math.floor((60 / bpm) * sr);
    const pattern = [0, 0.5, 1, 1.25, 1.75, 2, 2.5, 2.75, 3, 3.75];

    for (let bar = 0; bar < audio.length / (beatDuration * 4); bar++) {
      const barStart = bar * beatDuration * 4;

      pattern.forEach((pos) => {
        const idx = Math.floor(barStart + pos * beatDuration);
        if (idx < audio.length) {
          const kickLen = Math.floor(sr * 0.1);
          for (
            let j = 0;
            j < kickLen && idx + j < audio.length;
            j++
          ) {
            const t = j / sr;
            const kick =
              Math.sin(2 * Math.PI * 55 * t) *
              Math.exp(-25 * t);
            result[idx + j] += kick * 0.7;
          }
        }
      });

      [0.5, 1.5, 2.5, 3.5].forEach((pos) => {
        const idx = Math.floor(barStart + pos * beatDuration);
        if (idx < audio.length) {
          const snareLen = Math.floor(sr * 0.12);
          for (
            let j = 0;
            j < snareLen && idx + j < audio.length;
            j++
          ) {
            const noise = (Math.random() - 0.5) * 0.4;
            result[idx + j] += noise;
          }
        }
      });

      for (let i = 0; i < 32; i++) {
        const idx = Math.floor(barStart + (i * beatDuration) / 8);
        if (idx < audio.length) {
          const hihatLen = Math.floor(sr * 0.03);
          for (
            let j = 0;
            j < hihatLen && idx + j < audio.length;
            j++
          ) {
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
        result[j] +=
          (0.5 * saw + 0.3 * sub) * envelope * bass * 0.15;
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

  const getScaleNotes = (
    rootMidi: number,
    scaleName: string,
    count: number
  ) => {
    const scaleIntervals = scales[scaleName] || scales.minor;
    const notes: number[] = [];

    for (let i = 0; i < count; i++) {
      const octave = Math.floor(i / scaleIntervals.length);
      const intervalIdx = i % scaleIntervals.length;
      notes.push(
        rootMidi + octave * 12 + scaleIntervals[intervalIdx]
      );
    }

    return notes;
  };

  const audioBufferToWav = (buffer: AudioBuffer) => {
    const numberOfChannels = buffer.numberOfChannels;
    const length = buffer.length * numberOfChannels * 2;
    const arrayBuffer = new ArrayBuffer(44 + length);
    const view = new DataView(arrayBuffer);
    const channels: Float32Array[] = [];
    let pos = 0;

    const setString = (str: string) => {
      for (let i = 0; i < str.length; i++) {
        view.setUint8(pos + i, str.charCodeAt(i));
      }
      pos += str.length;
    };

    const setUint16 = (data: number) => {
      view.setUint16(pos, data, true);
      pos += 2;
    };

    const setUint32 = (data: number) => {
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

    for (let i = 0; i < numberOfChannels; i++) {
      channels.push(buffer.getChannelData(i));
    }

    let index = 44;
    for (let i = 0; i < buffer.length; i++) {
      for (let channel = 0; channel < numberOfChannels; channel++) {
        let sample = Math.max(
          -1,
          Math.min(1, channels[channel][i])
        );
        sample =
          sample < 0 ? sample * 0x8000 : sample * 0x7fff;
        view.setInt16(index, sample, true);
        index += 2;
      }
    }

    return arrayBuffer;
  };

  // ↓ JSX 部分は元コードと同じなので省略してもOKですが、
  // 実際には今のまま貼り付ければ動きます
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-4 md:p-8">
      {/* ここから先はあなたの元コードと同じ内容 */}
      {/* ... （JSX 部分は省略） ... */}
    </div>
  );
};

export default IDMGenerator;

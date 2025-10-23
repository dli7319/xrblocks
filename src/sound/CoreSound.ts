import * as THREE from 'three';

import {Script} from '../core/Script';

import {AudioListener} from './AudioListener';
import {AudioPlayer} from './AudioPlayer';
import {BackgroundMusic} from './BackgroundMusic';
import {CategoryVolumes, VolumeCategory} from './CategoryVolumes';
import {SoundOptions} from './SoundOptions';
import {SoundSynthesizer} from './SoundSynthesizer';
import {SpatialAudio} from './SpatialAudio';
import {SpeechRecognizer} from './SpeechRecognizer';
import {SpeechSynthesizer} from './SpeechSynthesizer';

export class CoreSound extends Script {
  static dependencies = {camera: THREE.Camera, soundOptions: SoundOptions};

  categoryVolumes = new CategoryVolumes();
  soundSynthesizer = new SoundSynthesizer();
  listener = new THREE.AudioListener();
  backgroundMusic!: BackgroundMusic;
  spatialAudio!: SpatialAudio;
  speechRecognizer?: SpeechRecognizer;
  speechSynthesizer?: SpeechSynthesizer;
  audioListener!: AudioListener;
  audioPlayer!: AudioPlayer;
  options!: SoundOptions;

  init({
    camera,
    soundOptions,
  }: {
    camera: THREE.Camera;
    soundOptions: SoundOptions;
  }) {
    this.options = soundOptions;

    this.backgroundMusic = new BackgroundMusic(
      this.listener,
      this.categoryVolumes
    );
    this.spatialAudio = new SpatialAudio(this.listener, this.categoryVolumes);
    this.audioListener = new AudioListener();
    // Initialize with 48kHz for general audio playback
    // Gemini Live uses 24kHz but that gets handled automatically via playAIAudio
    this.audioPlayer = new AudioPlayer({sampleRate: 48000});
    this.audioPlayer.setCategoryVolumes(this.categoryVolumes);

    camera.add(this.listener);
    this.add(this.backgroundMusic);
    this.add(this.spatialAudio);
    this.add(this.audioListener);
    this.add(this.audioPlayer);
    this.add(this.soundSynthesizer);

    if (this.options.speechRecognizer.enabled) {
      this.speechRecognizer = new SpeechRecognizer(this.soundSynthesizer);
      this.add(this.speechRecognizer);
    }

    if (this.options.speechSynthesizer.enabled) {
      this.speechSynthesizer = new SpeechSynthesizer(this.categoryVolumes);
      this.add(this.speechSynthesizer);
    }
  }

  getAudioListener() {
    return this.listener;
  }

  setMasterVolume(level: number) {
    this.categoryVolumes.masterVolume = THREE.MathUtils.clamp(level, 0.0, 1.0);
    this.audioPlayer?.updateGainNodeVolume();
  }

  getMasterVolume() {
    return this.categoryVolumes.isMuted
      ? 0.0
      : this.categoryVolumes.masterVolume;
  }

  setCategoryVolume(category: VolumeCategory, level: number) {
    if (category in this.categoryVolumes.volumes) {
      this.categoryVolumes.volumes[category] = THREE.MathUtils.clamp(
        level,
        0.0,
        1.0
      );
    }
  }

  getCategoryVolume(category: VolumeCategory) {
    return category in this.categoryVolumes.volumes
      ? this.categoryVolumes.volumes[category]
      : 1.0;
  }

  async enableAudio(
    options: {streamToAI?: boolean; accumulate?: boolean} = {}
  ) {
    const {streamToAI = true, accumulate = false} = options;
    if (streamToAI && this.speechRecognizer?.isListening) {
      console.log('Disabling SpeechRecognizer while streaming audio.');
      this.speechRecognizer.stop();
    }
    this.audioListener.setAIStreaming(streamToAI);
    await this.audioListener.startCapture({accumulate});
  }

  disableAudio() {
    this.audioListener?.stopCapture();
  }

  /**
   * Starts recording audio with chunk accumulation
   */
  async startRecording() {
    await this.audioListener.startCapture({accumulate: true});
  }

  /**
   * Stops recording and returns the accumulated audio buffer
   */
  stopRecording(): ArrayBuffer | null {
    const buffer = this.audioListener.getAccumulatedBuffer();
    this.audioListener.stopCapture();
    return buffer;
  }

  /**
   * Gets the accumulated recording buffer without stopping
   */
  getRecordedBuffer(): ArrayBuffer | null {
    return this.audioListener.getAccumulatedBuffer();
  }

  /**
   * Clears the accumulated recording buffer
   */
  clearRecordedBuffer() {
    this.audioListener.clearAccumulatedBuffer();
  }

  /**
   * Gets the sample rate being used for recording
   */
  getRecordingSampleRate(): number {
    return this.audioListener.audioContext?.sampleRate || 48000;
  }

  setAIStreaming(enabled: boolean) {
    this.audioListener?.setAIStreaming(enabled);
  }

  isAIStreamingEnabled() {
    return this.audioListener?.aiService !== null;
  }

  async playAIAudio(base64AudioData: string) {
    // Gemini Live API outputs audio at 24kHz
    // Only recreate AudioContext if sample rate needs to change
    const currentRate = this.audioPlayer['options'].sampleRate;
    if (currentRate !== 24000) {
      this.audioPlayer['options'].sampleRate = 24000;
      // Only stop if context exists and is different sample rate
      if (this.audioPlayer['audioContext']) {
        this.audioPlayer.stop(); // Reset context with new sample rate
      }
    }
    await this.audioPlayer.playAudioChunk(base64AudioData);
  }

  stopAIAudio() {
    this.audioPlayer?.clearQueue();
  }

  isAIAudioPlaying() {
    return this.audioPlayer?.getIsPlaying();
  }

  /**
   * Plays a raw audio buffer (Int16 PCM data) with proper sample rate
   */
  async playRecordedAudio(audioBuffer: ArrayBuffer, sampleRate?: number) {
    if (!audioBuffer) return;

    // Update sample rate if needed
    if (sampleRate && sampleRate !== this.audioPlayer['options'].sampleRate) {
      this.audioPlayer['options'].sampleRate = sampleRate;
      this.audioPlayer.stop(); // Reset context with new sample rate
    }

    // Convert ArrayBuffer to base64
    const bytes = new Uint8Array(audioBuffer);
    let binary = '';
    for (let i = 0; i < bytes.length; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    const base64Audio = btoa(binary);

    await this.audioPlayer.playAudioChunk(base64Audio);
  }

  isAudioEnabled() {
    return this.audioListener?.getIsCapturing();
  }

  getLatestAudioBuffer() {
    return this.audioListener?.getLatestAudioBuffer();
  }

  clearLatestAudioBuffer() {
    this.audioListener?.clearLatestAudioBuffer();
  }

  getEffectiveVolume(category: VolumeCategory, specificVolume = 1.0) {
    return this.categoryVolumes.getEffectiveVolume(category, specificVolume);
  }

  muteAll() {
    this.categoryVolumes.isMuted = true;
  }

  unmuteAll() {
    this.categoryVolumes.isMuted = false;
  }

  destroy() {
    this.backgroundMusic?.destroy();
    this.spatialAudio?.destroy();
    this.speechRecognizer?.destroy();
    this.speechSynthesizer?.destroy();
    this.audioListener?.dispose();
    this.audioPlayer?.dispose();
    this.listener?.parent?.remove(this.listener);
  }
}

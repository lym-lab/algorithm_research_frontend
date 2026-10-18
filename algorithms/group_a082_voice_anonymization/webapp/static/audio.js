// Browser-side audio helpers: decode any format the browser can play, convert to 16 kHz mono WAV,
// record from the microphone as PCM (no codec needed on the server), and draw waveforms.
const SR = 16000;

async function decodeToMono16k(arrayBuffer) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));
  ctx.close();
  return resampleTo16k(decoded);
}

async function resampleTo16k(buffer) {
  const frames = Math.max(1, Math.ceil(buffer.duration * SR));
  const off = new OfflineAudioContext(1, frames, SR);
  const src = off.createBufferSource();
  src.buffer = buffer;               // channels are mixed down by the 1-channel destination
  src.connect(off.destination);
  src.start();
  const out = await off.startRendering();
  return out.getChannelData(0).slice();
}

function encodeWav(samples, sr = SR) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

class PcmRecorder {
  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false } });
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = this.ctx.createScriptProcessor(4096, 1, 1);
    this.chunks = [];
    this.node.onaudioprocess = (e) => this.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    src.connect(this.node);
    this.node.connect(this.ctx.destination);
  }
  async stop() {
    this.node.disconnect();
    this.stream.getTracks().forEach(t => t.stop());
    const n = this.chunks.reduce((a, c) => a + c.length, 0);
    const buffer = this.ctx.createBuffer(1, Math.max(1, n), this.ctx.sampleRate);
    let o = 0;
    for (const c of this.chunks) { buffer.getChannelData(0).set(c, o); o += c.length; }
    await this.ctx.close();
    return resampleTo16k(buffer);
  }
}

function drawWave(canvas, samples, color, progress = 0) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext('2d');
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  const cols = Math.max(1, Math.floor(w / 3));
  const step = samples.length / cols;
  for (let c = 0; c < cols; c++) {
    let peak = 0;
    const a = Math.floor(c * step), b = Math.min(samples.length, Math.floor((c + 1) * step));
    for (let i = a; i < b; i++) peak = Math.max(peak, Math.abs(samples[i]));
    const bh = Math.max(1, peak * (h - 8));
    g.globalAlpha = (c / cols) < progress ? 1 : 0.55;
    g.fillStyle = color;
    g.fillRect(c * 3 + 1, (h - bh) / 2, 2, bh);
  }
}

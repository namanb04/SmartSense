"""
voice_component.py
==================
Single mic button that simultaneously:
  1. SpeechRecognition API → live transcript (for BERT text model)
  2. MediaRecorder API     → records webm audio, then converts to
                             PCM WAV in the browser using Web Audio API
                             so Librosa can decode it without ffmpeg.

WAV conversion pipeline (all in JS):
  webm blob → AudioContext.decodeAudioData() → AudioBuffer (PCM float32)
  → interleave channels → write WAV header → base64 encode → send to Python

Returns
-------
transcript : str — speech-to-text result
audio_b64  : str — base64 WAV (no header stripping needed, Librosa reads directly)
Both are empty strings if mic not used.
"""

import json
import streamlit as st


def voice_input_component() -> tuple[str, str]:
    """Renders mic UI. Returns (transcript, audio_b64_wav)."""

    params     = st.query_params
    transcript = ""
    audio_b64  = ""

    if "_voice" in params:
        try:
            payload    = json.loads(params["_voice"])
            transcript = payload.get("t", "")
            audio_b64  = payload.get("a", "")
        except Exception:
            pass

    st.components.v1.html("""
    <style>
      * { box-sizing: border-box; margin: 0; padding: 0; }
      .voice-wrapper {
        display: flex; align-items: center;
        gap: 10px; padding: 4px 0 8px 0;
      }
      #micBtn {
        width: 42px; height: 42px; border-radius: 50%;
        border: 2px solid #388bfd; background: transparent;
        color: #388bfd; font-size: 18px; cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        transition: all 0.2s ease; flex-shrink: 0;
      }
      #micBtn:hover { background: rgba(56,139,253,0.15); transform: scale(1.05); }
      #micBtn.listening {
        background: #e74c3c; border-color: #e74c3c; color: white;
        animation: pulse 1.2s infinite;
      }
      @keyframes pulse {
        0%   { box-shadow: 0 0 0 0    rgba(231,76,60,0.6); }
        70%  { box-shadow: 0 0 0 10px rgba(231,76,60,0);   }
        100% { box-shadow: 0 0 0 0    rgba(231,76,60,0);   }
      }
      #statusPill {
        font-family: sans-serif; font-size: 13px; color: #8b949e;
        padding: 4px 12px; border-radius: 20px;
        border: 1px solid #30363d; background: #161b22;
        transition: all 0.2s; white-space: nowrap;
      }
      #statusPill.listening { color: #e74c3c; border-color: #e74c3c; background: rgba(231,76,60,0.1); }
      #statusPill.done      { color: #3fb950; border-color: #3fb950; background: rgba(63,185,80,0.1); }
      #transcriptBox {
        margin-top: 8px; width: 100%; padding: 6px 10px;
        background: #1c2128; border: 1px solid #388bfd;
        border-radius: 6px; color: #c9d1d9; font-family: sans-serif;
        font-size: 13px; line-height: 1.4; word-break: break-word; min-height: 32px;
      }
      #audioNote { margin-top: 6px; font-family: sans-serif; font-size: 11px; }
      #copyBtn {
        margin-top: 8px; margin-right: 8px; background: #238636; color: white;
        border: none; border-radius: 6px; padding: 6px 16px; font-size: 13px; cursor: pointer;
      }
      #copyBtn:hover { background: #2ea043; }
      #resetBtn {
        margin-top: 8px; background: transparent; color: #8b949e;
        border: 1px solid #30363d; border-radius: 6px;
        padding: 6px 16px; font-size: 13px; cursor: pointer;
      }
      #resetBtn:hover { border-color: #e74c3c; color: #e74c3c; }
    </style>

    <div class="voice-wrapper">
      <button id="micBtn" onclick="toggleVoice()" title="Click to speak">🎙️</button>
      <span id="statusPill">Click mic to speak</span>
    </div>
    <div id="transcriptBox" style="display:none"></div>
    <div id="audioNote"     style="display:none"></div>
    <div id="btnRow"        style="display:none; margin-top:8px;">
      <button id="copyBtn"  onclick="copyText()">📋 Copy text</button>
      <button id="resetBtn" onclick="resetAll()">↺ Reset</button>
    </div>

    <script>
    // ── State ──────────────────────────────────────────────────────────────
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognition    = null;
    let mediaRecorder  = null;
    let audioChunks    = [];
    let isListening    = false;
    let fullTranscript = "";

    if (!SR) {
      pill("❌ Use Chrome or Edge", "");
      document.getElementById("micBtn").disabled = true;
    } else {
      recognition = new SR();
      recognition.continuous     = true;
      recognition.interimResults = true;
      recognition.lang           = "en-US";
      recognition.onresult = e => {
        let interim = "", final_ = "";
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const t = e.results[i][0].transcript;
          e.results[i].isFinal ? (final_ += t + " ") : (interim += t);
        }
        fullTranscript += final_;
        const box = document.getElementById("transcriptBox");
        box.style.display = "block";
        box.innerText = (fullTranscript + interim).trim();
      };
      recognition.onerror = e => { pill("❌ " + e.error, ""); isListening = false; micIdle(); };
      recognition.onend   = () => { if (isListening) recognition.start(); };
    }

    // ── Toggle ─────────────────────────────────────────────────────────────
    function toggleVoice() { isListening ? stopRecording() : startRecording(); }

    // ── Start ──────────────────────────────────────────────────────────────
    async function startRecording() {
      fullTranscript = ""; audioChunks = []; isListening = true;
      ["btnRow","audioNote"].forEach(id => document.getElementById(id).style.display = "none");
      const box = document.getElementById("transcriptBox");
      box.style.display = "none"; box.innerText = "";
      document.getElementById("micBtn").innerHTML = "⏹️";
      document.getElementById("micBtn").className = "listening";
      pill("🔴 Listening...", "listening");

      try { recognition.start(); } catch(e) {}

      try {
        const stream  = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mime    = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
                        ? "audio/webm;codecs=opus" : "audio/webm";
        mediaRecorder = new MediaRecorder(stream, { mimeType: mime });
        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) audioChunks.push(e.data); };
        mediaRecorder.start(100);
      } catch(e) {
        mediaRecorder = null;
        showAudioNote("⚠️ Audio capture unavailable — text only", "#f39c12");
      }
    }

    // ── Stop ───────────────────────────────────────────────────────────────
    function stopRecording() {
      isListening = false;
      try { recognition.stop(); } catch(e) {}
      micIdle();

      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.onstop = async () => {
          mediaRecorder.stream.getTracks().forEach(t => t.stop());

          const webmBlob  = new Blob(audioChunks, { type: mediaRecorder.mimeType });
          const arrayBuf  = await webmBlob.arrayBuffer();

          // ── Convert webm → PCM WAV using Web Audio API ─────────────────
          // This is why Librosa works without ffmpeg:
          // AudioContext.decodeAudioData() handles all codec decoding
          // natively in the browser and gives us raw float32 PCM samples.
          try {
            const audioCtx    = new AudioContext({ sampleRate: 22050 });
            const audioBuffer = await audioCtx.decodeAudioData(arrayBuf);
            const wavB64      = audioBufferToWavB64(audioBuffer);
            pushToStreamlit(fullTranscript.trim(), wavB64);
            audioCtx.close();
          } catch(err) {
            console.warn("WAV conversion failed:", err);
            // Fall back to sending empty audio — text-only mode
            pushToStreamlit(fullTranscript.trim(), "");
            showAudioNote("⚠️ Audio conversion failed — text only", "#f39c12");
          }
        };
        mediaRecorder.stop();
      } else {
        pushToStreamlit(fullTranscript.trim(), "");
      }

      if (fullTranscript.trim()) {
        document.getElementById("transcriptBox").style.display = "block";
        document.getElementById("transcriptBox").innerText     = fullTranscript.trim();
        document.getElementById("btnRow").style.display        = "block";
        pill("✅ Done! Converting audio...", "done");
      } else {
        pill("Click mic to speak", "");
      }
    }

    // ── audioBufferToWavB64 ────────────────────────────────────────────────
    // Encodes an AudioBuffer (float32 PCM) into a WAV file and returns
    // it as a base64 string. Librosa reads this directly — no ffmpeg needed.
    function audioBufferToWavB64(buffer) {
      const numChannels = 1;                        // mono — matches librosa mono=True
      const sampleRate  = buffer.sampleRate;        // 22050 (set above in AudioContext)
      const samples     = buffer.getChannelData(0); // take channel 0 (left / mono)
      const numSamples  = samples.length;

      // WAV format: 16-bit PCM
      const bytesPerSample = 2;
      const dataSize       = numSamples * bytesPerSample;
      const wavBuffer      = new ArrayBuffer(44 + dataSize);
      const view           = new DataView(wavBuffer);

      // ── WAV header ──────────────────────────────────────────────────────
      function writeStr(offset, str) {
        for (let i = 0; i < str.length; i++)
          view.setUint8(offset + i, str.charCodeAt(i));
      }
      writeStr(0,  "RIFF");
      view.setUint32(4,  36 + dataSize, true);      // file size - 8
      writeStr(8,  "WAVE");
      writeStr(12, "fmt ");
      view.setUint32(16, 16, true);                  // PCM chunk size
      view.setUint16(20, 1,  true);                  // PCM format
      view.setUint16(22, numChannels, true);
      view.setUint32(24, sampleRate,  true);
      view.setUint32(28, sampleRate * numChannels * bytesPerSample, true); // byte rate
      view.setUint16(32, numChannels * bytesPerSample, true);              // block align
      view.setUint16(34, 16, true);                  // bits per sample
      writeStr(36, "data");
      view.setUint32(40, dataSize, true);

      // ── PCM samples (float32 → int16) ───────────────────────────────────
      let offset = 44;
      for (let i = 0; i < numSamples; i++) {
        // Clamp to [-1, 1] then scale to int16 range
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
      }

      // ── Convert ArrayBuffer → base64 ────────────────────────────────────
      const bytes  = new Uint8Array(wavBuffer);
      let   binary = "";
      for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
      return btoa(binary);
    }

    // ── Push to Streamlit via query param ──────────────────────────────────
    function pushToStreamlit(transcript, audioB64) {
      if (!transcript && !audioB64) return;
      const payload = JSON.stringify({ t: transcript, a: audioB64 });
      const url = new URL(window.parent.location.href);
      url.searchParams.set("_voice", payload);
      window.parent.history.replaceState({}, "", url.toString());
      window.parent.dispatchEvent(new Event("popstate"));

      // Update status after audio is ready
      if (audioB64) {
        pill("✅ Done! (audio + text captured)", "done");
        showAudioNote("🎵 WAV audio ready for voice analysis", "#3fb950");
      } else {
        pill("✅ Done! (text only)", "done");
      }
    }

    // ── Helpers ────────────────────────────────────────────────────────────
    function pill(text, cls) {
      const p = document.getElementById("statusPill");
      p.innerText = text; p.className = cls;
    }
    function micIdle() {
      document.getElementById("micBtn").innerHTML = "🎙️";
      document.getElementById("micBtn").className = "";
    }
    function showAudioNote(text, color) {
      const n = document.getElementById("audioNote");
      n.style.display = "block"; n.innerText = text; n.style.color = color;
    }
    function copyText() {
      navigator.clipboard.writeText(fullTranscript.trim()).then(() => {
        const btn = document.getElementById("copyBtn");
        btn.innerText = "✅ Copied!";
        setTimeout(() => { btn.innerText = "📋 Copy text"; }, 2000);
      });
    }
    function resetAll() {
      fullTranscript = ""; audioChunks = []; isListening = false;
      try { recognition.abort(); } catch(e) {}
      try { if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop(); } catch(e) {}
      micIdle();
      ["btnRow","audioNote","transcriptBox"].forEach(id => {
        document.getElementById(id).style.display = "none";
      });
      document.getElementById("transcriptBox").innerText = "";
      pill("Click mic to speak", "");
      const url = new URL(window.parent.location.href);
      url.searchParams.delete("_voice");
      window.parent.history.replaceState({}, "", url.toString());
    }
    </script>
    """, height=210)

    return transcript, audio_b64
// ══════════════════════════════════════════════
// ── Audio recording with waveform visualizer ─
// ══════════════════════════════════════════════
let mediaRecorder = null;
let audioChunks = [];
let audioContext = null;
let analyser = null;
let vizAnimFrame = null;
let vizTimerInterval = null;
let vizStartTime = 0;

function startVisualizer(stream) {
    const container = document.getElementById('visualizer-container');
    const canvas = document.getElementById('audio-visualizer');
    container.classList.add('active');

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);

    const bufLen = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufLen);
    const ctx = canvas.getContext('2d');

    // Match canvas to CSS size
    canvas.width = canvas.offsetWidth * (window.devicePixelRatio || 1);
    canvas.height = canvas.offsetHeight * (window.devicePixelRatio || 1);
    ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;

    function draw() {
        vizAnimFrame = requestAnimationFrame(draw);
        analyser.getByteFrequencyData(dataArray);

        ctx.clearRect(0, 0, W, H);
        const barW = Math.max(2, (W / bufLen) * 2.5);
        const gap = 1;
        let x = 0;
        for (let i = 0; i < bufLen; i++) {
            const v = dataArray[i] / 255;
            const barH = Math.max(2, v * H);
            // Gradient from #e94560 at top to #0f3460 at bottom
            const r = Math.round(233 * v + 15 * (1 - v));
            const g = Math.round(69 * v + 52 * (1 - v));
            const b = Math.round(96 * v + 96 * (1 - v));
            ctx.fillStyle = `rgb(${r},${g},${b})`;
            ctx.fillRect(x, H - barH, barW, barH);
            x += barW + gap;
            if (x > W) break;
        }
    }
    draw();

    // Timer
    vizStartTime = Date.now();
    const timerEl = document.getElementById('viz-timer');
    timerEl.textContent = '0:00';
    vizTimerInterval = setInterval(() => {
        const sec = Math.floor((Date.now() - vizStartTime) / 1000);
        timerEl.textContent = Math.floor(sec / 60) + ':' + String(sec % 60).padStart(2, '0');
    }, 500);
}

function stopVisualizer() {
    const container = document.getElementById('visualizer-container');
    container.classList.remove('active');
    if (vizAnimFrame) { cancelAnimationFrame(vizAnimFrame); vizAnimFrame = null; }
    if (vizTimerInterval) { clearInterval(vizTimerInterval); vizTimerInterval = null; }
    if (audioContext) { audioContext.close().catch(() => {}); audioContext = null; }
}

async function toggleMic() {
    const micBtn = document.getElementById('mic-btn');

    // If currently recording, stop
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioChunks = [];
        mediaRecorder = new MediaRecorder(stream);

        // Start waveform visualiser
        startVisualizer(stream);

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            stopVisualizer();
            micBtn.classList.remove('recording');
            micBtn.classList.add('processing');
            micBtn.textContent = '⏳';

            const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
            const form = new FormData();
            form.append('file', blob, 'recording.webm');
            // On tutor page, use auto-detect so Whisper handles both native & target languages
            const isTutor = !!document.getElementById('tutor-input');
            const lang = isTutor ? 'auto' : document.getElementById('lang-select').value;

            try {
                const resp = await fetch('/transcribe?language=' + lang, { method: 'POST', body: form, headers: authHeaders() });
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.text) {
                        lastInputWasVoice = true;
                        const targetInput = isTutor ? document.getElementById('tutor-input') : input;
                        if (dialogMode) {
                            // True dialog: auto-send immediately
                            targetInput.value = data.text;
                            if (isTutor && typeof sendTutorMessage === 'function') {
                                sendTutorMessage();
                            } else if (typeof sendMessage === 'function') {
                                sendMessage();
                            }
                        } else {
                            targetInput.value = (targetInput.value ? targetInput.value + ' ' : '') + data.text;
                            targetInput.dispatchEvent(new Event('input'));
                            targetInput.focus();
                        }
                    } else {
                        showToast('No speech detected');
                    }
                } else {
                    showToast('Transcription failed');
                }
            } catch (e) {
                showToast('Transcription error: ' + e.message);
            } finally {
                micBtn.classList.remove('processing');
                micBtn.textContent = '🎤';
            }
        };

        mediaRecorder.start();
        micBtn.classList.add('recording');
        micBtn.textContent = '⏹';
    } catch (e) {
        showToast('Microphone access denied');
    }
}

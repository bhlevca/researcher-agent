// ══════════════════════════════════════════════
// ── TTS (Text-to-Speech) engine — server-side via edge-tts ──
// ══════════════════════════════════════════════
let ttsVoices = [];
let selectedVoiceName = localStorage.getItem('ttsVoice') || 'en-US-AriaNeural';
let dialogMode = false;
let isSpeaking = false;
let lastInputWasVoice = false;   // tracks whether user used mic
let currentAudio = null;         // currently playing Audio element

async function loadVoices() {
    try {
        const resp = await fetch('/tts/voices', { headers: authHeaders() });
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        ttsVoices = await resp.json();
    } catch (e) {
        console.warn('[TTS] Failed to load voices:', e);
        document.getElementById('voice-select').innerHTML = '<option>server TTS unavailable</option>';
        return;
    }
    const sel = document.getElementById('voice-select');
    sel.innerHTML = '';

    // Sort: match current language first, then English, then rest
    const lang = (document.getElementById('lang-select')?.value || 'en').toLowerCase();
    const sorted = [...ttsVoices].sort((a, b) => {
        const aLang = a.locale.toLowerCase().startsWith(lang) ? 0 : (a.locale.startsWith('en') ? 1 : 2);
        const bLang = b.locale.toLowerCase().startsWith(lang) ? 0 : (b.locale.startsWith('en') ? 1 : 2);
        if (aLang !== bLang) return aLang - bLang;
        return a.name.localeCompare(b.name);
    });
    sorted.forEach((v) => {
        const opt = document.createElement('option');
        opt.value = v.name;
        opt.textContent = v.name + ' (' + v.locale + ', ' + v.gender + ')';
        sel.appendChild(opt);
    });

    // Restore saved preference or default
    let found = false;
    for (const opt of sel.options) {
        if (opt.value === selectedVoiceName) {
            opt.selected = true;
            found = true;
            break;
        }
    }
    if (!found && sel.options.length) {
        sel.options[0].selected = true;
        selectedVoiceName = sel.options[0].value;
    }
    console.log('[TTS] Loaded ' + ttsVoices.length + ' server voices, selected: ' + selectedVoiceName);
}

function toggleDialogMode() {
    dialogMode = !dialogMode;
    const btn = document.getElementById('dialog-toggle');
    btn.classList.toggle('active', dialogMode);
    btn.textContent = dialogMode ? '🗣️ Dialog ON' : '🗣️ Dialog';
    showToast(dialogMode ? 'Dialog mode ON — responses will be spoken' : 'Dialog mode OFF');
}

// Strip markdown/HTML/emoji for cleaner TTS (preserve hyphens for foreign words)
function textForTTS(text) {
    let t = text;
    t = t.replace(/```[\s\S]*?```/g, ' code block omitted ');  // code blocks
    t = t.replace(/`[^`]+`/g, '');            // inline code
    t = t.replace(/!\[[^\]]*\]\([^)]+\)/g, ' image shown ');  // images
    t = t.replace(/\[[^\]]+\]\([^)]+\)/g, (m) => m.replace(/\[([^\]]+)\].*/, '$1')); // links → text
    t = t.replace(/\*\*(.+?)\*\*/g, '$1');     // bold
    t = t.replace(/\*(.+?)\*/g, '$1');          // italic
    t = t.replace(/^#{1,6}\s+/gm, '');          // heading markers at line start
    t = t.replace(/^>\s?/gm, '');                // blockquote markers
    t = t.replace(/~~(.+?)~~/g, '$1');           // strikethrough
    // Emoji & emoticons
    t = t.replace(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{27BF}\u{FE00}-\u{FE0F}\u{200D}\u{20E3}\u{E0020}-\u{E007F}]/gu, '');
    t = t.replace(/[:;][-']?[)(DPpOo3><|/\\]/g, '');  // text emoticons :) ;-P etc
    // Dashes (em/en) → comma pause, but keep regular hyphens for compound words
    t = t.replace(/\s*[—–]\s*/g, ', ');
    // Bullets and decorative chars
    t = t.replace(/[•●○■□▪▸▹►▻★☆✓✗✔✘✦✧⟹→←↑↓↔⇒⇐⇑⇓]/g, '');
    // Repeated punctuation (... is ok, but !!!! or ??? → single)
    t = t.replace(/([!?]){2,}/g, '$1');
    t = t.replace(/\s+/g, ' ').trim();
    return t;
}

function speakText(text, wrapper) {
    if (!text) return Promise.resolve();
    return new Promise(async (resolve) => {
        stopSpeaking();  // stop any prior speech
        const clean = textForTTS(text);
        if (!clean) { resolve(); return; }

        // Show speaking indicator on the message
        let indicator = null;
        if (wrapper) {
            indicator = document.createElement('div');
            indicator.className = 'speaking-indicator';
            indicator.innerHTML = '<div class="bars"><span></span><span></span><span></span><span></span></div> Speaking...';
            wrapper.appendChild(indicator);
        }
        isSpeaking = true;
        document.getElementById('stop-speak-btn').classList.add('visible');

        try {
            const resp = await fetch('/tts/speak', {
                method: 'POST',
                headers: jsonAuthHeaders(),
                body: JSON.stringify({ text: clean, voice: selectedVoiceName })
            });
            if (!resp.ok) throw new Error('TTS server error');
            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);
            currentAudio = audio;
            audio.onended = () => {
                URL.revokeObjectURL(url);
                currentAudio = null;
                if (indicator) indicator.remove();
                isSpeaking = false;
                document.getElementById('stop-speak-btn').classList.remove('visible');
                resolve();
            };
            audio.onerror = () => {
                URL.revokeObjectURL(url);
                currentAudio = null;
                if (indicator) indicator.remove();
                isSpeaking = false;
                document.getElementById('stop-speak-btn').classList.remove('visible');
                resolve();
            };
            audio.play();
        } catch (e) {
            console.error('[TTS] Error:', e);
            if (indicator) indicator.remove();
            isSpeaking = false;
            document.getElementById('stop-speak-btn').classList.remove('visible');
            resolve();
        }
    });
}

function stopSpeaking() {
    isSpeaking = false;
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
        currentAudio = null;
    }
    document.getElementById('stop-speak-btn').classList.remove('visible');
    // Remove any speaking indicators
    document.querySelectorAll('.speaking-indicator').forEach(el => el.remove());
}

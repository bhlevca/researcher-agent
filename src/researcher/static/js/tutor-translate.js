// ══════════════════════════════════════════════
// ── Translator ──
// ══════════════════════════════════════════════

let translateHistory = [];

// Full language name → Whisper / speech-recognition 2-letter code
const _langToCode = {
    English: 'en', French: 'fr', German: 'de', Spanish: 'es', Italian: 'it',
    Portuguese: 'pt', Dutch: 'nl', Russian: 'ru', Chinese: 'zh', Japanese: 'ja',
    Romanian: 'ro', Korean: 'ko', Arabic: 'ar', Turkish: 'tr', Polish: 'pl',
    Swedish: 'sv', Greek: 'el', Hindi: 'hi',
};

function _allLangs() {
    return Object.keys(_langToCode);
}

function initTranslateSelects() {
    const srcSel = document.getElementById('translate-source-lang');
    const tgtSel = document.getElementById('translate-target-lang');
    if (!srcSel || !tgtSel) return;
    if (srcSel.options.length > 0) return; // already populated

    const langs = _allLangs();
    const nativeLang = document.getElementById('tutor-native-lang')?.value || 'English';
    const targetLang = document.getElementById('tutor-target-lang')?.value || 'French';

    langs.forEach(l => {
        const o1 = document.createElement('option');
        o1.value = l; o1.textContent = l;
        if (l === nativeLang) o1.selected = true;
        srcSel.appendChild(o1);

        const o2 = document.createElement('option');
        o2.value = l; o2.textContent = l;
        if (l === targetLang) o2.selected = true;
        tgtSel.appendChild(o2);
    });
}

function swapTranslateLangs() {
    const src = document.getElementById('translate-source-lang');
    const tgt = document.getElementById('translate-target-lang');
    const tmp = src.value;
    src.value = tgt.value;
    tgt.value = tmp;
    // Also swap text if there's a translation
    const input = document.getElementById('translate-input');
    const output = document.getElementById('translate-output');
    if (output.dataset.text) {
        input.value = output.dataset.text;
        output.textContent = 'Translation will appear here';
        output.dataset.text = '';
    }
}

function clearTranslateInput() {
    document.getElementById('translate-input').value = '';
    const output = document.getElementById('translate-output');
    output.textContent = 'Translation will appear here';
    output.dataset.text = '';
}

// Options object: { autoSpeak: true } to auto-play TTS on result
async function doTranslate(opts) {
    const text = document.getElementById('translate-input').value.trim();
    if (!text) { tutorShowToast('Enter text to translate'); return; }

    const srcLang = document.getElementById('translate-source-lang').value;
    const tgtLang = document.getElementById('translate-target-lang').value;
    if (srcLang === tgtLang) { tutorShowToast('Source and target are the same'); return; }

    const btn = document.getElementById('translate-btn');
    const output = document.getElementById('translate-output');
    btn.disabled = true;
    btn.textContent = '⏳ Translating...';
    output.textContent = 'Translating...';
    output.dataset.text = '';

    try {
        const resp = await fetch('/tutor/translate', {
            method: 'POST',
            headers: tutorJsonHeaders(),
            body: JSON.stringify({ text, source_lang: srcLang, target_lang: tgtLang }),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || 'Translation failed');
        }
        const data = await resp.json();
        output.textContent = data.translation;
        output.dataset.text = data.translation;

        // Add to history
        translateHistory.unshift({
            source: text, translation: data.translation,
            srcLang, tgtLang, time: new Date().toLocaleTimeString(),
        });
        if (translateHistory.length > 20) translateHistory.pop();
        renderTranslateHistory();

        // Auto-speak the translation (e.g. after voice input)
        if (opts && opts.autoSpeak && data.translation) {
            speakTranslation('target');
        }
    } catch (e) {
        output.textContent = 'Error: ' + e.message;
        tutorShowToast('Translation error: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '🔄 Translate';
    }
}

function renderTranslateHistory() {
    const container = document.getElementById('translate-history');
    if (!container || translateHistory.length === 0) { if (container) container.innerHTML = ''; return; }

    let html = '<h4 style="color:#64748b;font-size:0.78rem;margin:0.5rem 0;">Recent translations</h4>';
    translateHistory.forEach((h, i) => {
        html += '<div class="translate-history-item" onclick="useTranslateHistory(' + i + ')">' +
            '<div class="th-langs">' + escapeHtml(h.srcLang) + ' → ' + escapeHtml(h.tgtLang) +
            '<span class="th-time">' + h.time + '</span></div>' +
            '<div class="th-source">' + escapeHtml(h.source.substring(0, 80)) + '</div>' +
            '<div class="th-target">' + escapeHtml(h.translation.substring(0, 80)) + '</div>' +
            '</div>';
    });
    container.innerHTML = html;
}

function useTranslateHistory(idx) {
    const h = translateHistory[idx];
    if (!h) return;
    document.getElementById('translate-source-lang').value = h.srcLang;
    document.getElementById('translate-target-lang').value = h.tgtLang;
    document.getElementById('translate-input').value = h.source;
    const output = document.getElementById('translate-output');
    output.textContent = h.translation;
    output.dataset.text = h.translation;
}

function speakTranslation(which) {
    if (which === 'source') {
        const text = document.getElementById('translate-input').value.trim();
        if (text) speakText(text);
        else tutorShowToast('No source text');
    } else {
        const text = document.getElementById('translate-output').dataset.text || '';
        if (text) speakText(text);
        else tutorShowToast('No translation yet');
    }
}

function copyTranslation() {
    const text = document.getElementById('translate-output').dataset.text || '';
    if (!text) { tutorShowToast('Nothing to copy'); return; }
    navigator.clipboard.writeText(text).then(
        () => tutorShowToast('Copied!'),
        () => tutorShowToast('Copy failed')
    );
}

// ── Voice-to-translate: record → transcribe in source lang → auto-translate ──

let _translateRecorder = null;
let _translateChunks = [];

async function translateVoiceInput() {
    const micBtn = document.getElementById('translate-mic-btn');

    // If already recording, stop
    if (_translateRecorder && _translateRecorder.state === 'recording') {
        _translateRecorder.stop();
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        _translateChunks = [];
        _translateRecorder = new MediaRecorder(stream);

        // Reuse visualizer if available
        if (typeof startVisualizer === 'function') startVisualizer(stream);

        _translateRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) _translateChunks.push(e.data);
        };

        _translateRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            if (typeof stopVisualizer === 'function') stopVisualizer();
            if (micBtn) { micBtn.classList.remove('recording'); micBtn.textContent = '🎤'; }

            const blob = new Blob(_translateChunks, { type: _translateRecorder.mimeType });
            const form = new FormData();
            form.append('file', blob, 'recording.webm');

            // Use the source language from the translate panel for transcription
            const srcLangName = document.getElementById('translate-source-lang')?.value || 'English';
            const langCode = _langToCode[srcLangName] || 'en';

            const statusEl = document.getElementById('translate-input');
            const origPlaceholder = statusEl.placeholder;
            statusEl.placeholder = '⏳ Transcribing...';

            try {
                const resp = await fetch('/transcribe?language=' + langCode, {
                    method: 'POST', body: form, headers: tutorAuthHeaders(),
                });
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.text) {
                        statusEl.value = data.text;
                        // Auto-translate + auto-speak after voice input
                        await doTranslate({ autoSpeak: true });
                    } else {
                        tutorShowToast('No speech detected');
                    }
                } else {
                    tutorShowToast('Transcription failed');
                }
            } catch (e) {
                tutorShowToast('Transcription error: ' + e.message);
            } finally {
                statusEl.placeholder = origPlaceholder;
            }
        };

        _translateRecorder.start();
        if (micBtn) { micBtn.classList.add('recording'); micBtn.textContent = '⏹'; }
    } catch (e) {
        tutorShowToast('Microphone access denied');
    }
}

// Keyboard shortcut: Ctrl+Enter to translate
document.getElementById('translate-input')?.addEventListener('keydown', function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        doTranslate();
    }
});

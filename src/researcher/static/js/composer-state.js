// ══════════════════════════════════════════════
// ── Composer state & helpers ──
// ══════════════════════════════════════════════
let composerSessionId = null;
let composerSessionName = null;
let composerChatHistory = [];
let composerScoreRawResponse = '';

const composerChat = document.getElementById('composer-chat');
const composerInput = document.getElementById('composer-input');
const composerSendBtn = document.getElementById('composer-send-btn');

function composerAuthHeaders() {
    const h = {};
    const token = localStorage.getItem('authToken');
    if (token) h['Authorization'] = 'Bearer ' + token;
    return h;
}

function composerJsonHeaders() {
    return { 'Content-Type': 'application/json', ...composerAuthHeaders() };
}

function getComposerConfig() {
    return {
        genre: document.getElementById('composer-genre').value,
        key_signature: document.getElementById('composer-key').value,
        time_signature: document.getElementById('composer-time').value,
        tempo: parseInt(document.getElementById('composer-tempo').value) || 120,
    };
}

function composerShowToast(msg) {
    if (typeof showToast === 'function') { showToast(msg); return; }
    const t = document.getElementById('toast');
    if (!t) { alert(msg); return; }
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 3000);
}

function updateComposerSessionUI() {
    const el = document.getElementById('composer-active-session');
    if (composerSessionId) {
        el.style.display = 'inline';
        el.textContent = '🎵 ' + (composerSessionName || composerSessionId);
    } else {
        el.style.display = 'none';
    }
}

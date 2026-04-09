// ══════════════════════════════════════════════
// ── Tutor state & helpers ──
// ══════════════════════════════════════════════
let tutorSessionId = null;
let tutorSessionName = null;
let tutorChatHistory = [];
let tutorQuizData = null;       // current quiz questions from server
let tutorQuizRawResponse = '';  // raw markdown from quiz generation
let tutorLessonRawResponse = ''; // raw markdown from lesson generation

const tutorChat = document.getElementById('tutor-chat');
const tutorInput = document.getElementById('tutor-input');
const tutorSendBtn = document.getElementById('tutor-send-btn');

function tutorAuthHeaders() {
    const h = {};
    const token = localStorage.getItem('authToken');
    if (token) h['Authorization'] = 'Bearer ' + token;
    return h;
}

function tutorJsonHeaders() {
    return { 'Content-Type': 'application/json', ...tutorAuthHeaders() };
}

function getTutorConfig() {
    return {
        target_lang: document.getElementById('tutor-target-lang').value,
        native_lang: document.getElementById('tutor-native-lang').value,
        level: document.getElementById('tutor-level').value,
    };
}

function tutorShowToast(msg) {
    // Reuse main showToast if available, else simple alert
    if (typeof showToast === 'function') { showToast(msg); return; }
    const t = document.getElementById('toast');
    if (!t) { alert(msg); return; }
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 3000);
}

// ══════════════════════════════════════════════
// ── Tutor tab switching ──
// ══════════════════════════════════════════════
function switchTutorTab(tabName) {
    document.querySelectorAll('.tutor-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tutor-panel').forEach(p => p.classList.remove('active'));
    document.querySelector(`.tutor-tab[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById('panel-' + tabName).classList.add('active');

    // Refresh data when switching to certain tabs
    if (tabName === 'vocab') loadVocabulary();
    if (tabName === 'lessons' && tutorSessionId) loadLessons();
    if (tabName === 'quiz' && tutorSessionId) loadQuizHistory();
    if (tabName === 'stats' && tutorSessionId) loadStats();
    if (tabName === 'translate' && typeof initTranslateSelects === 'function') initTranslateSelects();
}

// ══════════════════════════════════════════════
// ── Tutor session management ──
// ══════════════════════════════════════════════
function openTutorSessionsPanel() {
    document.getElementById('tutor-sessions-panel').classList.add('open');
    document.getElementById('tutor-sessions-overlay').classList.add('open');
    loadTutorSessions();
}

function closeTutorSessionsPanel() {
    document.getElementById('tutor-sessions-panel').classList.remove('open');
    document.getElementById('tutor-sessions-overlay').classList.remove('open');
}

async function loadTutorSessions() {
    const list = document.getElementById('tutor-sessions-list');
    try {
        const resp = await fetch('/tutor/sessions', { headers: tutorAuthHeaders() });
        const data = await resp.json();
        if (!data.sessions || data.sessions.length === 0) {
            list.innerHTML = '<div class="no-sessions">No tutor sessions</div>';
            return;
        }
        list.innerHTML = '';
        data.sessions.forEach(s => {
            const item = document.createElement('div');
            item.className = 'session-item' + (s.id === tutorSessionId ? ' active' : '');
            const nm = document.createElement('div');
            nm.className = 'session-item-name';
            nm.textContent = s.name + ' — ' + s.target_language + ' (' + s.level + ')';
            item.appendChild(nm);
            const meta = document.createElement('div');
            meta.className = 'session-item-meta';
            const dt = new Date(s.updated_at || s.created_at);
            meta.textContent = dt.toLocaleString() + ' · ' + s.message_count + ' msgs';
            item.appendChild(meta);
            const acts = document.createElement('div');
            acts.className = 'session-item-actions';
            const loadBtn = document.createElement('button');
            loadBtn.textContent = '▶ Load';
            loadBtn.onclick = (e) => { e.stopPropagation(); loadTutorSession(s.id); };
            acts.appendChild(loadBtn);
            const delBtn = document.createElement('button');
            delBtn.className = 'delete';
            delBtn.textContent = '✕ Delete';
            delBtn.onclick = (e) => { e.stopPropagation(); deleteTutorSession(s.id, s.name); };
            acts.appendChild(delBtn);
            item.appendChild(acts);
            item.onclick = () => loadTutorSession(s.id);
            list.appendChild(item);
        });
    } catch (e) {
        list.innerHTML = '<div class="no-sessions">Error loading sessions</div>';
    }
}

async function createTutorSession() {
    const cfg = getTutorConfig();
    const name = prompt('Session name:', cfg.target_lang + ' — ' + new Date().toLocaleDateString());
    if (!name) return;
    try {
        const resp = await fetch('/tutor/sessions', {
            method: 'POST',
            headers: tutorJsonHeaders(),
            body: JSON.stringify({
                name, target_lang: cfg.target_lang,
                native_lang: cfg.native_lang, level: cfg.level,
            }),
        });
        if (resp.ok) {
            const data = await resp.json();
            tutorSessionId = data.id;
            tutorSessionName = data.name;
            tutorChatHistory = [];
            tutorChat.innerHTML = '';
            updateTutorSessionUI();
            tutorShowToast('Session created: ' + name);
        } else {
            tutorShowToast('Failed to create session');
        }
    } catch (e) { tutorShowToast('Error: ' + e.message); }
}

async function loadTutorSession(id) {
    try {
        const resp = await fetch('/tutor/sessions/' + id, { headers: tutorAuthHeaders() });
        if (!resp.ok) { tutorShowToast('Failed to load session'); return; }
        const data = await resp.json();
        tutorSessionId = data.id;
        tutorSessionName = data.name;
        tutorChatHistory = data.messages || [];

        // Set config dropdowns
        document.getElementById('tutor-target-lang').value = data.target_language || 'French';
        document.getElementById('tutor-native-lang').value = data.native_language || 'English';
        document.getElementById('tutor-level').value = data.level || 'A1';

        // Render chat history
        tutorChat.innerHTML = '';
        tutorChatHistory.forEach(msg => {
            if (msg.role === 'user') {
                addTutorMsg('user', msg.text);
            } else {
                addTutorMsg('assistant', msg.text);
            }
        });
        updateTutorSessionUI();
        closeTutorSessionsPanel();
        tutorShowToast('Loaded: ' + data.name);
    } catch (e) { tutorShowToast('Error loading session'); }
}

async function saveTutorSession() {
    if (!tutorSessionId) { tutorShowToast('Create a session first'); return; }
    if (tutorChatHistory.length === 0) { tutorShowToast('Nothing to save'); return; }
    try {
        const resp = await fetch('/tutor/sessions/' + tutorSessionId, {
            method: 'PUT',
            headers: tutorJsonHeaders(),
            body: JSON.stringify({
                name: tutorSessionName,
                messages: tutorChatHistory,
            }),
        });
        if (resp.ok) tutorShowToast('Session saved');
        else tutorShowToast('Failed to save');
    } catch (e) { tutorShowToast('Error saving session'); }
}

async function deleteTutorSession(id, name) {
    if (!confirm('Delete session "' + name + '"?')) return;
    try {
        const resp = await fetch('/tutor/sessions/' + id, { method: 'DELETE', headers: tutorAuthHeaders() });
        if (resp.ok) {
            if (tutorSessionId === id) {
                tutorSessionId = null;
                tutorSessionName = null;
                tutorChatHistory = [];
                tutorChat.innerHTML = '';
                updateTutorSessionUI();
            }
            loadTutorSessions();
            tutorShowToast('Deleted: ' + name);
        }
    } catch (e) { tutorShowToast('Error deleting session'); }
}

function updateTutorSessionUI() {
    const el = document.getElementById('tutor-active-session');
    if (tutorSessionId) {
        el.textContent = '📖 ' + tutorSessionName;
        el.style.display = '';
    } else {
        el.textContent = '';
        el.style.display = 'none';
    }
}

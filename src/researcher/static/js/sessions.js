// ── Session management ──
function newSession() {
    if (chatHistory.length > 0 && !confirm('Start a new session? Unsaved messages will be lost.')) return;
    chatHistory = [];
    currentSessionId = null;
    currentSessionName = null;
    chat.innerHTML = '';
    clearSessionFiles();
    showToast('New session started');
}

async function saveSession() {
    if (chatHistory.length === 0) { showToast('Nothing to save'); return; }
    const name = prompt('Session name:', currentSessionName || 'Session ' + new Date().toLocaleDateString());
    if (!name) return;
    try {
        let resp;
        if (currentSessionId) {
            resp = await fetch('/sessions/' + currentSessionId, {
                method: 'PUT',
                headers: jsonAuthHeaders(),
                body: JSON.stringify({ name: name, messages: chatHistory }),
            });
        } else {
            resp = await fetch('/sessions', {
                method: 'POST',
                headers: jsonAuthHeaders(),
                body: JSON.stringify({ name: name, messages: chatHistory }),
            });
        }
        if (resp.ok) {
            const data = await resp.json();
            currentSessionId = data.id;
            currentSessionName = name;
            showToast('Session saved: ' + name);
        } else { showToast('Failed to save session'); }
    } catch (e) { showToast('Error saving session'); }
}

function openSessionsPanel() {
    document.getElementById('sessions-panel').classList.add('open');
    document.getElementById('sessions-overlay').classList.add('open');
    loadSessionsList();
}

function closeSessionsPanel() {
    document.getElementById('sessions-panel').classList.remove('open');
    document.getElementById('sessions-overlay').classList.remove('open');
}

async function loadSessionsList() {
    const list = document.getElementById('sessions-list');
    try {
        const resp = await fetch('/sessions', { headers: authHeaders() });
        const data = await resp.json();
        if (data.sessions.length === 0) {
            list.innerHTML = '<div class="no-sessions">No saved sessions</div>';
            return;
        }
        list.innerHTML = '';
        data.sessions.forEach(s => {
            const item = document.createElement('div');
            item.className = 'session-item' + (s.id === currentSessionId ? ' active' : '');
            const nm = document.createElement('div');
            nm.className = 'session-item-name';
            nm.textContent = s.name;
            item.appendChild(nm);
            const meta = document.createElement('div');
            meta.className = 'session-item-meta';
            const dt = new Date(s.updated_at || s.created_at);
            meta.textContent = dt.toLocaleString() + ' \u00b7 ' + s.message_count + ' msgs' + (s.model ? ' \u00b7 ' + s.model : '');
            item.appendChild(meta);
            const acts = document.createElement('div');
            acts.className = 'session-item-actions';
            const loadBtn = document.createElement('button');
            loadBtn.textContent = '\u25b6 Load';
            loadBtn.onclick = (e) => { e.stopPropagation(); restoreSession(s.id); };
            acts.appendChild(loadBtn);
            const delBtn = document.createElement('button');
            delBtn.className = 'delete';
            delBtn.textContent = '\u2715 Delete';
            delBtn.onclick = (e) => { e.stopPropagation(); deleteSession(s.id, s.name); };
            acts.appendChild(delBtn);
            item.appendChild(acts);
            item.onclick = () => restoreSession(s.id);
            list.appendChild(item);
        });
    } catch (e) {
        list.innerHTML = '<div class="no-sessions">Error loading sessions</div>';
    }
}

async function restoreSession(id) {
    try {
        const resp = await fetch('/sessions/' + id, { headers: authHeaders() });
        if (!resp.ok) { showToast('Failed to load session'); return; }
        const data = await resp.json();
        chatHistory = data.messages || [];
        currentSessionId = data.id;
        currentSessionName = data.name;
        chat.innerHTML = '';
        chatHistory.forEach(msg => {
            if (msg.role === 'user') {
                addSimpleMsg('user', msg.text);
            } else if (msg.role === 'assistant') {
                addAssistantMessage(msg.text, msg.reasoning || [], msg.tokenUsage || {});
            }
        });
        closeSessionsPanel();
        showToast('Loaded: ' + data.name);
    } catch (e) { showToast('Error loading session'); }
}

async function deleteSession(id, name) {
    if (!confirm('Delete session "' + name + '"?')) return;
    try {
        const resp = await fetch('/sessions/' + id, { method: 'DELETE', headers: authHeaders() });
        if (resp.ok) {
            if (currentSessionId === id) { currentSessionId = null; currentSessionName = null; }
            loadSessionsList();
            showToast('Deleted: ' + name);
        }
    } catch (e) { showToast('Error deleting session'); }
}

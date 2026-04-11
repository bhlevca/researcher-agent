// ══════════════════════════════════════════════
// ── Composer tab switching ──
// ══════════════════════════════════════════════
function switchComposerTab(tabName) {
    document.querySelectorAll('.composer-tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.composer-panel').forEach(p => p.classList.remove('active'));
    document.querySelector(`.composer-tab[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById('cpanel-' + tabName).classList.add('active');

    if (tabName === 'compositions' && composerSessionId) loadCompositions();
}

// ══════════════════════════════════════════════
// ── Composer session management ──
// ══════════════════════════════════════════════
function openComposerSessionsPanel() {
    document.getElementById('composer-sessions-panel').classList.add('open');
    document.getElementById('composer-sessions-overlay').classList.add('open');
    loadComposerSessions();
}

function closeComposerSessionsPanel() {
    document.getElementById('composer-sessions-panel').classList.remove('open');
    document.getElementById('composer-sessions-overlay').classList.remove('open');
}

async function loadComposerSessions() {
    const list = document.getElementById('composer-sessions-list');
    try {
        const resp = await fetch('/composer/sessions', { headers: composerAuthHeaders() });
        const data = await resp.json();
        if (!data.sessions || data.sessions.length === 0) {
            list.innerHTML = '<div class="no-sessions">No composer sessions</div>';
            return;
        }
        list.innerHTML = '';
        data.sessions.forEach(s => {
            const item = document.createElement('div');
            item.className = 'session-item' + (s.id === composerSessionId ? ' active' : '');
            const nm = document.createElement('div');
            nm.className = 'session-item-name';
            nm.textContent = s.name + ' — ' + s.genre + ' (' + s.key_signature + ')';
            item.appendChild(nm);
            const meta = document.createElement('div');
            meta.className = 'session-item-meta';
            const dt = new Date(s.updated_at || s.created_at);
            meta.textContent = dt.toLocaleString() + ' · ' + s.time_signature + ' · ' + s.tempo + ' BPM';
            item.appendChild(meta);
            const acts = document.createElement('div');
            acts.className = 'session-item-actions';
            const loadBtn = document.createElement('button');
            loadBtn.textContent = '▶ Load';
            loadBtn.onclick = (e) => { e.stopPropagation(); loadComposerSession(s.id); };
            acts.appendChild(loadBtn);
            const delBtn = document.createElement('button');
            delBtn.className = 'delete';
            delBtn.textContent = '✕ Delete';
            delBtn.onclick = (e) => { e.stopPropagation(); deleteComposerSession(s.id, s.name); };
            acts.appendChild(delBtn);
            item.appendChild(acts);
            item.onclick = () => loadComposerSession(s.id);
            list.appendChild(item);
        });
    } catch (e) {
        list.innerHTML = '<div class="no-sessions">Error loading sessions</div>';
    }
}

async function createComposerSession() {
    const cfg = getComposerConfig();
    const name = prompt('Session name:', cfg.genre + ' — ' + new Date().toLocaleDateString());
    if (!name) return;
    try {
        const resp = await fetch('/composer/sessions', {
            method: 'POST',
            headers: composerJsonHeaders(),
            body: JSON.stringify({
                name,
                genre: cfg.genre,
                key_signature: cfg.key_signature,
                time_signature: cfg.time_signature,
                tempo: cfg.tempo,
            }),
        });
        if (resp.ok) {
            const data = await resp.json();
            composerSessionId = data.id;
            composerSessionName = data.name;
            composerChatHistory = [];
            composerChat.innerHTML = '';
            updateComposerSessionUI();
            composerShowToast('Session created: ' + name);
        } else {
            composerShowToast('Failed to create session');
        }
    } catch (e) { composerShowToast('Error: ' + e.message); }
}

async function loadComposerSession(id) {
    try {
        const resp = await fetch('/composer/sessions/' + id, { headers: composerAuthHeaders() });
        if (!resp.ok) { composerShowToast('Failed to load session'); return; }
        const data = await resp.json();
        composerSessionId = data.id;
        composerSessionName = data.name;
        composerChatHistory = data.messages || [];

        // Set config dropdowns
        document.getElementById('composer-genre').value = data.genre || 'classical';
        document.getElementById('composer-key').value = data.key_signature || 'C major';
        document.getElementById('composer-time').value = data.time_signature || '4/4';
        document.getElementById('composer-tempo').value = data.tempo || 120;

        // Render chat history
        composerChat.innerHTML = '';
        composerChatHistory.forEach(msg => {
            if (msg.role === 'user') {
                addComposerMsg('user', msg.text);
            } else {
                addComposerMsg('assistant', msg.text, msg.reasoning);
            }
        });

        updateComposerSessionUI();
        closeComposerSessionsPanel();
        composerShowToast('Loaded: ' + data.name);
    } catch (e) {
        composerShowToast('Error: ' + e.message);
    }
}

async function saveComposerSession() {
    if (!composerSessionId) { composerShowToast('Create a session first'); return; }
    try {
        const resp = await fetch('/composer/sessions/' + composerSessionId, {
            method: 'PUT',
            headers: composerJsonHeaders(),
            body: JSON.stringify({
                name: composerSessionName || 'Untitled',
                messages: composerChatHistory,
            }),
        });
        if (resp.ok) {
            composerShowToast('Session saved');
        } else {
            composerShowToast('Failed to save session');
        }
    } catch (e) { composerShowToast('Error: ' + e.message); }
}

async function deleteComposerSession(id, name) {
    if (!confirm('Delete session "' + name + '"?')) return;
    try {
        const resp = await fetch('/composer/sessions/' + id, {
            method: 'DELETE',
            headers: composerAuthHeaders(),
        });
        if (resp.ok) {
            if (id === composerSessionId) {
                composerSessionId = null;
                composerSessionName = null;
                composerChatHistory = [];
                composerChat.innerHTML = '';
                updateComposerSessionUI();
            }
            loadComposerSessions();
            composerShowToast('Session deleted');
        } else {
            composerShowToast('Failed to delete session');
        }
    } catch (e) { composerShowToast('Error: ' + e.message); }
}

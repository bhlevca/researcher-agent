// ══════════════════════════════════════════════
// ── Score generation ──
// ══════════════════════════════════════════════

async function generateScore() {
    if (!composerSessionId) { composerShowToast('Create or load a session first'); return; }

    const desc = document.getElementById('score-description').value.trim();
    if (!desc) { composerShowToast('Enter a description of what to compose'); return; }

    const instruments = getSelectedInstruments();
    const measures = parseInt(document.getElementById('score-measures').value) || 16;

    const content = document.getElementById('score-content');
    content.innerHTML = '';

    // Show streaming indicator
    const label = document.createElement('div');
    label.className = 'composer-streaming-label';
    label.innerHTML = '<span class="pulse"></span> Generating score...';
    content.appendChild(label);
    const liveBox = document.createElement('div');
    liveBox.className = 'composer-streaming';
    content.appendChild(liveBox);

    let reasoningLines = [];

    try {
        const resp = await fetch('/composer/score', {
            method: 'POST',
            headers: composerJsonHeaders(),
            body: JSON.stringify({
                session_id: composerSessionId,
                description: desc,
                instruments: instruments,
                measures: measures,
                style: '',
            }),
        });

        if (!resp.ok) {
            content.innerHTML = '<div class="message error">Error: ' + resp.statusText + '</div>';
            return;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            let lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    var currentEvent = line.slice(7).trim();
                } else if (line.startsWith('data: ') && currentEvent) {
                    const data = JSON.parse(line.slice(6));
                    if (currentEvent === 'reasoning') {
                        reasoningLines.push(data);
                        liveBox.textContent = reasoningLines.join('\n');
                        liveBox.scrollTop = liveBox.scrollHeight;
                    } else if (currentEvent === 'done') {
                        content.innerHTML = '';
                        composerScoreRawResponse = data.response || '';
                        renderScoreResponse(data.response, data.reasoning, content, data.has_score);
                    } else if (currentEvent === 'error') {
                        content.innerHTML = '<div class="message error">Error: ' + data + '</div>';
                    }
                    currentEvent = null;
                }
            }
        }
    } catch (e) {
        content.innerHTML = '<div class="message error">Connection error: ' + e.message + '</div>';
    }
}

function renderScoreResponse(text, reasoning, container, hasScore) {
    // Render the markdown portion
    const div = document.createElement('div');
    div.className = 'message assistant';
    div.innerHTML = renderContent(text);
    container.appendChild(div);

    // Add action buttons
    const acts = document.createElement('div');
    acts.className = 'score-actions';

    const copyBtn = document.createElement('button');
    copyBtn.textContent = '📋 Copy';
    copyBtn.onclick = () => navigator.clipboard.writeText(div.innerText || text).then(() => composerShowToast('Copied'));
    acts.appendChild(copyBtn);

    const saveTxtBtn = document.createElement('button');
    saveTxtBtn.textContent = '💾 Save';
    saveTxtBtn.onclick = () => composerSaveText(div.innerText || text, null);
    acts.appendChild(saveTxtBtn);

    if (reasoning && reasoning.length > 0) {
        const copyAll = document.createElement('button');
        copyAll.textContent = '📋 Copy all';
        copyAll.onclick = () => {
            const full = '=== Response ===\n' + (div.innerText || text) + '\n\n=== Reasoning ===\n' + reasoning.join('\n');
            navigator.clipboard.writeText(full).then(() => composerShowToast('Copied response + reasoning'));
        };
        acts.appendChild(copyAll);

        const saveAll = document.createElement('button');
        saveAll.textContent = '💾 Save all';
        saveAll.onclick = () => composerSaveText(div.innerText || text, reasoning);
        acts.appendChild(saveAll);
    }

    if (hasScore || _hasScore(text)) {
        const dlBtn = document.createElement('button');
        dlBtn.textContent = '📥 Download MusicXML';
        dlBtn.onclick = () => downloadMusicXMLFromText(text);
        acts.appendChild(dlBtn);

        const saveBtn = document.createElement('button');
        saveBtn.textContent = '🎼 Save Composition';
        saveBtn.onclick = () => saveScoreFromResponse(text);
        acts.appendChild(saveBtn);
    }
    // Reasoning (before action buttons)
    if (reasoning && reasoning.length > 0) {
        const details = document.createElement('details');
        details.className = 'reasoning-toggle';
        const summary = document.createElement('summary');
        summary.textContent = 'Reasoning (' + reasoning.length + ' steps)';
        details.appendChild(summary);
        const rc = document.createElement('div');
        rc.className = 'reasoning-content';
        rc.textContent = reasoning.join('\n');
        details.appendChild(rc);
        container.appendChild(details);
    }

    container.appendChild(acts);
}

function getSelectedInstruments() {
    const sel = document.getElementById('score-instruments');
    const result = [];
    for (const opt of sel.selectedOptions) {
        result.push(opt.value);
    }
    return result.length > 0 ? result : ['Piano'];
}

// ══════════════════════════════════════════════
// ── Harmonize ──
// ══════════════════════════════════════════════

async function generateHarmonization() {
    if (!composerSessionId) { composerShowToast('Create or load a session first'); return; }

    const melody = document.getElementById('harmonize-melody').value.trim();
    if (!melody) { composerShowToast('Enter a melody to harmonize'); return; }

    const style = document.getElementById('harmonize-style').value;
    const content = document.getElementById('harmonize-content');
    content.innerHTML = '';

    const label = document.createElement('div');
    label.className = 'composer-streaming-label';
    label.innerHTML = '<span class="pulse"></span> Harmonizing...';
    content.appendChild(label);
    const liveBox = document.createElement('div');
    liveBox.className = 'composer-streaming';
    content.appendChild(liveBox);

    let reasoningLines = [];

    try {
        const resp = await fetch('/composer/harmonize', {
            method: 'POST',
            headers: composerJsonHeaders(),
            body: JSON.stringify({
                session_id: composerSessionId,
                melody: melody,
                style: style,
            }),
        });

        if (!resp.ok) {
            content.innerHTML = '<div class="message error">Error: ' + resp.statusText + '</div>';
            return;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            let lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    var currentEvent = line.slice(7).trim();
                } else if (line.startsWith('data: ') && currentEvent) {
                    const data = JSON.parse(line.slice(6));
                    if (currentEvent === 'reasoning') {
                        reasoningLines.push(data);
                        liveBox.textContent = reasoningLines.join('\n');
                        liveBox.scrollTop = liveBox.scrollHeight;
                    } else if (currentEvent === 'done') {
                        content.innerHTML = '';
                        renderScoreResponse(data.response, data.reasoning, content, data.has_score);
                    } else if (currentEvent === 'error') {
                        content.innerHTML = '<div class="message error">Error: ' + data + '</div>';
                    }
                    currentEvent = null;
                }
            }
        }
    } catch (e) {
        content.innerHTML = '<div class="message error">Connection error: ' + e.message + '</div>';
    }
}

// ══════════════════════════════════════════════
// ── Analyze ──
// ══════════════════════════════════════════════

async function generateAnalysis() {
    if (!composerSessionId) { composerShowToast('Create or load a session first'); return; }

    const input = document.getElementById('analyze-input').value.trim();
    if (!input) { composerShowToast('Enter musical content to analyze'); return; }

    const content = document.getElementById('analyze-content');
    content.innerHTML = '';

    const label = document.createElement('div');
    label.className = 'composer-streaming-label';
    label.innerHTML = '<span class="pulse"></span> Analyzing...';
    content.appendChild(label);
    const liveBox = document.createElement('div');
    liveBox.className = 'composer-streaming';
    content.appendChild(liveBox);

    let reasoningLines = [];

    try {
        const resp = await fetch('/composer/analyze', {
            method: 'POST',
            headers: composerJsonHeaders(),
            body: JSON.stringify({
                session_id: composerSessionId,
                content: input,
            }),
        });

        if (!resp.ok) {
            content.innerHTML = '<div class="message error">Error: ' + resp.statusText + '</div>';
            return;
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            let lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    var currentEvent = line.slice(7).trim();
                } else if (line.startsWith('data: ') && currentEvent) {
                    const data = JSON.parse(line.slice(6));
                    if (currentEvent === 'reasoning') {
                        reasoningLines.push(data);
                        liveBox.textContent = reasoningLines.join('\n');
                        liveBox.scrollTop = liveBox.scrollHeight;
                    } else if (currentEvent === 'done') {
                        content.innerHTML = '';
                        renderScoreResponse(data.response, data.reasoning, content);
                    } else if (currentEvent === 'error') {
                        content.innerHTML = '<div class="message error">Error: ' + data + '</div>';
                    }
                    currentEvent = null;
                }
            }
        }
    } catch (e) {
        content.innerHTML = '<div class="message error">Connection error: ' + e.message + '</div>';
    }
}

// ══════════════════════════════════════════════
// ── Compositions list ──
// ══════════════════════════════════════════════

async function loadCompositions() {
    if (!composerSessionId) return;
    const list = document.getElementById('compositions-list');
    try {
        const resp = await fetch('/composer/sessions/' + composerSessionId + '/compositions',
            { headers: composerAuthHeaders() });
        const data = await resp.json();
        if (!data.compositions || data.compositions.length === 0) {
            list.innerHTML = '<div style="color:#64748b;font-size:0.82rem;">No saved compositions yet</div>';
            return;
        }
        list.innerHTML = '';
        data.compositions.forEach(c => {
            const item = document.createElement('div');
            item.className = 'composition-item';
            item.innerHTML = `
                <div class="comp-title">${escapeHtml(c.title)}</div>
                <div class="comp-meta">${c.genre} · ${c.key_signature} · ${c.time_signature} · ${c.tempo} BPM · ${new Date(c.created_at).toLocaleString()}</div>
            `;
            item.onclick = () => loadComposition(c.id);
            list.appendChild(item);
        });
    } catch (e) {
        list.innerHTML = '<div style="color:#e94560;font-size:0.82rem;">Error loading compositions</div>';
    }
}

async function loadComposition(id) {
    try {
        const resp = await fetch('/composer/compositions/' + id, { headers: composerAuthHeaders() });
        if (!resp.ok) { composerShowToast('Failed to load composition'); return; }
        const data = await resp.json();
        const content = document.getElementById('composition-detail');
        content.innerHTML = '';

        const title = document.createElement('h3');
        title.textContent = data.title;
        title.style.color = '#e0e0e0';
        content.appendChild(title);

        if (data.musicxml) {
            const pre = document.createElement('div');
            pre.className = 'musicxml-block';
            pre.textContent = data.musicxml.substring(0, 2000) + (data.musicxml.length > 2000 ? '\n... (truncated)' : '');
            content.appendChild(pre);

            const acts = document.createElement('div');
            acts.className = 'score-actions';
            const dlBtn = document.createElement('button');
            dlBtn.textContent = '📥 Download MusicXML';
            dlBtn.onclick = () => {
                const blob = new Blob([data.musicxml], { type: 'application/vnd.recordare.musicxml+xml' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = (data.title || 'composition').replace(/[^\w\s-]/g, '') + '.musicxml';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            };
            acts.appendChild(dlBtn);

            const delBtn = document.createElement('button');
            delBtn.textContent = '🗑️ Delete';
            delBtn.style.color = '#e94560';
            delBtn.onclick = async () => {
                if (!confirm('Delete "' + data.title + '"?')) return;
                const r = await fetch('/composer/compositions/' + id, {
                    method: 'DELETE', headers: composerAuthHeaders(),
                });
                if (r.ok) {
                    composerShowToast('Deleted');
                    content.innerHTML = '';
                    loadCompositions();
                }
            };
            acts.appendChild(delBtn);
            content.appendChild(acts);
        } else {
            content.innerHTML += '<div style="color:#64748b;">No MusicXML data stored</div>';
        }
    } catch (e) {
        composerShowToast('Error: ' + e.message);
    }
}

function escapeHtml(str) {
    const d = document.createElement('div');
    d.appendChild(document.createTextNode(str));
    return d.innerHTML;
}

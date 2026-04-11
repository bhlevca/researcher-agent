// ══════════════════════════════════════════════
// ── Composer chat (conversational composition) ──
// ══════════════════════════════════════════════

function addComposerMsg(role, text, reasoning, hasScore) {
    const w = document.createElement('div');
    w.className = 'msg-wrapper ' + role;
    const m = document.createElement('div');
    m.className = 'message ' + role;
    m.innerHTML = renderContent(text);
    w.appendChild(m);

    // Action buttons for assistant messages
    if (role === 'assistant') {
        const acts = document.createElement('div');
        acts.className = 'msg-actions';
        acts.style.opacity = '1';

        const copyBtn = document.createElement('button');
        copyBtn.textContent = '📋 Copy';
        copyBtn.onclick = () => navigator.clipboard.writeText(m.innerText || text).then(() => composerShowToast('Copied'));
        acts.appendChild(copyBtn);

        const saveBtn = document.createElement('button');
        saveBtn.textContent = '💾 Save';
        saveBtn.onclick = () => composerSaveText(m.innerText || text, null);
        acts.appendChild(saveBtn);

        if (reasoning && reasoning.length > 0) {
            const copyAll = document.createElement('button');
            copyAll.textContent = '📋 Copy all';
            copyAll.onclick = () => {
                const full = '=== Response ===\n' + (m.innerText || text) + '\n\n=== Reasoning ===\n' + reasoning.join('\n');
                navigator.clipboard.writeText(full).then(() => composerShowToast('Copied response + reasoning'));
            };
            acts.appendChild(copyAll);

            const saveAll = document.createElement('button');
            saveAll.textContent = '💾 Save all';
            saveAll.onclick = () => composerSaveText(m.innerText || text, reasoning);
            acts.appendChild(saveAll);
        }

        // If response contains a score (server-detected or client-detected)
        if (hasScore || _hasScore(text)) {
            const dlBtn = document.createElement('button');
            dlBtn.textContent = '📥 Download MusicXML';
            dlBtn.onclick = () => downloadMusicXMLFromText(text);
            acts.appendChild(dlBtn);

            const scoreBtn = document.createElement('button');
            scoreBtn.textContent = '🎼 Save Score';
            scoreBtn.onclick = () => saveScoreFromResponse(text);
            acts.appendChild(scoreBtn);
        }
        // Reasoning collapsible (before action buttons)
        if (reasoning && reasoning.length > 0) {
            const details = document.createElement('details');
            details.className = 'reasoning-toggle';
            const summary = document.createElement('summary');
            summary.textContent = 'Reasoning (' + reasoning.length + ' steps)';
            details.appendChild(summary);
            const content = document.createElement('div');
            content.className = 'reasoning-content';
            content.textContent = reasoning.join('\n');
            details.appendChild(content);
            w.appendChild(details);
        }

        w.appendChild(acts);
    }

    composerChat.appendChild(w);
    composerChat.scrollTop = composerChat.scrollHeight;
}

/** Check whether the response contains a JSON score or legacy XML score. */
function _hasScore(text) {
    if (text.includes('<score-partwise') || text.includes('```xml')) return true;
    // JSON score: look for ```json with "parts" inside
    const jm = text.match(/```json\s*([\s\S]*?)```/);
    if (jm && jm[1].includes('"parts"')) return true;
    // Bare JSON with parts array
    if (text.includes('"parts"') && text.includes('"measures"')) return true;
    return false;
}

async function downloadMusicXMLFromText(text) {
    // Try legacy client-side XML extraction first
    const xmlMatch = text.match(/```xml\s*([\s\S]*?)```/);
    if (xmlMatch && xmlMatch[1].includes('<score-partwise')) {
        const xml = xmlMatch[1].trim();
        _downloadBlob(xml, 'composition.musicxml', 'application/vnd.recordare.musicxml+xml');
        composerShowToast('MusicXML downloaded');
        return;
    }
    // JSON score → send to server for conversion
    try {
        const resp = await fetch('/composer/convert-musicxml', {
            method: 'POST',
            headers: composerJsonHeaders(),
            body: JSON.stringify({
                session_id: composerSessionId || '',
                response: text,
            }),
        });
        if (!resp.ok) {
            composerShowToast('Conversion failed: ' + resp.statusText);
            return;
        }
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'composition.musicxml';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        composerShowToast('MusicXML downloaded');
    } catch (e) {
        composerShowToast('Error: ' + e.message);
    }
}

function _downloadBlob(content, filename, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function composerSaveText(text, reasoning) {
    var content = text;
    if (reasoning && reasoning.length > 0) {
        content += '\n\n--- Reasoning ---\n' + reasoning.join('\n');
    }
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'composer-response.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    composerShowToast('Saved to file');
}

async function saveScoreFromResponse(responseText) {
    if (!composerSessionId) { composerShowToast('Create a session first'); return; }
    const title = prompt('Composition title:', 'Untitled');
    if (!title) return;
    try {
        const resp = await fetch('/composer/compositions/save', {
            method: 'POST',
            headers: composerJsonHeaders(),
            body: JSON.stringify({
                session_id: composerSessionId,
                title: title,
                description: '',
                response: responseText,
                instruments: [],
            }),
        });
        if (resp.ok) {
            const data = await resp.json();
            composerShowToast('Saved: ' + title + (data.has_musicxml ? ' (with MusicXML)' : ' (no MusicXML found)'));
        } else {
            composerShowToast('Failed to save composition');
        }
    } catch (e) { composerShowToast('Error: ' + e.message); }
}

async function sendComposerMessage() {
    const text = composerInput.value.trim();
    if (!text) return;
    if (!composerSessionId) { composerShowToast('Create or load a session first'); return; }

    addComposerMsg('user', text);
    composerChatHistory.push({ role: 'user', text });
    composerInput.value = '';
    composerSendBtn.disabled = true;

    // Show streaming indicator
    const liveWrapper = document.createElement('div');
    liveWrapper.className = 'msg-wrapper assistant';
    const label = document.createElement('div');
    label.className = 'composer-streaming-label';
    label.innerHTML = '<span class="pulse"></span> Composing...';
    liveWrapper.appendChild(label);
    const liveBox = document.createElement('div');
    liveBox.className = 'composer-streaming';
    liveWrapper.appendChild(liveBox);
    composerChat.appendChild(liveWrapper);
    composerChat.scrollTop = composerChat.scrollHeight;

    let reasoningLines = [];

    try {
        const resp = await fetch('/composer/chat', {
            method: 'POST',
            headers: composerJsonHeaders(),
            body: JSON.stringify({
                message: text,
                session_id: composerSessionId,
                mode: 'compose',
            }),
        });

        if (!resp.ok) {
            liveWrapper.remove();
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            addComposerMsg('error', 'Error: ' + (err.detail || resp.statusText));
            composerSendBtn.disabled = false;
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
                        liveWrapper.remove();
                        const response = data.response || '';
                        composerChatHistory.push({ role: 'assistant', text: response, reasoning: data.reasoning });
                        addComposerMsg('assistant', response, data.reasoning, data.has_score);
                    } else if (currentEvent === 'error') {
                        liveWrapper.remove();
                        addComposerMsg('error', 'Error: ' + data);
                    } else if (currentEvent === 'cancelled') {
                        liveWrapper.remove();
                        addComposerMsg('error', 'Cancelled');
                    }
                    currentEvent = null;
                }
            }
        }
    } catch (e) {
        liveWrapper.remove();
        addComposerMsg('error', 'Connection error: ' + e.message);
    }
    composerSendBtn.disabled = false;
}

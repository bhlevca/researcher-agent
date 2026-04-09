// ══════════════════════════════════════════════
// ── Tutor chat (conversation mode) ──
// ══════════════════════════════════════════════

function addTutorMsg(role, text) {
    const w = document.createElement('div');
    w.className = 'msg-wrapper ' + role;
    const m = document.createElement('div');
    m.className = 'message ' + role;
    m.innerHTML = renderContent(text);
    w.appendChild(m);

    // Speak button for assistant messages
    if (role === 'assistant') {
        const acts = document.createElement('div');
        acts.className = 'msg-actions';
        acts.style.opacity = '1';
        const speakBtn = document.createElement('button');
        speakBtn.textContent = '🔊 Speak';
        speakBtn.onclick = () => {
            if (typeof speakText === 'function') speakText(text, w);
            else tutorShowToast('TTS not available');
        };
        acts.appendChild(speakBtn);
        const copyBtn = document.createElement('button');
        copyBtn.textContent = '📋 Copy';
        copyBtn.onclick = () => navigator.clipboard.writeText(text).then(() => tutorShowToast('Copied'));
        acts.appendChild(copyBtn);
        w.appendChild(acts);
    }

    tutorChat.appendChild(w);
    tutorChat.scrollTop = tutorChat.scrollHeight;
}

async function sendTutorMessage() {
    const text = tutorInput.value.trim();
    if (!text) return;
    if (!tutorSessionId) { tutorShowToast('Create or load a session first'); return; }

    addTutorMsg('user', text);
    tutorChatHistory.push({ role: 'user', text });
    tutorInput.value = '';
    tutorSendBtn.disabled = true;

    // Show streaming indicator
    const liveWrapper = document.createElement('div');
    liveWrapper.className = 'msg-wrapper assistant';
    const label = document.createElement('div');
    label.className = 'tutor-streaming-label';
    label.innerHTML = '<span class="pulse"></span> Thinking...';
    liveWrapper.appendChild(label);
    const liveBox = document.createElement('div');
    liveBox.className = 'tutor-streaming';
    liveWrapper.appendChild(liveBox);
    tutorChat.appendChild(liveWrapper);
    tutorChat.scrollTop = tutorChat.scrollHeight;

    const cfg = getTutorConfig();
    let reasoningLines = [];

    try {
        const resp = await fetch('/tutor/chat', {
            method: 'POST',
            headers: tutorJsonHeaders(),
            body: JSON.stringify({
                message: text,
                session_id: tutorSessionId,
                mode: 'conversation',
                target_language: cfg.target_lang,
                native_language: cfg.native_lang,
                level: cfg.level,
                history: tutorChatHistory,
            }),
        });

        if (!resp.ok) {
            liveWrapper.remove();
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            addTutorMsg('error', 'Error: ' + (err.detail || resp.statusText));
            return;
        }

        const result = await processTutorSSE(resp, liveBox, reasoningLines);
        liveWrapper.remove();

        if (result) {
            addTutorMsg('assistant', result.response);
            tutorChatHistory.push({ role: 'assistant', text: result.response });
            // Auto-save after each exchange
            saveTutorSession();
            // Dialog mode: auto-speak response then auto-record
            if (dialogMode && typeof speakText === 'function') {
                const lastWrapper = tutorChat.querySelector('.msg-wrapper.assistant:last-child');
                await speakText(result.response, lastWrapper);
                if (dialogMode && typeof toggleMic === 'function') toggleMic();
            }
        }
    } catch (e) {
        liveWrapper.remove();
        addTutorMsg('error', 'Connection error: ' + e.message);
    } finally {
        tutorSendBtn.disabled = false;
        tutorInput.focus();
    }
}

// ── SSE stream processor (shared by chat, lessons, quiz, appraisal) ──
async function processTutorSSE(resp, liveBox, reasoningLines) {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let result = null;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        while (buffer.includes('\n\n')) {
            const idx = buffer.indexOf('\n\n');
            const eventStr = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 2);

            let eventType = '';
            let dataStr = '';
            for (const line of eventStr.split('\n')) {
                if (line.startsWith('event: ')) eventType = line.slice(7);
                else if (line.startsWith('data: ')) dataStr += line.slice(6);
            }

            if (eventType === 'reasoning' && dataStr) {
                const line = JSON.parse(dataStr);
                reasoningLines.push(line);
                if (liveBox) {
                    liveBox.textContent = reasoningLines.join('\n');
                    liveBox.scrollTop = liveBox.scrollHeight;
                }
            } else if (eventType === 'done' && dataStr) {
                result = JSON.parse(dataStr);
            } else if (eventType === 'error' && dataStr) {
                tutorShowToast('Error: ' + JSON.parse(dataStr));
            }
        }
    }
    return result;
}

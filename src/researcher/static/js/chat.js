// ── Cancel in-flight request ──
let _activeReader = null;       // set by sendMessage, cleared on finish
let _activeLiveWrapper = null;  // the live-reasoning DOM node

async function cancelRequest() {
    // 1. Tell backend to stop streaming
    try { await fetch('/chat/cancel', { method: 'POST', headers: authHeaders() }); } catch (_) {}
    // 2. Abort the reader so the fetch loop exits
    if (_activeReader) {
        try { _activeReader.cancel(); } catch (_) {}
        _activeReader = null;
    }
    // 3. Clean up UI
    if (_activeLiveWrapper && _activeLiveWrapper.parentNode) {
        _activeLiveWrapper.remove();
    }
    _activeLiveWrapper = null;
    cancelBtn.classList.remove('visible');
    sendBtn.disabled = false;
    input.disabled = false;
    input.focus();
    addSimpleMsg('error', 'Request cancelled.');
}

// ── Send message ──
async function sendMessage() {
    const text = input.value.trim();
    if (!text) return;

    addSimpleMsg('user', text);
    chatHistory.push({ role: 'user', text: text });
    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;
    cancelBtn.classList.add('visible');

    // Create live reasoning container
    const liveWrapper = document.createElement('div');
    liveWrapper.className = 'msg-wrapper assistant';
    _activeLiveWrapper = liveWrapper;

    const liveLabel = document.createElement('div');
    liveLabel.className = 'live-reasoning-label';
    liveLabel.innerHTML = '<span class="pulse"></span> Thinking...';
    liveWrapper.appendChild(liveLabel);

    const liveBox = document.createElement('div');
    liveBox.className = 'live-reasoning';
    liveWrapper.appendChild(liveBox);

    chat.appendChild(liveWrapper);
    chat.scrollTop = chat.scrollHeight;

    let reasoningLines = [];

    try {
        const resp = await fetch('/chat', {
            method: 'POST',
            headers: jsonAuthHeaders(),
            body: JSON.stringify({ message: text, history: chatHistory, file_ids: getAttachedFileIds() }),
        });

        if (!resp.ok) {
            liveWrapper.remove();
            const err = await resp.json().catch(() => ({ detail: resp.statusText }));
            addSimpleMsg('error', 'Error: ' + (err.detail || resp.statusText));
            return;
        }

        const reader = resp.body.getReader();
        _activeReader = reader;
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // Process complete SSE events (separated by \n\n)
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
                    liveBox.textContent = reasoningLines.join('\n');
                    liveBox.scrollTop = liveBox.scrollHeight;
                    chat.scrollTop = chat.scrollHeight;
                } else if (eventType === 'done' && dataStr) {
                    const result = JSON.parse(dataStr);
                    liveWrapper.remove();
                    const msgWrapper = addAssistantMessage(
                        result.response,
                        result.reasoning || reasoningLines,
                        result.token_usage || {}
                    );
                    chatHistory.push({
                        role: 'assistant',
                        text: result.response,
                        reasoning: result.reasoning || reasoningLines,
                        tokenUsage: result.token_usage || {}
                    });
                    // ── Incomplete: offer to continue ──
                    if (result.incomplete) {
                        const contBtn = document.createElement('button');
                        contBtn.textContent = '🔄 Continue — the agent ran out of steps';
                        contBtn.style.cssText = 'margin-top:0.5rem;background:#e94560;color:white;border:none;padding:0.45rem 1rem;border-radius:0.5rem;cursor:pointer;font-size:0.82rem;';
                        contBtn.onclick = () => {
                            contBtn.remove();
                            continueLastRequest();
                        };
                        msgWrapper.appendChild(contBtn);
                    }
                    // ── Dialog: auto-speak then auto-record ──
                    if (dialogMode && result.response) {
                        speakText(result.response, msgWrapper).then(() => {
                            // Auto-start mic for next turn if last input was voice
                            if (dialogMode && lastInputWasVoice) {
                                setTimeout(() => toggleMic(), 400);
                            }
                        });
                    }
                } else if (eventType === 'cancelled' && dataStr) {
                    liveWrapper.remove();
                    addSimpleMsg('error', 'Request cancelled.');
                    return;
                } else if (eventType === 'error' && dataStr) {
                    // Preserve reasoning lines and show error below them
                    liveWrapper.remove();
                    const errText = 'Error: ' + JSON.parse(dataStr);
                    if (reasoningLines.length > 0) {
                        // Show reasoning + error together
                        const w = document.createElement('div');
                        w.className = 'msg-wrapper assistant';

                        const errMsg = document.createElement('div');
                        errMsg.className = 'message error';
                        errMsg.textContent = errText;
                        w.appendChild(errMsg);

                        const det = document.createElement('details');
                        det.className = 'reasoning-toggle';
                        det.setAttribute('open', '');
                        const sum = document.createElement('summary');
                        sum.textContent = 'Show reasoning (' + reasoningLines.length + ' steps before error)';
                        det.appendChild(sum);
                        const c = document.createElement('div');
                        c.className = 'reasoning-content';
                        c.textContent = reasoningLines.join('\n');
                        det.appendChild(c);
                        w.appendChild(det);

                        chat.appendChild(w);
                        chat.scrollTop = chat.scrollHeight;
                    } else {
                        addSimpleMsg('error', errText);
                    }
                }
            }
        }

        // If stream ended without a 'done' event, show what we have
        if (liveWrapper.parentNode) {
            liveWrapper.remove();
            if (reasoningLines.length > 0) {
                addAssistantMessage(
                    'The agent completed but produced no final answer.',
                    reasoningLines, {}
                );
            }
        }
    } catch (e) {
        if (liveWrapper.parentNode) liveWrapper.remove();
        // Don't show connection error if user cancelled
        if (!cancelBtn.classList.contains('visible') || e.message !== 'The reader was released.') {
            addSimpleMsg('error', 'Connection error: ' + e.message);
        }
    } finally {
        _activeReader = null;
        _activeLiveWrapper = null;
        cancelBtn.classList.remove('visible');
        sendBtn.disabled = false;
        input.focus();
    }
}

// ── Continue incomplete request ──
async function continueLastRequest() {
    try {
    showToast('Continue: starting…');
    // Find the last user message and last assistant response
    let lastUserMsg = '';
    for (let i = chatHistory.length - 1; i >= 0; i--) {
        if (chatHistory[i].role === 'user') {
            lastUserMsg = chatHistory[i].text;
            break;
        }
    }
    const lastAssistant = chatHistory.length > 0 ? chatHistory[chatHistory.length - 1].text : '';

    // Find the last assistant message wrapper to append to
    const allMsgWrappers = document.querySelectorAll('.msg-wrapper.assistant');
    const lastMsgWrapper = allMsgWrappers[allMsgWrappers.length - 1];
    const lastMsgDiv = lastMsgWrapper ? lastMsgWrapper.querySelector('.message.assistant') : null;

    // Show a "continuing…" indicator in the chat
    const spinner = document.createElement('div');
    spinner.className = 'msg-wrapper assistant';
    spinner.innerHTML = '<div class="message assistant" style="background:#2a1a3e;border:2px solid #e94560;padding:1rem;">⏳ <strong>Continuing from where the agent left off…</strong><br>This may take up to a minute while the LLM generates the completion.</div>';
    chat.appendChild(spinner);
    chat.scrollTop = chat.scrollHeight;

    showToast('Calling LLM to continue…');
    const resp = await fetch('/chat/continue', {
        method: 'POST',
        headers: jsonAuthHeaders(),
        body: JSON.stringify({
            original_query: lastUserMsg,
            partial_response: lastAssistant,
            file_ids: [],
        }),
    });
    if (!resp.ok) {
        spinner.remove();
        const errText = await resp.text();
        showToast('Continue failed (' + resp.status + '): ' + errText.slice(0, 100));
        return;
    }

    const result = await resp.json();
    spinner.remove();

    if (result.response && lastMsgDiv) {
        // Check if the original response was a failure/empty message
        const failurePatterns = [
            'could not complete the request',
            'Please try rephrasing',
            'Maximum iterations reached',
            'Invalid response from LLM call',
        ];
        const origIsFailure = failurePatterns.some(p => lastAssistant.includes(p))
            || lastAssistant.trim().length < 100;
        // If original had real content, combine; otherwise replace entirely
        const finalText = origIsFailure
            ? result.response
            : lastAssistant + '\n\n' + result.response;
        lastMsgDiv.innerHTML = renderContent(finalText);
        for (let i = chatHistory.length - 1; i >= 0; i--) {
            if (chatHistory[i].role === 'assistant') {
                chatHistory[i].text = finalText;
                break;
            }
        }
        showToast('✅ Response continued successfully');
    } else if (result.response) {
        addAssistantMessage(result.response, result.reasoning || [], result.token_usage || {});
        chatHistory.push({ role: 'assistant', text: result.response });
        showToast('✅ Continuation added as new message');
    } else {
        showToast('⚠️ LLM returned empty response');
    }
    chat.scrollTop = chat.scrollHeight;
    } catch (e) {
        showToast('❌ Continue error: ' + e.message);
    }
}

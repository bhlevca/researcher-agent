// ── Message rendering ──
function addSimpleMsg(role, text) {
    const w = document.createElement('div');
    w.className = 'msg-wrapper ' + role;
    const m = document.createElement('div');
    m.className = 'message ' + role;
    m.textContent = text;
    w.appendChild(m);
    chat.appendChild(w);
    chat.scrollTop = chat.scrollHeight;
}

function addAssistantMessage(text, reasoning, tokenUsage) {
    const w = document.createElement('div');
    w.className = 'msg-wrapper assistant';

    const m = document.createElement('div');
    m.className = 'message assistant';
    m.innerHTML = renderContent(text);
    w.appendChild(m);

    // Collapsible reasoning
    if (reasoning && reasoning.length > 0) {
        const det = document.createElement('details');
        det.className = 'reasoning-toggle';
        const sum = document.createElement('summary');
        sum.textContent = 'Show reasoning (' + reasoning.length + ' steps)';
        det.appendChild(sum);
        const c = document.createElement('div');
        c.className = 'reasoning-content';
        c.textContent = reasoning.join('\n');
        det.appendChild(c);
        w.appendChild(det);
    }

    // Token usage
    if (tokenUsage && tokenUsage.total_tokens) {
        const b = document.createElement('div');
        b.className = 'token-badge';
        b.textContent = 'tokens: ' + tokenUsage.total_tokens +
            ' (' + tokenUsage.prompt_tokens + ' in / ' + tokenUsage.completion_tokens + ' out)';
        w.appendChild(b);
    }

    // Action buttons
    const acts = document.createElement('div');
    acts.className = 'msg-actions';

    const copyBtn = document.createElement('button');
    copyBtn.textContent = '\ud83d\udccb Copy';
    copyBtn.onclick = () => {
        const liveText = m.innerText || text;
        navigator.clipboard.writeText(liveText).then(() => showToast('Copied response'));
    };
    acts.appendChild(copyBtn);

    const saveBtn = document.createElement('button');
    saveBtn.textContent = '\ud83d\udcbe Save';
    saveBtn.onclick = () => saveText(m.innerText || text, null);
    acts.appendChild(saveBtn);

    if (reasoning && reasoning.length > 0) {
        const copyAll = document.createElement('button');
        copyAll.textContent = '\ud83d\udccb Copy all';
        copyAll.onclick = () => {
            const liveText = m.innerText || text;
            const full = '=== Response ===\n' + liveText + '\n\n=== Reasoning ===\n' + reasoning.join('\n');
            navigator.clipboard.writeText(full).then(() => showToast('Copied response + reasoning'));
        };
        acts.appendChild(copyAll);

        const saveAll = document.createElement('button');
        saveAll.textContent = '\ud83d\udcbe Save all';
        saveAll.onclick = () => saveText(m.innerText || text, reasoning);
        acts.appendChild(saveAll);
    }

    // Export dropdown
    const exportDiv = document.createElement('div');
    exportDiv.className = 'export-dropdown';
    const exportTrig = document.createElement('button');
    exportTrig.className = 'export-trigger';
    exportTrig.textContent = '\u2b07\ufe0f Export';
    exportTrig.onclick = (e) => {
        e.stopPropagation();
        // Close any other open export menus
        document.querySelectorAll('.export-menu.open').forEach(m => {
            if (m !== exportMenu) m.classList.remove('open');
        });
        exportMenu.classList.toggle('open');
    };
    exportDiv.appendChild(exportTrig);
    const exportMenu = document.createElement('div');
    exportMenu.className = 'export-menu';
    ['md', 'txt', 'docx', 'xlsx'].forEach(fmt => {
        const btn = document.createElement('button');
        btn.textContent = fmt.toUpperCase();
        btn.onclick = () => {
            exportMenu.classList.remove('open');
            // Use live text from DOM so continuation content is included
            const liveText = m.innerText || text;
            exportResponse(liveText, fmt);
        };
        exportMenu.appendChild(btn);
    });
    exportDiv.appendChild(exportMenu);
    acts.appendChild(exportDiv);

    w.appendChild(acts);
    chat.appendChild(w);
    chat.scrollTop = chat.scrollHeight;
    return w;  // return wrapper so TTS can attach speaking indicator
}

function saveText(text, reasoning) {
    var content = text;
    if (reasoning && reasoning.length > 0) {
        content += '\n\n--- Reasoning ---\n' + reasoning.join('\n');
    }
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'response-' + new Date().toISOString().slice(0,19).replace(/:/g,'-') + '.txt';
    a.click();
    URL.revokeObjectURL(url);
    showToast('Saved to file');
}

function showThinking() {
    const d = document.createElement('div');
    d.className = 'thinking'; d.id = 'thinking-indicator';
    d.innerHTML = 'Thinking <span class="dots"><span>.</span><span>.</span><span>.</span></span>';
    chat.appendChild(d);
    chat.scrollTop = chat.scrollHeight;
}

function hideThinking() {
    const el = document.getElementById('thinking-indicator');
    if (el) el.remove();
}

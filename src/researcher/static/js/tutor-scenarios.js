// ══════════════════════════════════════════════
// ── Situational Dialog Scenarios ──
// ══════════════════════════════════════════════

let _activeScenario = null;
let _scenarioMessages = [];

async function loadScenarios() {
    if (!tutorSessionId) return;
    const level = (document.getElementById('scenario-level-filter') || {}).value || '';
    const cards = document.getElementById('scenario-cards');
    const active = document.getElementById('scenario-active');
    if (!cards) return;
    cards.innerHTML = '<div style="color:var(--muted);font-size:0.85rem;">Loading…</div>';
    active.style.display = 'none';

    try {
        const url = '/tutor/scenarios' + (level ? '?level=' + level : '');
        const resp = await fetch(url, { headers: tutorAuthHeaders() });
        if (!resp.ok) { cards.innerHTML = ''; return; }
        const data = await resp.json();
        cards.innerHTML = '';
        (data.scenarios || []).forEach(s => {
            const card = document.createElement('div');
            card.style.cssText = 'background:var(--surface);border:1px solid #1a3a6e;border-radius:0.6rem;padding:0.75rem;cursor:pointer;transition:border-color 0.2s;';
            card.onmouseover = () => { card.style.borderColor = '#e94560'; };
            card.onmouseout = () => { card.style.borderColor = '#1a3a6e'; };
            card.innerHTML =
                '<div style="font-weight:600;font-size:0.9rem;margin-bottom:0.3rem;">' + escapeHtml(s.title) + '</div>' +
                '<div style="font-size:0.75rem;color:#e94560;margin-bottom:0.4rem;">' + escapeHtml(s.level) + ' · ' + escapeHtml(s.role_student || '') + '</div>' +
                '<div style="font-size:0.8rem;color:var(--muted);">' + escapeHtml(s.description || '') + '</div>';
            card.onclick = () => startScenario(s);
            cards.appendChild(card);
        });
        if (!data.scenarios || !data.scenarios.length) {
            cards.innerHTML = '<div style="color:var(--muted);font-size:0.85rem;">No scenarios for this level.</div>';
        }
    } catch (e) { cards.innerHTML = ''; }
}

function startScenario(scenario) {
    _activeScenario = scenario;
    _scenarioMessages = [];
    document.getElementById('scenario-cards').style.display = 'none';
    const active = document.getElementById('scenario-active');
    active.style.display = 'block';
    const label = document.getElementById('scenario-title-label');
    label.textContent = scenario.title + ' — you are: ' + (scenario.role_student || 'Student');
    const chat = document.getElementById('scenario-chat');
    chat.innerHTML = '';
    // Show the opening line from the tutor's character
    if (scenario.starter) {
        const msg = document.createElement('div');
        msg.className = 'chat-message assistant';
        msg.innerHTML = '<strong>' + escapeHtml(scenario.role_tutor) + ':</strong> ' + escapeHtml(scenario.starter);
        chat.appendChild(msg);
        chat.scrollTop = chat.scrollHeight;
    }
}

function exitScenario() {
    _activeScenario = null;
    _scenarioMessages = [];
    document.getElementById('scenario-active').style.display = 'none';
    document.getElementById('scenario-cards').style.display = 'grid';
}

async function sendScenarioMessage() {
    if (!_activeScenario || !tutorSessionId) { tutorShowToast('Start a scenario first'); return; }
    const input = document.getElementById('scenario-input');
    const text = (input.value || '').trim();
    if (!text) return;
    input.value = '';

    const chat = document.getElementById('scenario-chat');
    const userMsg = document.createElement('div');
    userMsg.className = 'chat-message user';
    userMsg.innerHTML = '<strong>' + escapeHtml(_activeScenario.role_student) + ':</strong> ' + escapeHtml(text);
    chat.appendChild(userMsg);
    chat.scrollTop = chat.scrollHeight;

    _scenarioMessages.push({ role: 'user', content: text });

    const liveBox = document.createElement('div');
    liveBox.className = 'chat-message assistant';
    liveBox.innerHTML = '<span class="pulse"></span>';
    chat.appendChild(liveBox);
    chat.scrollTop = chat.scrollHeight;

    try {
        const resp = await fetch('/tutor/chat', {
            method: 'POST',
            headers: tutorJsonHeaders(),
            body: JSON.stringify({
                session_id: tutorSessionId,
                message: text,
                mode: 'scenario',
                scenario_id: _activeScenario.id,
            }),
        });
        if (!resp.ok) { liveBox.remove(); tutorShowToast('Failed'); return; }

        const result = await processTutorSSE(resp, liveBox, []);
        liveBox.remove();

        if (result && result.response) {
            const aMsg = document.createElement('div');
            aMsg.className = 'chat-message assistant';
            aMsg.innerHTML = '<strong>' + escapeHtml(_activeScenario.role_tutor) + ':</strong> ' + renderContent(result.response);
            chat.appendChild(aMsg);
            _scenarioMessages.push({ role: 'assistant', content: result.response });
            chat.scrollTop = chat.scrollHeight;
        }
    } catch (e) {
        liveBox.remove();
        tutorShowToast('Error: ' + e.message);
    }
}

// Enter key to send
document.addEventListener('DOMContentLoaded', () => {
    const inp = document.getElementById('scenario-input');
    if (inp) {
        inp.addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendScenarioMessage(); }
        });
    }
});

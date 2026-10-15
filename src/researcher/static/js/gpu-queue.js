/* The GPU panel: who has the card, who is waiting, and what has happened.
 *
 * One card, several people, five tabs. This is deliberately *always* visible
 * rather than appearing only when two requests collide: a panel that shows up
 * only under contention cannot be trusted, because when it is absent you
 * cannot tell whether the queue is empty or the queue is broken. It also
 * means a wait never arrives as a surprise — you can see the card is busy
 * before you press anything.
 *
 * The log keeps the session's history. Queue notices are moments, and a
 * moment that has passed is still the explanation for why your request took
 * ninety seconds, so nothing is erased until the tab closes. It survives
 * navigation between tabs via sessionStorage.
 *
 * Every streaming endpoint sends the same `queue` event, so this lives in one
 * file loaded by every page. The last time this logic was duplicated, two
 * copies of a function in different scripts silently overrode each other.
 */

const GPU_POLL_MS = 3000;
const GPU_LOG_KEY = 'gpuSessionLog';
const GPU_LOG_MAX = 200;

let _gpuPanel = null;
let _gpuTimer = null;
let _gpuMyTicket = 0;          // our place in line, when we are queued
let _gpuLastSeen = '';         // to log only actual changes

/* ── the session log ──────────────────────────────────────────────────── */

function gpuLogRead() {
    try {
        return JSON.parse(sessionStorage.getItem(GPU_LOG_KEY) || '[]');
    } catch (e) { return []; }
}

function gpuLogWrite(entries) {
    try {
        sessionStorage.setItem(GPU_LOG_KEY,
                               JSON.stringify(entries.slice(-GPU_LOG_MAX)));
    } catch (e) { /* private window, or full: the panel still works */ }
}

function gpuLog(text) {
    const entries = gpuLogRead();
    const now = new Date();
    const stamp = now.toTimeString().slice(0, 8);
    // Collapse an immediate repeat rather than printing the same line twice.
    if (entries.length && entries[entries.length - 1].text === text) return;
    entries.push({ at: stamp, text });
    gpuLogWrite(entries);
    gpuRenderLog();
}

/* ── the panel ────────────────────────────────────────────────────────── */

function gpuEnsurePanel() {
    if (_gpuPanel && _gpuPanel.isConnected) return _gpuPanel;
    const el = document.createElement('div');
    el.className = 'gpu-panel';
    el.innerHTML = `
      <div class="gpu-panel-head">
        <span class="gpu-panel-title">🖥 GPU</span>
        <span class="gpu-panel-now">checking…</span>
        <button class="gpu-panel-toggle" title="Show or hide details">▾</button>
      </div>
      <div class="gpu-panel-body">
        <div class="gpu-panel-queue"></div>
        <div class="gpu-panel-logwrap">
          <div class="gpu-panel-loghead">This session</div>
          <ul class="gpu-panel-log"></ul>
        </div>
      </div>`;
    document.body.appendChild(el);
    _gpuPanel = el;

    const body = el.querySelector('.gpu-panel-body');
    const toggle = el.querySelector('.gpu-panel-toggle');
    // Collapsed by default: the headline is the part you need at a glance.
    let open = false;
    try { open = localStorage.getItem('gpuPanelOpen') === '1'; } catch (e) { /* fine */ }
    const apply = () => {
        body.style.display = open ? 'block' : 'none';
        toggle.textContent = open ? '▴' : '▾';
    };
    apply();
    toggle.onclick = () => {
        open = !open;
        try { localStorage.setItem('gpuPanelOpen', open ? '1' : '0'); } catch (e) { /* fine */ }
        apply();
    };
    gpuRenderLog();
    return el;
}

function gpuRenderLog() {
    if (!_gpuPanel) return;
    const ul = _gpuPanel.querySelector('.gpu-panel-log');
    if (!ul) return;
    const entries = gpuLogRead();
    ul.innerHTML = '';
    if (!entries.length) {
        const li = document.createElement('li');
        li.className = 'gpu-log-empty';
        li.textContent = 'Nothing yet.';
        ul.appendChild(li);
        return;
    }
    for (const e of entries.slice().reverse()) {
        const li = document.createElement('li');
        const t = document.createElement('span');
        t.className = 'gpu-log-at';
        t.textContent = e.at;
        li.appendChild(t);
        li.appendChild(document.createTextNode(' ' + e.text));
        ul.appendChild(li);
    }
}

function gpuDescribe(e, running, you) {
    const who = (you && e.user === you) ? 'you' : (e.user || 'someone');
    const secs = Math.round(running ? e.running_seconds : e.waiting_seconds);
    return running
        ? `${who} — ${e.label}, ${secs}s in`
        : `${who} — ${e.label}, waiting ${secs}s`;
}

function gpuRenderPanel(board) {
    const el = gpuEnsurePanel();
    const now = el.querySelector('.gpu-panel-now');
    const queue = el.querySelector('.gpu-panel-queue');
    const waiting = (board && board.waiting) || [];
    const running = board && board.running;

    el.classList.toggle('is-busy', !!running);
    el.classList.toggle('is-queued', waiting.length > 0);

    if (!board) {
        now.textContent = 'unreachable';
    } else if (running) {
        now.textContent = gpuDescribe(running, true, board.you)
            + (waiting.length ? ` · ${waiting.length} waiting` : '');
    } else {
        now.textContent = 'idle';
    }

    queue.innerHTML = '';
    if (!board) return;

    const line = (e, isRunning) => {
        const row = document.createElement('div');
        row.className = 'gpu-panel-row' + (isRunning ? ' is-running' : '');
        if (board.you && e.user === board.you) row.className += ' is-you';
        const label = document.createElement('span');
        label.textContent = (isRunning ? '▶ ' : '· ')
            + gpuDescribe(e, isRunning, board.you);
        row.appendChild(label);
        if (e.note) {
            const why = document.createElement('em');
            why.textContent = ' “' + e.note + '”';
            row.appendChild(why);
        }
        // Our own waiting entry gets the one action that makes sense.
        if (!isRunning && e.ticket === _gpuMyTicket) {
            const ask = document.createElement('button');
            if (e.priority) {
                ask.textContent = 'Asked to go first';
                ask.disabled = true;
            } else {
                ask.textContent = 'Ask to go first';
                ask.onclick = () => gpuRequestPriority(e.ticket, ask);
            }
            row.appendChild(ask);
        }
        return row;
    };

    if (running) queue.appendChild(line(running, true));
    for (const e of waiting) queue.appendChild(line(e, false));
    if (!running && !waiting.length) {
        const idle = document.createElement('div');
        idle.className = 'gpu-panel-row gpu-panel-idle';
        idle.textContent = 'Nobody is using the card.';
        queue.appendChild(idle);
    }
}

/* Poll for the whole session, not only while something of ours is waiting. */
async function gpuPoll() {
    let board = null;
    try {
        const r = await fetch('/gpu/queue', { headers: gpuAuthHeaders() });
        if (r.ok) board = await r.json();
    } catch (e) { /* render as unreachable */ }
    gpuRenderPanel(board);

    // Log transitions rather than every poll.
    if (board) {
        const running = board.running;
        const key = running
            ? `run:${running.ticket}`
            : `idle:${(board.waiting || []).map(w => w.ticket).join(',')}`;
        if (key !== _gpuLastSeen) {
            _gpuLastSeen = key;
            if (running) {
                gpuLog('started — ' + gpuDescribe(running, true, board.you));
            }
            for (const w of board.waiting || []) {
                gpuLog('queued — ' + gpuDescribe(w, false, board.you)
                       + (w.note ? ` (“${w.note}”)` : ''));
            }
            if (!running && !(board.waiting || []).length) gpuLog('card idle');
        }
    }
    return board;
}

function gpuStartPolling() {
    gpuEnsurePanel();
    if (_gpuTimer) return;
    gpuPoll();
    _gpuTimer = setInterval(gpuPoll, GPU_POLL_MS);
}

/* ── the inline note, for the tab you are actually waiting in ─────────── */

/* Called from each tab's SSE handler on a `queue` event. The panel is the
   permanent view; this is the one that sits where you are looking. */
function gpuShowQueueNote(liveBox, info) {
    if (info && info.ticket) _gpuMyTicket = info.ticket;
    if (info) {
        gpuLog(info.position
               ? `you — ${info.text}`
               : 'you — your turn, starting');
    }
    gpuPoll();      // reflect it in the panel immediately

    if (!liveBox || !liveBox.parentElement) return;
    const host = liveBox.parentElement;
    let note = host.querySelector('.gpu-queue-note');

    if (!info || !info.position) {
        if (note) note.remove();
        return;
    }
    if (!note) {
        note = document.createElement('div');
        note.className = 'gpu-queue-note';
        host.insertBefore(note, liveBox);
    }
    gpuRenderQueueNote(note, info);
}

function gpuRenderQueueNote(note, info) {
    note.innerHTML = '';
    const text = document.createElement('span');
    text.textContent = info.text;
    note.appendChild(text);
    if (info.ticket) {
        const ask = document.createElement('button');
        ask.textContent = 'Ask to go first';
        ask.onclick = () => gpuRequestPriority(info.ticket, ask);
        note.appendChild(ask);
    }
}

/* ── asking to go first ───────────────────────────────────────────────── */

/* Self-declared on purpose. This is a household, not a datacentre: the note
   is visible to everyone waiting, and that is the check on it. */
async function gpuRequestPriority(ticket, btn) {
    const why = prompt('Why do you need to go first?\n'
                       + 'Everyone waiting can see this.');
    if (!why || !why.trim()) return;        // no reason, no jumping
    if (btn) { btn.disabled = true; btn.textContent = 'Asking…'; }
    try {
        const headers = Object.assign({ 'Content-Type': 'application/json' },
                                      gpuAuthHeaders());
        const r = await fetch('/gpu/queue/priority', {
            method: 'POST', headers,
            body: JSON.stringify({ ticket, note: why.trim() }),
        });
        if (!r.ok) {
            let detail = 'Could not move you up the queue';
            try { detail = (await r.json()).detail || detail; } catch (e) { /* keep */ }
            gpuToast(detail);
            if (btn) { btn.disabled = false; btn.textContent = 'Ask to go first'; }
            return;
        }
        gpuLog(`you asked to go first — “${why.trim()}”`);
        if (btn) btn.textContent = 'You are next';
        gpuPoll();
    } catch (e) {
        gpuToast('Error: ' + e.message);
        if (btn) { btn.disabled = false; btn.textContent = 'Ask to go first'; }
    }
}

/* ── odds and ends ────────────────────────────────────────────────────── */

/* Auth headers, however the host page happens to keep its token. */
function gpuAuthHeaders() {
    try {
        if (typeof jsonAuthHeaders === 'function') return jsonAuthHeaders();
        if (typeof authHeaders === 'function') return authHeaders();
    } catch (e) { /* fall through */ }
    let token = null;
    try { token = localStorage.getItem('token'); } catch (e) { /* fine */ }
    return token ? { 'Authorization': 'Bearer ' + token } : {};
}

function gpuToast(msg) {
    if (typeof tutorShowToast === 'function') return tutorShowToast(msg);
    if (typeof showToast === 'function') return showToast(msg);
    alert(msg);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', gpuStartPolling);
} else {
    gpuStartPolling();
}

// ══════════════════════════════════════════════
// ── Song audio (ACE-Step) ──
//
// The score stays the thing that is correct; this renders a performance of
// it. "Perform my score" sends the last generated score, which the server
// turns into audio through fluidsynth and hands to ACE-Step as the source for
// a cover — same structure, real production.
//
// The engine is a separate process holding the GPU, so this tab also shows
// its state and offers a way to hand the card back.
// ══════════════════════════════════════════════

let _songEngineTimer = null;

async function refreshSongStatus() {
    const el = document.getElementById('song-status');
    if (!el) return null;
    try {
        const r = await fetch('/composer/song/status', { headers: authHeaders() });
        if (!r.ok) { el.textContent = 'Could not reach the server.'; return null; }
        const s = await r.json();
        const bits = [];
        if (!s.installed) {
            el.className = 'song-status is-missing';
            el.textContent = s.hint || 'The song engine is not installed.';
            return s;
        }
        bits.push(s.running
            ? `engine running (idle ${Math.round(s.idle_seconds)}s) — holding the GPU`
            : 'engine stopped — starts on first render');
        bits.push(`model ${s.model}`);
        if (!s.fluidsynth) {
            bits.push('no fluidsynth: "Perform my score" will fall back to the description');
        }
        el.className = 'song-status' + (s.running ? ' is-running' : '');
        el.textContent = bits.join(' · ');
        return s;
    } catch (e) {
        el.textContent = 'Could not reach the server.';
        return null;
    }
}

async function stopSongEngine() {
    const btn = document.getElementById('song-stop-btn');
    if (btn) { btn.disabled = true; btn.textContent = 'Stopping…'; }
    try {
        const r = await fetch('/composer/song/stop', {
            method: 'POST', headers: jsonAuthHeaders(),
        });
        const d = await r.json().catch(() => ({}));
        composerShowToast(d.stopped ? 'Song engine stopped, GPU free'
                                   : 'The engine was not running');
    } catch (e) {
        composerShowToast('Error: ' + e.message);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '⏏ Free GPU'; }
        refreshSongStatus();
    }
}

async function generateSong() {
    if (!composerSessionId) {
        composerShowToast('Create or load a session first');
        return;
    }
    const prompt = document.getElementById('song-prompt').value.trim();
    const lyrics = document.getElementById('song-lyrics').value.trim();
    const useScore = document.getElementById('song-use-score').checked;
    const strength = parseFloat(document.getElementById('song-strength').value);
    const duration = parseFloat(document.getElementById('song-duration').value) || 0;
    const language = document.getElementById('song-language').value;

    const score = useScore ? (composerScoreRawResponse || '') : '';
    if (!prompt && !score) {
        composerShowToast('Describe the sound, or generate a score first');
        return;
    }
    if (useScore && !score) {
        composerShowToast('No score yet — generating from your description alone');
    }

    const btn = document.getElementById('song-generate-btn');
    const content = document.getElementById('song-content');
    btn.disabled = true;
    btn.textContent = 'Rendering…';

    // The live box sits above the result, same as every other tab.
    content.innerHTML = '';
    const label = document.createElement('div');
    label.className = 'composer-streaming-label';
    label.textContent = 'Rendering';
    content.appendChild(label);
    const live = document.createElement('div');
    live.className = 'composer-streaming';
    live.textContent = 'Starting…';
    content.appendChild(live);

    const lines = [];
    try {
        const r = await fetch('/composer/song', {
            method: 'POST', headers: jsonAuthHeaders(),
            body: JSON.stringify({
                session_id: composerSessionId,
                prompt, lyrics,
                response: score,
                duration,
                cover_strength: strength,
                vocal_language: language,
            }),
        });
        if (!r.ok) {
            let detail = 'Could not start the render';
            try { detail = (await r.json()).detail || detail; } catch (e) { /* keep */ }
            live.remove();
            content.innerHTML = '<div class="message error"></div>';
            content.firstChild.textContent = detail;
            return;
        }

        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        let result = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });

            while (buf.includes('\n\n')) {
                const idx = buf.indexOf('\n\n');
                const chunk = buf.slice(0, idx);
                buf = buf.slice(idx + 2);

                let ev = '', dataStr = '';
                for (const line of chunk.split('\n')) {
                    if (line.startsWith('event: ')) ev = line.slice(7);
                    else if (line.startsWith('data: ')) dataStr += line.slice(6);
                }
                if (!dataStr) continue;

                if (ev === 'queue') {
                    try { gpuShowQueueNote(live, JSON.parse(dataStr)); }
                    catch (e) { /* malformed: ignore */ }
                } else if (ev === 'reasoning') {
                    lines.push(JSON.parse(dataStr));
                    live.textContent = lines.join('\n');
                    live.scrollTop = live.scrollHeight;
                } else if (ev === 'done') {
                    result = JSON.parse(dataStr);
                } else if (ev === 'error') {
                    gpuShowQueueNote(live, null);
                    label.remove();
                    live.remove();
                    const err = document.createElement('div');
                    err.className = 'message error';
                    err.textContent = 'Error: ' + JSON.parse(dataStr);
                    content.appendChild(err);
                    return;
                }
            }
        }

        gpuShowQueueNote(live, null);
        if (result) {
            label.remove();
            live.remove();
            renderScoreResponse(result.response, lines, content, false);
        } else {
            live.textContent = lines.join('\n') + '\n(the stream ended without a result)';
        }
    } catch (e) {
        if (typeof label !== 'undefined') label.remove();
        live.remove();
        const err = document.createElement('div');
        err.className = 'message error';
        err.textContent = 'Error: ' + e.message;
        content.appendChild(err);
    } finally {
        btn.disabled = false;
        btn.textContent = '🎤 Render song';
        refreshSongStatus();
    }
}

/* Keep the engine state fresh while the tab is open: it holds the GPU, so
   whether it is running is something the household needs to be able to see. */
function startSongStatusPolling() {
    refreshSongStatus();
    if (_songEngineTimer) return;
    _songEngineTimer = setInterval(() => {
        const panel = document.getElementById('cpanel-song');
        if (panel && panel.classList.contains('active')) refreshSongStatus();
    }, 10000);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startSongStatusPolling);
} else {
    startSongStatusPolling();
}

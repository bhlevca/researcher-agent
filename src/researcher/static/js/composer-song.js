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
// The last render, so Save knows what to keep. A render is scratch until
// saved: the server prunes anything no composition refers to.
let _songLastRender = null;

/* Fill the "Perform" list: the latest score from this sitting, plus every
   saved composition. This tab renders audio and does not write notation, so
   where the notation comes from has to be on screen rather than implied by a
   checkbox. */
async function refreshSongSources() {
    const sel = document.getElementById('song-source');
    const note = document.getElementById('song-source-note');
    if (!sel) return;
    const chosen = sel.value;
    sel.innerHTML = '';

    const none = document.createElement('option');
    none.value = '';
    none.textContent = 'Description only — no score';
    sel.appendChild(none);

    if (composerScoreRawResponse) {
        const o = document.createElement('option');
        o.value = 'latest';
        o.textContent = 'The score I just generated';
        sel.appendChild(o);
    }

    let saved = [];
    if (composerSessionId) {
        try {
            const r = await fetch(
                `/composer/sessions/${composerSessionId}/compositions`,
                { headers: authHeaders() });
            if (r.ok) saved = (await r.json()).compositions || [];
        } catch (e) { /* leave the list short */ }
    }
    for (const c of saved) {
        const o = document.createElement('option');
        o.value = 'comp:' + c.id;
        o.textContent = `Saved: ${c.title}`;
        sel.appendChild(o);
    }

    // Keep the choice across refreshes; otherwise default to a real score.
    const values = Array.from(sel.options).map(o => o.value);
    sel.value = values.includes(chosen) ? chosen
              : (composerScoreRawResponse ? 'latest'
                 : (saved.length ? 'comp:' + saved[0].id : ''));
    onSongSourceChange();

    if (note && !composerScoreRawResponse && !saved.length) {
        note.textContent = 'No score yet — compose one, or render from the description alone.';
    }
}

function onSongSourceChange() {
    const sel = document.getElementById('song-source');
    const note = document.getElementById('song-source-note');
    if (!sel || !note) return;
    if (!sel.value) {
        note.textContent = 'A new piece from your description. No score to save.';
        note.className = 'song-source-note is-warn';
    } else {
        note.textContent = 'The audio will follow this composition, and the score is saved with it.';
        note.className = 'song-source-note';
    }
}

/* The Score tab is where notation is written. Hand the description over and
   switch there, rather than pretending this tab can do it. */
function composeScoreForSong() {
    const desc = (document.getElementById('song-prompt').value || '').trim();
    const target = document.getElementById('score-description');
    if (target && desc) target.value = desc;
    switchComposerTab('score');
    composerShowToast(desc
        ? 'Description carried over — generate a score, then return to Song'
        : 'Describe what to compose, then generate a score here');
}

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
    const instrumental = document.getElementById('song-instrumental').checked;
    const source = document.getElementById('song-source').value;
    const strength = parseFloat(document.getElementById('song-strength').value);
    const duration = parseFloat(document.getElementById('song-duration').value) || 0;
    const language = document.getElementById('song-language').value;

    const score = source === 'latest' ? (composerScoreRawResponse || '') : '';
    const compositionId = source.startsWith('comp:') ? source.slice(5) : '';
    if (!prompt && !score && !compositionId) {
        composerShowToast('Describe the sound, or pick a composition to perform');
        return;
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
                prompt, lyrics, instrumental,
                response: score,
                composition_id: compositionId,
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
            _songLastRender = {
                audio_url: result.audio_url,
                has_score: !!result.has_score,
                prompt, score,
                composition_id: compositionId,
                source: result.source || '',
            };
            renderScoreResponse(result.response, lines, content, false);
            content.appendChild(buildSongSaveRow());
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
    refreshSongSources();
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


/* Saving is what makes a render permanent — unsaved songs are pruned — so the
   button says so rather than looking like an optional extra. In cover mode it
   keeps the score too, and that score is exact: it is the notation the audio
   was rendered from. */
function buildSongSaveRow() {
    const row = document.createElement('div');
    row.className = 'score-actions song-save-row';

    const label = document.createElement('label');
    label.className = 'song-save-name';
    label.textContent = 'Name:';
    const name = document.createElement('input');
    name.type = 'text';
    name.id = 'song-save-name';
    // A sensible default that is still fully editable — this is the name the
    // downloaded file gets, not just a database row.
    name.value = defaultSongName();
    name.placeholder = 'Name for this song';
    label.appendChild(name);
    row.appendChild(label);

    const save = document.createElement('button');
    save.textContent = '💾 Save song';
    save.onclick = () => saveSong(save);
    row.appendChild(save);

    const note = document.createElement('span');
    note.className = 'song-save-note';
    note.textContent = _songLastRender && _songLastRender.has_score
        ? 'Keeps the audio and its score. Unsaved renders are deleted.'
        : 'Keeps the audio. No score — this was generated from the description. Unsaved renders are deleted.';
    row.appendChild(note);
    return row;
}

/* Name it after what it is: the composition performed, or the description. */
function defaultSongName() {
    const r = _songLastRender || {};
    if (r.source) return r.source;
    const words = (r.prompt || '').split(/\s+/).filter(Boolean).slice(0, 5);
    return words.length ? words.join(' ') : 'Untitled song';
}

async function saveSong(btn) {
    if (!_songLastRender) { composerShowToast('Nothing to save'); return; }
    const field = document.getElementById('song-save-name');
    const title = ((field && field.value) || '').trim();
    if (!title) {
        composerShowToast('Give the song a name');
        if (field) field.focus();
        return;
    }

    if (btn) { btn.disabled = true; btn.textContent = 'Saving…'; }
    try {
        const r = await fetch('/composer/song/save', {
            method: 'POST', headers: jsonAuthHeaders(),
            body: JSON.stringify({
                session_id: composerSessionId,
                audio_url: _songLastRender.audio_url,
                response: _songLastRender.score || '',
                composition_id: _songLastRender.composition_id || '',
                prompt: _songLastRender.prompt || '',
                title,
            }),
        });
        if (!r.ok) {
            let detail = 'Could not save';
            try { detail = (await r.json()).detail || detail; } catch (e) { /* keep */ }
            composerShowToast(detail);
            if (btn) { btn.disabled = false; btn.textContent = '💾 Save song'; }
            return;
        }
        const d = await r.json();
        composerShowToast(d.has_score ? 'Saved, with its score'
                                      : 'Saved (audio only)');
        if (btn) { btn.textContent = '✓ Saved'; btn.disabled = true; }
        if (field) field.disabled = true;

        // Now that it is kept, offer it under the name that was given.
        const links = document.createElement('span');
        links.className = 'song-saved-links';
        const id = d.composition.id;
        const audio = document.createElement('a');
        audio.href = `/composer/compositions/${id}/audio`;
        audio.textContent = '⬇ audio';
        links.appendChild(audio);
        if (d.has_score) {
            const xml = document.createElement('a');
            xml.href = `/composer/compositions/${id}/musicxml`;
            xml.textContent = '⬇ score (MusicXML)';
            links.appendChild(xml);
        }
        if (btn.parentElement) btn.parentElement.appendChild(links);

        if (typeof loadCompositions === 'function') loadCompositions();
        refreshSongSources();
    } catch (e) {
        composerShowToast('Error: ' + e.message);
        if (btn) { btn.disabled = false; btn.textContent = '💾 Save song'; }
    }
}

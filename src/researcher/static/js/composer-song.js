// ══════════════════════════════════════════════
// ── Song audio (ACE-Step) ──
//
// Description + lyrics in, takes of audio out. Nothing here writes or attaches
// a score: ACE-Step only makes audio, and the scores this tab used to offer
// had nothing to do with what it produced.
//
// Each take gets its own player, so you can listen to all of them and save the
// ones you want. Saving does two things at once: it opens your browser's save
// dialog so the file lands where you choose, under the name you choose, and it
// keeps the take in Compositions so it is not cleaned up with the unsaved ones.
// ══════════════════════════════════════════════

let _songStatusTimer = null;
let _songSettings = { keepHours: 2 };

/* ── engine status ─────────────────────────────────────────────────────── */

async function refreshSongStatus() {
    const el = document.getElementById('song-status');
    if (!el) return null;
    try {
        const r = await fetch('/composer/song/status', { headers: authHeaders() });
        if (!r.ok) { el.textContent = 'Could not reach the server.'; return null; }
        const s = await r.json();
        _songSettings.keepHours = s.keep_hours || 2;
        fillSongModels(s.models);

        el.innerHTML = '';
        if (!s.installed) {
            el.className = 'song-status is-missing';
            el.textContent = s.hint || 'The song engine is not installed.';
            return s;
        }
        const owner = s.managed_by === 'systemd' ? `${s.unit}` : 'started by the app';
        const parts = [`Song engine: ${owner}`];
        if (!s.running) parts.push('stopped — starts on the first render');
        else if (s.holding_gpu) parts.push(`holding the GPU (${s.loaded_model || 'loaded'})`);
        else parts.push('idle, not using the GPU');
        el.appendChild(document.createTextNode(parts.join(' · ')));

        // Freeing the GPU only means something when it is held.
        if (s.holding_gpu) {
            el.appendChild(document.createTextNode(' · '));
            const free = document.createElement('a');
            free.href = '#';
            free.textContent = 'free it';
            free.title = 'Restart the engine so the GPU is available to the tutor and researcher. The next render reloads the model (~40 s).';
            free.onclick = (e) => { e.preventDefault(); stopSongEngine(); };
            el.appendChild(free);
        }
        el.className = 'song-status' + (s.holding_gpu ? ' is-running' : '');
        return s;
    } catch (e) {
        el.textContent = 'Could not reach the server.';
        return null;
    }
}

/* Fast stays the default even when the engine last ran Detailed: the choice
   is per render, and a leftover from the previous one is not a preference. */
function fillSongModels(models) {
    const sel = document.getElementById('song-model');
    if (!sel || !models || !models.length) return;
    const chosen = sel.value;
    sel.innerHTML = '';
    for (const m of models) {
        const o = document.createElement('option');
        o.value = m.id;
        o.textContent = `${m.label} — ${m.steps} steps`;
        o.title = m.note;
        sel.appendChild(o);
    }
    const ids = models.map(m => m.id);
    sel.value = ids.includes(chosen) ? chosen : ids[0];
}

async function stopSongEngine() {
    try {
        const r = await fetch('/composer/song/stop', { method: 'POST', headers: jsonAuthHeaders() });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) composerShowToast(d.detail || 'Could not free the GPU');
        else composerShowToast(d.stopped ? 'GPU freed' : 'The engine was not holding the GPU');
    } catch (e) {
        composerShowToast('Error: ' + e.message);
    } finally {
        refreshSongStatus();
    }
}

/* ── rendering ─────────────────────────────────────────────────────────── */

function _songVal(id) {
    const el = document.getElementById(id);
    return el ? (el.value || '').trim() : '';
}

async function generateSong() {
    if (!composerSessionId) { composerShowToast('Create or load a session first'); return; }
    const prompt = _songVal('song-prompt');
    if (!prompt) {
        composerShowToast('Describe the song first');
        document.getElementById('song-prompt').focus();
        return;
    }
    const lyrics = _songVal('song-lyrics');
    const seedRaw = _songVal('song-seed');
    const bpmRaw = _songVal('song-bpm');
    const body = {
        session_id: composerSessionId,
        prompt, lyrics,
        instrumental: document.getElementById('song-instrumental').checked || !lyrics,
        vocal_language: _songVal('song-language') || 'auto',
        duration: parseFloat(_songVal('song-duration')) || 0,
        takes: parseInt(_songVal('song-takes'), 10) || 2,
        model: _songVal('song-model'),
        seed: seedRaw === '' ? null : parseInt(seedRaw, 10),
        bpm: bpmRaw === '' ? null : parseInt(bpmRaw, 10),
        key_scale: _songVal('song-key'),
        time_signature: _songVal('song-time'),
    };

    const btn = document.getElementById('song-generate-btn');
    const content = document.getElementById('song-content');
    btn.disabled = true;
    btn.textContent = 'Rendering…';
    content.innerHTML = '';

    const label = document.createElement('div');
    label.className = 'composer-streaming-label';
    label.textContent = `Rendering ${body.takes} take${body.takes > 1 ? 's' : ''}`;
    const live = document.createElement('div');
    live.className = 'composer-streaming';
    live.textContent = 'Starting…';
    content.append(label, live);

    const lines = [];
    let result = null;
    try {
        const r = await fetch('/composer/song', {
            method: 'POST', headers: jsonAuthHeaders(), body: JSON.stringify(body),
        });
        if (!r.ok) {
            let detail = 'Could not start the render';
            try { detail = (await r.json()).detail || detail; } catch (e) { /* keep */ }
            throw new Error(detail);
        }
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            while (buf.includes('\n\n')) {
                const idx = buf.indexOf('\n\n');
                const chunk = buf.slice(0, idx);
                buf = buf.slice(idx + 2);
                let ev = '', data = '';
                for (const line of chunk.split('\n')) {
                    if (line.startsWith('event: ')) ev = line.slice(7);
                    else if (line.startsWith('data: ')) data += line.slice(6);
                }
                if (!data) continue;
                if (ev === 'queue') {
                    try { gpuShowQueueNote(live, JSON.parse(data)); } catch (e) { /* ignore */ }
                } else if (ev === 'reasoning') {
                    lines.push(JSON.parse(data));
                    live.textContent = lines.join('\n');
                    live.scrollTop = live.scrollHeight;
                } else if (ev === 'done') {
                    result = JSON.parse(data);
                } else if (ev === 'error') {
                    throw new Error(JSON.parse(data));
                }
            }
        }
        gpuShowQueueNote(live, null);
        label.remove();
        live.remove();
        if (!result) throw new Error('The render ended without a result');
        renderTakes(content, result, body);
    } catch (e) {
        gpuShowQueueNote(live, null);
        label.remove();
        live.remove();
        const err = document.createElement('div');
        err.className = 'message error';
        err.textContent = 'Error: ' + e.message;
        content.appendChild(err);
    } finally {
        btn.disabled = false;
        btn.textContent = '🎤 Render';
        refreshSongStatus();
    }
}

const _LANG_NAMES = {
    en: 'English', fr: 'French', ro: 'Romanian', de: 'German', es: 'Spanish',
    it: 'Italian', pt: 'Portuguese', ja: 'Japanese', zh: 'Chinese',
};

function renderTakes(container, result, request) {
    const takes = result.takes || [];
    const head = document.createElement('div');
    head.className = 'song-result-head';
    const lang = result.language ? (_LANG_NAMES[result.language] || result.language) : '';
    const sung = request.instrumental ? 'instrumental'
        : (lang ? `sung in ${lang}${request.vocal_language === 'auto' ? ' (detected)' : ''}` : '');
    head.textContent = [
        `${takes.length} take${takes.length === 1 ? '' : 's'}`,
        `${result.engine} engine, ${result.steps} steps`,
        sung,
    ].filter(Boolean).join(' · ');
    container.appendChild(head);

    takes.forEach(t => container.appendChild(buildTake(t, request)));

    const note = document.createElement('div');
    note.className = 'song-keep-note';
    note.textContent = `Takes you don't save are deleted after ${_songSettings.keepHours} hours.`;
    container.appendChild(note);
}

function _songDefaultName(request, take) {
    const words = (request.prompt || '').replace(/[^\p{L}\p{N}\s'-]/gu, ' ')
        .split(/\s+/).filter(Boolean).slice(0, 5).join(' ');
    return `${words || 'Song'} - take ${take.take}`;
}

function _fmtDuration(sec) {
    if (!sec && sec !== 0) return '';
    const s = Math.round(sec);
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function buildTake(take, request) {
    const card = document.createElement('div');
    card.className = 'song-take';

    const title = document.createElement('div');
    title.className = 'song-take-title';
    title.textContent = [`Take ${take.take}`,
                         take.duration ? _fmtDuration(take.duration) : '',
                         take.seed != null ? `seed ${take.seed}` : '']
        .filter(Boolean).join(' · ');
    card.appendChild(title);

    const audio = document.createElement('audio');
    audio.controls = true;
    audio.preload = 'metadata';
    audio.src = take.url;
    card.appendChild(audio);

    const row = document.createElement('div');
    row.className = 'song-take-actions';

    const name = document.createElement('input');
    name.type = 'text';
    name.className = 'song-take-name';
    name.value = _songDefaultName(request, take);
    name.title = 'The file name, and its name in Compositions';
    row.appendChild(name);

    const save = document.createElement('button');
    save.textContent = '💾 Save…';
    save.title = 'Choose where to save the file, and keep it in Compositions';
    save.onclick = () => saveTake(take, name, save, card, request);
    row.appendChild(save);

    if (take.seed != null) {
        const again = document.createElement('button');
        again.textContent = '↻ Use this seed';
        again.title = 'Put this seed in Advanced, so the next render reproduces this take and you can change one thing';
        again.onclick = () => {
            document.getElementById('song-seed').value = take.seed;
            const adv = document.querySelector('#cpanel-song .song-advanced');
            if (adv) adv.open = true;
            composerShowToast(`Seed ${take.seed} set — change something and render again`);
        };
        row.appendChild(again);
    }
    card.appendChild(row);
    return card;
}

/* ── saving ────────────────────────────────────────────────────────────── */

/* Write a blob where the person chooses.
 *
 * Chrome and Edge offer a real "Save as" dialog through showSaveFilePicker,
 * but only on a secure origin — localhost or https. Over plain http from
 * another machine, and in Firefox, the browser instead downloads the file
 * under the given name; whether it then asks for a folder is the browser's
 * own setting ("Always ask where to save files").
 *
 * Returns the chosen name, or null if the person cancelled.
 */
async function saveBlobAs(blob, suggestedName) {
    if (window.showSaveFilePicker) {
        try {
            const handle = await window.showSaveFilePicker({
                suggestedName,
                types: [{ description: 'MP3 audio', accept: { 'audio/mpeg': ['.mp3'] } }],
            });
            const w = await handle.createWritable();
            await w.write(blob);
            await w.close();
            return handle.name;
        } catch (e) {
            if (e && e.name === 'AbortError') return null;   // cancelled: do nothing
            // Anything else (permission, unsupported) falls back to a download.
        }
    }
    const a = document.createElement('a');
    const url = URL.createObjectURL(blob);
    a.href = url;
    a.download = suggestedName;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    return suggestedName;
}

function _safeFileName(name) {
    const n = (name || '').replace(/[\/\\:*?"<>|]+/g, ' ').replace(/\s+/g, ' ').trim();
    return (n || 'Song').replace(/\.mp3$/i, '') + '.mp3';
}

async function saveTake(take, nameField, btn, card, request) {
    const wanted = _safeFileName(nameField.value);
    btn.disabled = true;
    btn.textContent = 'Saving…';
    try {
        const blob = await (await fetch(take.url)).blob();
        const chosen = await saveBlobAs(blob, wanted);
        if (!chosen) {                       // cancelled in the dialog
            btn.disabled = false;
            btn.textContent = '💾 Save…';
            return;
        }
        // Keep it in Compositions under the name it was saved as, which also
        // takes it out of the cleanup. Saving the same take again renames it.
        const title = chosen.replace(/\.mp3$/i, '');
        const r = await fetch('/composer/song/save', {
            method: 'POST', headers: jsonAuthHeaders(),
            body: JSON.stringify({
                session_id: composerSessionId,
                file: take.file, title,
                prompt: request.prompt, seed: take.seed,
            }),
        });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(d.detail || 'Could not keep it in Compositions');

        // From now on the take lives in kept/; point the player there.
        take.file = d.file;
        take.url = d.url;
        const audio = card.querySelector('audio');
        if (audio && audio.src.indexOf(d.url) === -1) {
            const t = audio.currentTime;
            audio.src = d.url;
            audio.currentTime = t;
        }
        nameField.value = title;
        btn.textContent = '💾 Save again…';
        btn.disabled = false;
        card.classList.add('is-saved');
        composerShowToast(`Saved "${chosen}" and kept it in Compositions`);
        if (typeof loadCompositions === 'function') loadCompositions();
    } catch (e) {
        composerShowToast('Error: ' + e.message);
        btn.disabled = false;
        btn.textContent = '💾 Save…';
    }
}

/* ── status polling ────────────────────────────────────────────────────── */

function startSongStatusPolling() {
    refreshSongStatus();
    if (_songStatusTimer) return;
    _songStatusTimer = setInterval(() => {
        const panel = document.getElementById('cpanel-song');
        if (panel && panel.classList.contains('active')) refreshSongStatus();
    }, 10000);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startSongStatusPolling);
} else {
    startSongStatusPolling();
}

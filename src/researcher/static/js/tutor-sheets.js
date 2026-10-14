/* Printable sheets for the language tutor.
 *
 * The preview is the real PDF in an iframe, as on the math page: this tool
 * exists to produce paper, so the preview should be the paper. An HTML
 * approximation would be a second thing to keep in step with the first.
 */

let sheetSkills = [];
let sheetPdfUrls = { sheet: null, key: null };
let sheetShowingKey = false;
let sheetLastRequest = null;
let sheetMode = 'grammar';
let sheetStoryContent = null;      // the generated story, carried back for printing
let sheetBusy = false;
let sheetTimer = null;

/* A model call takes a minute or two. Without this the action buttons stay
   live, and pressing Preview mid-generation answered "Write a lesson first" —
   which is both wrong and the opposite of reassuring. */
function sheetSetBusy(on, label, hint) {
    sheetBusy = on;
    document.querySelectorAll('.sheets-actions button').forEach(b => {
        b.disabled = on;
    });
    document.querySelectorAll('#sheets-story-pane button')
        .forEach(b => { b.disabled = on; });
    clearInterval(sheetTimer);

    const panel = document.getElementById('sheets-working');
    const frame = document.getElementById('sheets-frame');
    const empty = document.getElementById('sheets-empty');
    if (on) {
        document.getElementById('sheets-working-what').textContent = label;
        document.getElementById('sheets-working-hint').textContent = hint || '';
        panel.style.display = 'flex';
        frame.style.display = 'none';
        empty.style.display = 'none';
        const t0 = Date.now();
        const tick = () => {
            const s = Math.round((Date.now() - t0) / 1000);
            const mm = Math.floor(s / 60), ss = String(s % 60).padStart(2, '0');
            document.getElementById('sheets-working-time').textContent =
                mm ? `${mm}:${ss}` : `${s}s`;
        };
        tick();
        sheetTimer = setInterval(tick, 1000);
        sheetStatus('');
    } else {
        panel.style.display = 'none';
        // Whichever of the two was showing before is restored by the caller.
        if (!sheetPdfUrls.sheet) empty.style.display = 'flex';
    }
}

/* The primary button does the obvious thing for the current mode, so there is
   never a second "real" button competing with it. */
function sheetSyncButton() {
    const btn = document.getElementById('sheets-go');
    if (!btn) return;
    if (sheetMode === 'story') {
        btn.textContent = sheetStoryContent ? '↻ Rewrite story' : '✍️ Write the story';
    } else {
        btn.textContent = 'Preview';
    }
}

function sheetOpenPanel() {
    document.getElementById('sheets-overlay').style.display = 'block';
    document.getElementById('sheets-panel').classList.add('open');
    sheetEnsureSkills();
}

/* sheetLoadSkills fills the Level dropdown, and it is async. Generating before
   it resolves sent level:"" — the model was then asked to write for a
   "-level student". Every generate awaits this first. */
let sheetSkillsPromise = null;
function sheetEnsureSkills() {
    if (!sheetSkillsPromise) sheetSkillsPromise = sheetLoadSkills();
    return sheetSkillsPromise;
}

function sheetClosePanel() {
    document.getElementById('sheets-overlay').style.display = 'none';
    document.getElementById('sheets-panel').classList.remove('open');
}

function sheetLevel() {
    /* Never return "" — the server rejects it, and a lesson pitched at no
       level is four wasted minutes. */
    const el = document.getElementById('sheets-level');
    return (el && el.value) || 'A2';
}

function sheetStatus(msg, bad) {
    const el = document.getElementById('sheets-status');
    if (!el) return;
    el.textContent = msg || '';
    el.style.color = bad ? '#f87171' : '#64748b';
}

async function sheetLoadSkills() {
    const lang = document.getElementById('sheets-language').value || 'French';
    try {
        const r = await fetch('/tutor/sheets/skills?language=' + encodeURIComponent(lang),
                              { headers: authHeaders() });
        if (!r.ok) throw new Error('Could not load skills');
        const d = await r.json();
        sheetSkills = d.skills;

        const langSel = document.getElementById('sheets-language');
        if (!langSel.dataset.filled) {
            langSel.innerHTML = '';
            d.languages.forEach(l => {
                const o = document.createElement('option');
                o.value = l; o.textContent = l;
                langSel.appendChild(o);
            });
            langSel.value = lang;
            langSel.dataset.filled = '1';
        }

        const lvl = document.getElementById('sheets-level');
        if (!lvl.options.length) {
            d.levels.forEach(v => {
                const o = document.createElement('option');
                o.value = v; o.textContent = v;
                lvl.appendChild(o);
            });
            lvl.value = 'A2';
        }

        sheetRenderSkillRows();
        if (!d.pdf_available) {
            sheetStatus('pdflatex is not installed — printable PDFs unavailable.', true);
        } else {
            sheetStatus('');
        }
    } catch (e) {
        sheetStatus(e.message, true);
    }
}

function sheetRenderSkillRows() {
    const box = document.getElementById('sheets-skills');
    box.innerHTML = '';
    sheetSkills.forEach(s => {
        const row = document.createElement('div');
        row.className = 'sheets-skill';
        row.innerHTML =
            `<label><input type="checkbox" value="${s.id}"> <span></span></label>` +
            `<input type="number" class="n" value="5" min="1" max="40" title="How many questions">` +
            `<input type="number" class="m" value="1" min="1" max="10" title="Marks each">`;
        row.querySelector('span').textContent = s.name;
        box.appendChild(row);
    });
    // A sheet with nothing ticked cannot be generated, so start with one.
    const first = box.querySelector('input[type=checkbox]');
    if (first) first.checked = true;
}

function sheetBuildRequest() {
    const sections = [];
    document.querySelectorAll('#sheets-skills .sheets-skill').forEach(row => {
        const cb = row.querySelector('input[type=checkbox]');
        if (!cb.checked) return;
        sections.push({
            skill: cb.value,
            count: parseInt(row.querySelector('.n').value, 10) || 5,
            marks_each: parseInt(row.querySelector('.m').value, 10) || 1,
        });
    });
    const seedRaw = document.getElementById('sheets-seed').value.trim();
    return {
        language: document.getElementById('sheets-language').value,
        level: sheetLevel(),
        native: (typeof getTutorConfig === 'function'
                 ? getTutorConfig().native_lang : 'English') || 'English',
        kind: document.getElementById('sheets-kind').value,
        title: document.getElementById('sheets-title').value.trim(),
        student: document.getElementById('sheets-student').value.trim(),
        seed: seedRaw === '' ? null : (parseInt(seedRaw, 10) || null),
        sections,
    };
}

async function sheetFetchPdf(req, answers) {
    const r = await fetch(answers ? '/tutor/sheets/answers' : '/tutor/sheets/pdf', {
        method: 'POST', headers: jsonAuthHeaders(), body: JSON.stringify(req),
    });
    if (!r.ok) {
        let detail = r.statusText;
        try { detail = (await r.json()).detail || detail; } catch (e) { /* keep */ }
        throw new Error(detail);
    }
    return URL.createObjectURL(await r.blob());
}

async function sheetPreview() {
    if (sheetBusy) return;
    // In the model-backed modes the primary button writes the thing — pressing
    // it is the user asking for a lesson, not asking why there isn't one.
    if (sheetMode === 'lesson') return generateLesson();
    if (sheetMode === 'story') return sheetGenerateStory();
    const req = sheetBuildRequest();
    if (!req.sections.length) { sheetStatus('Choose at least one skill.', true); return; }
    const btn = document.getElementById('sheets-go');
    btn.disabled = true;
    sheetStatus('Generating…');
    try {
        // Fix the seed first, so the sheet and its key are the same sheet.
        if (req.seed === null) {
            const r = await fetch('/tutor/sheets/generate', {
                method: 'POST', headers: jsonAuthHeaders(), body: JSON.stringify(req),
            });
            if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
            req.seed = (await r.json()).seed;
            document.getElementById('sheets-seed').value = req.seed;
        }
        ['sheet', 'key'].forEach(k => {
            if (sheetPdfUrls[k]) URL.revokeObjectURL(sheetPdfUrls[k]);
            sheetPdfUrls[k] = null;
        });
        sheetPdfUrls.sheet = await sheetFetchPdf(req, false);
        sheetPdfUrls.key = await sheetFetchPdf(req, true);
        sheetLastRequest = req;

        document.getElementById('sheets-empty').style.display = 'none';
        document.getElementById('sheets-frame').style.display = 'block';
        sheetShowPdf(sheetShowingKey);
        sheetStatus('Ready. The same seed always gives the same sheet.');
    } catch (e) {
        sheetStatus(e.message, true);
    } finally {
        btn.disabled = false;
    }
}

function sheetShowPdf(key) {
    sheetShowingKey = key;
    document.getElementById('sheets-tab-sheet').classList.toggle('on', !key);
    document.getElementById('sheets-tab-key').classList.toggle('on', key);
    const url = key ? sheetPdfUrls.key : sheetPdfUrls.sheet;
    if (url) document.getElementById('sheets-frame').src = url;
}

async function sheetDownload(answers) {
    if (sheetMode === 'story') return sheetDownloadStory(answers);
    const req = sheetLastRequest || sheetBuildRequest();
    if (!req.sections.length) { sheetStatus('Choose at least one skill.', true); return; }
    sheetStatus('Building PDF…');
    try {
        const url = await sheetFetchPdf(req, answers);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${req.language}-${req.kind}${answers ? '-answers' : ''}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
        sheetStatus('Downloaded.');
    } catch (e) {
        sheetStatus(e.message, true);
    }
}

async function sheetSave() {
    if (sheetMode === 'story') return sheetSaveStory();
    const req = sheetLastRequest || sheetBuildRequest();
    if (!req.sections.length) { sheetStatus('Choose at least one skill.', true); return; }
    try {
        const r = await fetch('/tutor/sheets', {
            method: 'POST', headers: jsonAuthHeaders(),
            body: JSON.stringify({ ...req, notes: '' }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        const rec = await r.json();
        sheetStatus(`Saved — reprints identically from seed ${rec.seed}.`);
    } catch (e) {
        sheetStatus(e.message, true);
    }
}

async function sheetListSaved() {
    const box = document.getElementById('sheets-saved');
    box.style.display = 'block';
    box.innerHTML = '<div style="color:#64748b;font-size:0.8rem">Loading…</div>';
    try {
        const r = await fetch('/tutor/sheets', { headers: authHeaders() });
        const { sheets } = await r.json();
        if (!sheets.length) {
            box.innerHTML = '<div style="color:#64748b;font-size:0.8rem">Nothing saved yet.</div>';
            return;
        }
        box.innerHTML = '';
        sheets.forEach(s => {
            const div = document.createElement('div');
            div.className = 'sheets-saved-item';
            div.innerHTML = '<div><strong></strong></div><div class="meta"></div>' +
                '<div class="btns"><button data-a="sheet">⬇ Sheet</button>' +
                '<button data-a="key">⬇ Key</button>' +
                '<button data-a="del">Delete</button></div>';
            div.querySelector('strong').textContent = s.title;
            div.querySelector('.meta').textContent =
                `${s.language} · ${s.level} · ${s.kind} · seed ${s.seed}`;
            div.querySelector('[data-a="sheet"]').onclick = () => sheetDownloadSaved(s.id, false);
            div.querySelector('[data-a="key"]').onclick = () => sheetDownloadSaved(s.id, true);
            div.querySelector('[data-a="del"]').onclick = async () => {
                if (!confirm('Delete this sheet?')) return;
                await fetch(`/tutor/sheets/${s.id}`, { method: 'DELETE', headers: authHeaders() });
                sheetListSaved();
            };
            box.appendChild(div);
        });
    } catch (e) {
        box.innerHTML = `<div style="color:#f87171;font-size:0.8rem">${e.message}</div>`;
    }
}

async function sheetDownloadSaved(id, answers) {
    const r = await fetch(`/tutor/sheets/${id}/pdf?answers=${answers}`,
                          { headers: authHeaders() });
    if (!r.ok) { sheetStatus('Could not build that PDF.', true); return; }
    const url = URL.createObjectURL(await r.blob());
    const a = document.createElement('a');
    a.href = url;
    a.download = `${id}${answers ? '-answers' : ''}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
}


/* ---- Story sheets ----------------------------------------------------- */

function sheetSetMode(mode) {
    sheetMode = mode;
    ['grammar', 'story'].forEach(m => {
        document.getElementById('sheets-mode-' + m).classList.toggle('on', m === mode);
        document.getElementById('sheets-' + m + '-pane').style.display =
            m === mode ? 'block' : 'none';
    });
    sheetStatus('');
    sheetSyncButton();
}

function sheetStoryBody() {
    return {
        content: sheetStoryContent,
        language: document.getElementById('sheets-language').value,
        level: sheetLevel(),
        native: (typeof getTutorConfig === 'function'
                 ? getTutorConfig().native_lang : 'English') || 'English',
        kind: document.getElementById('sheets-kind').value,
        title: document.getElementById('sheets-title').value.trim(),
        student: document.getElementById('sheets-student').value.trim(),
    };
}

async function sheetGenerateStory() {
    await sheetEnsureSkills();
    sheetSetBusy(true, 'Writing the story…', 'Usually under a minute.');
    try {
        const r = await fetch('/tutor/sheets/story', {
            method: 'POST', headers: jsonAuthHeaders(),
            body: JSON.stringify({
                language: document.getElementById('sheets-language').value,
                level: sheetLevel(),
                topic: document.getElementById('story-topic').value.trim(),
                num_questions: parseInt(document.getElementById('story-questions').value, 10) || 5,
                word_count: parseInt(document.getElementById('story-words').value, 10) || 200,
                summary_words: parseInt(document.getElementById('story-summary').value, 10) || 40,
            }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        sheetStoryContent = (await r.json()).content;
        await sheetPreviewStory();
        sheetStatus(`"${sheetStoryContent.title}" — ${sheetStoryContent.questions.length} questions.`);
    } catch (e) {
        sheetSetBusy(false);
        sheetStatus(e.message, true);
        return;
    } finally {
        sheetSetBusy(false);
        sheetSyncButton();
    }
}

async function sheetFetchStoryPdf(answers) {
    const r = await fetch(answers ? '/tutor/sheets/story/answers' : '/tutor/sheets/story/pdf', {
        method: 'POST', headers: jsonAuthHeaders(), body: JSON.stringify(sheetStoryBody()),
    });
    if (!r.ok) {
        let detail = r.statusText;
        try { detail = (await r.json()).detail || detail; } catch (e) { /* keep */ }
        throw new Error(detail);
    }
    return URL.createObjectURL(await r.blob());
}

async function sheetPreviewStory() {
    ['sheet', 'key'].forEach(k => {
        if (sheetPdfUrls[k]) URL.revokeObjectURL(sheetPdfUrls[k]);
        sheetPdfUrls[k] = null;
    });
    sheetPdfUrls.sheet = await sheetFetchStoryPdf(false);
    sheetPdfUrls.key = await sheetFetchStoryPdf(true);
    document.getElementById('sheets-empty').style.display = 'none';
    document.getElementById('sheets-frame').style.display = 'block';
    sheetShowPdf(sheetShowingKey);
}

async function sheetDownloadStory(answers) {
    if (!sheetStoryContent) { sheetStatus('Press "Write the story" first.', true); return; }
    try {
        const url = await sheetFetchStoryPdf(answers);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${sheetStoryContent.title || 'story'}${answers ? '-answers' : ''}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
        sheetStatus('Downloaded.');
    } catch (e) {
        sheetStatus(e.message, true);
    }
}

async function sheetSaveStory() {
    if (!sheetStoryContent) { sheetStatus('Press "Write the story" first.', true); return; }
    try {
        const r = await fetch('/tutor/sheets/story/save', {
            method: 'POST', headers: jsonAuthHeaders(),
            body: JSON.stringify({ ...sheetStoryBody(), notes: '' }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        sheetStatus('Saved — the story is stored as written, so a reprint matches.');
    } catch (e) {
        sheetStatus(e.message, true);
    }
}

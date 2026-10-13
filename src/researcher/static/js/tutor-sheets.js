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
let storyContent = null;      // the generated story, carried back for printing
let lessonMarkdown = null;    // the generated lesson, likewise

function openSheetsPanel() {
    document.getElementById('sheets-overlay').style.display = 'block';
    document.getElementById('sheets-panel').classList.add('open');
    if (!sheetSkills.length) loadSheetSkills();
}

function closeSheetsPanel() {
    document.getElementById('sheets-overlay').style.display = 'none';
    document.getElementById('sheets-panel').classList.remove('open');
}

function sheetStatus(msg, bad) {
    const el = document.getElementById('sheets-status');
    if (!el) return;
    el.textContent = msg || '';
    el.style.color = bad ? '#f87171' : '#64748b';
}

async function loadSheetSkills() {
    const lang = document.getElementById('sheets-language').value;
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

        renderSkillRows();
        if (!d.pdf_available) {
            sheetStatus('pdflatex is not installed — printable PDFs unavailable.', true);
        } else {
            sheetStatus('');
        }
    } catch (e) {
        sheetStatus(e.message, true);
    }
}

function renderSkillRows() {
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

function sheetRequest() {
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
        level: document.getElementById('sheets-level').value,
        kind: document.getElementById('sheets-kind').value,
        title: document.getElementById('sheets-title').value.trim(),
        student: document.getElementById('sheets-student').value.trim(),
        seed: seedRaw === '' ? null : (parseInt(seedRaw, 10) || null),
        sections,
    };
}

async function fetchSheetPdf(req, answers) {
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

async function previewSheet() {
    if (sheetMode === 'lesson') {
        if (!lessonMarkdown) { sheetStatus('Write or load a lesson first.', true); return; }
        try { await previewLesson(); } catch (e) { sheetStatus(e.message, true); }
        return;
    }
    if (sheetMode === 'story') {
        if (!storyContent) { sheetStatus('Write the story first.', true); return; }
        try { await previewStory(); } catch (e) { sheetStatus(e.message, true); }
        return;
    }
    const req = sheetRequest();
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
        sheetPdfUrls.sheet = await fetchSheetPdf(req, false);
        sheetPdfUrls.key = await fetchSheetPdf(req, true);
        sheetLastRequest = req;

        document.getElementById('sheets-empty').style.display = 'none';
        document.getElementById('sheets-frame').style.display = 'block';
        showSheetPdf(sheetShowingKey);
        sheetStatus('Ready. The same seed always gives the same sheet.');
    } catch (e) {
        sheetStatus(e.message, true);
    } finally {
        btn.disabled = false;
    }
}

function showSheetPdf(key) {
    sheetShowingKey = key;
    document.getElementById('sheets-tab-sheet').classList.toggle('on', !key);
    document.getElementById('sheets-tab-key').classList.toggle('on', key);
    const url = key ? sheetPdfUrls.key : sheetPdfUrls.sheet;
    if (url) document.getElementById('sheets-frame').src = url;
}

async function downloadSheet(answers) {
    if (sheetMode === 'lesson') return downloadLesson(answers);
    if (sheetMode === 'story') return downloadStory(answers);
    const req = sheetLastRequest || sheetRequest();
    if (!req.sections.length) { sheetStatus('Choose at least one skill.', true); return; }
    sheetStatus('Building PDF…');
    try {
        const url = await fetchSheetPdf(req, answers);
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

async function saveSheet() {
    if (sheetMode === 'lesson') {
        sheetStatus('Lessons are saved from the Lessons panel, not here.', true);
        return;
    }
    if (sheetMode === 'story') return saveStory();
    const req = sheetLastRequest || sheetRequest();
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

async function listSavedSheets() {
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
            div.querySelector('[data-a="sheet"]').onclick = () => downloadSaved(s.id, false);
            div.querySelector('[data-a="key"]').onclick = () => downloadSaved(s.id, true);
            div.querySelector('[data-a="del"]').onclick = async () => {
                if (!confirm('Delete this sheet?')) return;
                await fetch(`/tutor/sheets/${s.id}`, { method: 'DELETE', headers: authHeaders() });
                listSavedSheets();
            };
            box.appendChild(div);
        });
    } catch (e) {
        box.innerHTML = `<div style="color:#f87171;font-size:0.8rem">${e.message}</div>`;
    }
}

async function downloadSaved(id, answers) {
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

function setSheetMode(mode) {
    sheetMode = mode;
    ['grammar', 'story', 'lesson'].forEach(m => {
        document.getElementById('sheets-mode-' + m).classList.toggle('on', m === mode);
        document.getElementById('sheets-' + m + '-pane').style.display =
            m === mode ? 'block' : 'none';
    });
    sheetStatus('');
}

function storyBody() {
    return {
        content: storyContent,
        language: document.getElementById('sheets-language').value,
        level: document.getElementById('sheets-level').value,
        kind: document.getElementById('sheets-kind').value,
        title: document.getElementById('sheets-title').value.trim(),
        student: document.getElementById('sheets-student').value.trim(),
    };
}

async function generateStory() {
    const btn = document.getElementById('story-go');
    btn.disabled = true;
    sheetStatus('Writing the story… this uses the model and can take a minute.');
    try {
        const r = await fetch('/tutor/sheets/story', {
            method: 'POST', headers: jsonAuthHeaders(),
            body: JSON.stringify({
                language: document.getElementById('sheets-language').value,
                level: document.getElementById('sheets-level').value,
                topic: document.getElementById('story-topic').value.trim(),
                num_questions: parseInt(document.getElementById('story-questions').value, 10) || 5,
                word_count: parseInt(document.getElementById('story-words').value, 10) || 200,
                summary_words: parseInt(document.getElementById('story-summary').value, 10) || 40,
            }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        storyContent = (await r.json()).content;
        await previewStory();
        sheetStatus(`"${storyContent.title}" — ${storyContent.questions.length} questions.`);
    } catch (e) {
        sheetStatus(e.message, true);
    } finally {
        btn.disabled = false;
    }
}

async function fetchStoryPdf(answers) {
    const r = await fetch(answers ? '/tutor/sheets/story/answers' : '/tutor/sheets/story/pdf', {
        method: 'POST', headers: jsonAuthHeaders(), body: JSON.stringify(storyBody()),
    });
    if (!r.ok) {
        let detail = r.statusText;
        try { detail = (await r.json()).detail || detail; } catch (e) { /* keep */ }
        throw new Error(detail);
    }
    return URL.createObjectURL(await r.blob());
}

async function previewStory() {
    ['sheet', 'key'].forEach(k => {
        if (sheetPdfUrls[k]) URL.revokeObjectURL(sheetPdfUrls[k]);
        sheetPdfUrls[k] = null;
    });
    sheetPdfUrls.sheet = await fetchStoryPdf(false);
    sheetPdfUrls.key = await fetchStoryPdf(true);
    document.getElementById('sheets-empty').style.display = 'none';
    document.getElementById('sheets-frame').style.display = 'block';
    showSheetPdf(sheetShowingKey);
}

async function downloadStory(answers) {
    if (!storyContent) { sheetStatus('Write the story first.', true); return; }
    try {
        const url = await fetchStoryPdf(answers);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${storyContent.title || 'story'}${answers ? '-answers' : ''}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
        sheetStatus('Downloaded.');
    } catch (e) {
        sheetStatus(e.message, true);
    }
}

async function saveStory() {
    if (!storyContent) { sheetStatus('Write the story first.', true); return; }
    try {
        const r = await fetch('/tutor/sheets/story/save', {
            method: 'POST', headers: jsonAuthHeaders(),
            body: JSON.stringify({ ...storyBody(), notes: '' }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        sheetStatus('Saved — the story is stored as written, so a reprint matches.');
    } catch (e) {
        sheetStatus(e.message, true);
    }
}


/* ---- Lesson sheets ---------------------------------------------------- */

function lessonBody() {
    return {
        markdown: lessonMarkdown,
        title: document.getElementById('sheets-title').value.trim(),
        language: document.getElementById('sheets-language').value,
        level: document.getElementById('sheets-level').value,
        student: document.getElementById('sheets-student').value.trim(),
        practice_space: true,
    };
}

async function fetchLessonPdf(answers) {
    const r = await fetch(answers ? '/tutor/sheets/lesson/answers' : '/tutor/sheets/lesson/pdf', {
        method: 'POST', headers: jsonAuthHeaders(), body: JSON.stringify(lessonBody()),
    });
    if (!r.ok) {
        let detail = r.statusText;
        try { detail = (await r.json()).detail || detail; } catch (e) { /* keep */ }
        throw new Error(detail);
    }
    return URL.createObjectURL(await r.blob());
}

async function previewLesson() {
    ['sheet', 'key'].forEach(k => {
        if (sheetPdfUrls[k]) URL.revokeObjectURL(sheetPdfUrls[k]);
        sheetPdfUrls[k] = null;
    });
    sheetPdfUrls.sheet = await fetchLessonPdf(false);
    try {
        sheetPdfUrls.key = await fetchLessonPdf(true);
    } catch (e) {
        // A lesson without a Solutions section has no key; say so rather than
        // showing the sheet again under an "Answer key" tab.
        sheetPdfUrls.key = null;
        sheetStatus(e.message, true);
    }
    document.getElementById('sheets-empty').style.display = 'none';
    document.getElementById('sheets-frame').style.display = 'block';
    showSheetPdf(sheetShowingKey && !!sheetPdfUrls.key);
}

async function generateLesson() {
    const btn = document.getElementById('lesson-go');
    btn.disabled = true;
    sheetStatus('Writing the lesson… this uses the model and takes a minute or two.');
    try {
        const r = await fetch('/tutor/sheets/lesson', {
            method: 'POST', headers: jsonAuthHeaders(),
            body: JSON.stringify({
                language: document.getElementById('sheets-language').value,
                level: document.getElementById('sheets-level').value,
                topic: document.getElementById('lesson-topic').value.trim(),
            }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        const d = await r.json();
        lessonMarkdown = d.markdown;
        await previewLesson();
        sheetStatus(d.has_answers
            ? 'Ready — the lesson includes solutions, so the answer key works.'
            : 'Ready, but the lesson has no Solutions section: no answer key.');
    } catch (e) {
        sheetStatus(e.message, true);
    } finally {
        btn.disabled = false;
    }
}

async function downloadLesson(answers) {
    if (!lessonMarkdown) { sheetStatus('Write or load a lesson first.', true); return; }
    try {
        const url = await fetchLessonPdf(answers);
        const a = document.createElement('a');
        a.href = url;
        a.download = `lesson${answers ? '-answers' : ''}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
        sheetStatus('Downloaded.');
    } catch (e) {
        sheetStatus(e.message, true);
    }
}

/* Lessons saved earlier print with no model call at all. */
async function listSavedLessons() {
    const box = document.getElementById('lesson-saved');
    box.style.display = 'block';
    box.innerHTML = '<div style="color:#64748b;font-size:0.8rem">Loading…</div>';
    try {
        const r = await fetch('/tutor/lessons', { headers: authHeaders() });
        const data = await r.json();
        const lessons = data.lessons || data || [];
        if (!lessons.length) {
            box.innerHTML = '<div style="color:#64748b;font-size:0.8rem">'
                + 'No saved lessons yet — create one from the Lessons panel.</div>';
            return;
        }
        box.innerHTML = '';
        lessons.forEach(l => {
            const div = document.createElement('div');
            div.className = 'sheets-saved-item';
            div.innerHTML = '<div><strong></strong></div><div class="meta"></div>' +
                '<div class="btns"><button data-a="sheet">⬇ Sheet</button>' +
                '<button data-a="key">⬇ Key</button></div>';
            div.querySelector('strong').textContent = l.title || l.topic || 'Lesson';
            div.querySelector('.meta').textContent =
                `${l.target_lang || ''} · ${l.level || ''}`;
            div.querySelector('[data-a="sheet"]').onclick = () => downloadSavedLesson(l.id, false);
            div.querySelector('[data-a="key"]').onclick = () => downloadSavedLesson(l.id, true);
            box.appendChild(div);
        });
    } catch (e) {
        box.innerHTML = `<div style="color:#f87171;font-size:0.8rem">${e.message}</div>`;
    }
}

async function downloadSavedLesson(id, answers) {
    const r = await fetch(`/tutor/lessons/${id}/sheet?answers=${answers}`,
                          { headers: authHeaders() });
    if (!r.ok) {
        let detail = r.statusText;
        try { detail = (await r.json()).detail || detail; } catch (e) { /* keep */ }
        sheetStatus(detail, true);
        return;
    }
    const url = URL.createObjectURL(await r.blob());
    const a = document.createElement('a');
    a.href = url;
    a.download = `lesson-${id}${answers ? '-answers' : ''}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    sheetStatus('Downloaded.');
}

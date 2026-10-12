/* Math Tutor UI.
 *
 * The preview is the real PDF in an iframe rather than an HTML rendering of the
 * same problems. Mathematics typeset in HTML does not look like the printed
 * sheet — fractions misalign, radicals do not stretch — and this tool exists to
 * produce paper, so the preview should be the paper.
 */

let mathCurriculum = [];
let mathNaming = 'grade';       // which code leads in the level selector
let mathLastRequest = null;     // the request that produced the current preview
let mathPdfUrls = { sheet: null, key: null };
let mathShowingKey = false;

/* ------------------------------------------------------------------ level */

function levelText(g) {
    const ib = g.ib_label || g.ib;
    if (!ib) return g.label;
    return mathNaming === 'ib' ? `${ib}  ·  ${g.label}` : `${g.label}  ·  ${ib}`;
}

function setNaming(mode) {
    mathNaming = mode;
    document.getElementById('math-name-grade').classList.toggle('on', mode === 'grade');
    document.getElementById('math-name-ib').classList.toggle('on', mode === 'ib');
    const sel = document.getElementById('math-grade');
    const keep = sel.value;
    renderGrades();
    sel.value = keep;
}

function renderGrades() {
    const sel = document.getElementById('math-grade');
    sel.innerHTML = '';
    mathCurriculum.forEach(g => {
        const o = document.createElement('option');
        o.value = g.id;
        o.textContent = levelText(g);
        sel.appendChild(o);
    });
}

function currentGrade() {
    const id = document.getElementById('math-grade').value;
    return mathCurriculum.find(g => g.id === id);
}

function onGradeChange() {
    const g = currentGrade();
    if (!g) return;
    const box = document.getElementById('math-topics');
    box.innerHTML = '';
    g.topics.forEach(t => {
        const row = document.createElement('label');
        row.className = 'math-topic';
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.value = t.id;
        cb.checked = t.available;
        cb.disabled = !t.available;
        cb.addEventListener('change', updateTopicCount);
        row.appendChild(cb);
        const span = document.createElement('span');
        span.textContent = t.name;
        if (t.geometry) {
            const geo = document.createElement('span');
            geo.className = 'geo';
            geo.textContent = ' ✏️';
            geo.title = 'Includes a drawn figure';
            span.appendChild(geo);
        }
        row.appendChild(span);
        box.appendChild(row);
    });
    updateTopicCount();
}

function selectedTopics() {
    return Array.from(document.querySelectorAll('#math-topics input:checked'))
                .map(cb => cb.value);
}

function updateTopicCount() {
    const n = selectedTopics().length;
    document.getElementById('math-topic-count').textContent = n ? `(${n} selected)` : '(none)';
}

function selectTopics(on) {
    document.querySelectorAll('#math-topics input:not(:disabled)')
            .forEach(cb => { cb.checked = on; });
    updateTopicCount();
}

/* ------------------------------------------------------------- generation */

function buildRequest() {
    const seedRaw = document.getElementById('math-seed').value.trim();
    return {
        grade: document.getElementById('math-grade').value,
        topics: selectedTopics(),
        count: parseInt(document.getElementById('math-count').value, 10) || 10,
        seed: seedRaw === '' ? null : (parseInt(seedRaw, 10) || null),
        title: document.getElementById('math-title').value || 'Mathematics',
        student: document.getElementById('math-student').value.trim(),
        mix: true,
    };
}

function status(msg, bad) {
    const el = document.getElementById('math-status');
    el.textContent = msg || '';
    el.style.color = bad ? '#f87171' : '#64748b';
}

async function fetchPdf(req, answers) {
    const r = await fetch(answers ? '/math/answers' : '/math/pdf', {
        method: 'POST',
        headers: jsonAuthHeaders(),
        body: JSON.stringify(req),
    });
    if (!r.ok) {
        let detail = r.statusText;
        try { detail = (await r.json()).detail || detail; } catch (e) { /* keep */ }
        throw new Error(detail);
    }
    return URL.createObjectURL(await r.blob());
}

function revokePreview() {
    ['sheet', 'key'].forEach(k => {
        if (mathPdfUrls[k]) URL.revokeObjectURL(mathPdfUrls[k]);
        mathPdfUrls[k] = null;
    });
}

async function generateSheet() {
    const req = buildRequest();
    if (!req.topics.length) { status('Select at least one topic.', true); return; }

    const btn = document.getElementById('math-go');
    btn.disabled = true;
    status('Generating…');
    try {
        // Fix the seed now, so the worksheet and its key are the same sheet.
        if (req.seed === null) {
            const r = await fetch('/math/generate', {
                method: 'POST', headers: jsonAuthHeaders(), body: JSON.stringify(req),
            });
            if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
            req.seed = (await r.json()).seed;
            document.getElementById('math-seed').value = req.seed;
        }
        revokePreview();
        mathPdfUrls.sheet = await fetchPdf(req, false);
        mathPdfUrls.key = await fetchPdf(req, true);
        mathLastRequest = req;

        document.getElementById('math-empty').style.display = 'none';
        document.getElementById('math-frame').style.display = 'block';
        showPdf(mathShowingKey);

        const g = currentGrade();
        document.getElementById('math-preview-meta').textContent =
            `${req.count} questions · ${levelText(g)} · seed ${req.seed}`;
        status('Ready. Same seed always gives the same sheet.');
    } catch (e) {
        status(e.message, true);
    } finally {
        btn.disabled = false;
    }
}

function showPdf(key) {
    mathShowingKey = key;
    document.getElementById('math-tab-sheet').classList.toggle('on', !key);
    document.getElementById('math-tab-key').classList.toggle('on', key);
    const url = key ? mathPdfUrls.key : mathPdfUrls.sheet;
    if (url) document.getElementById('math-frame').src = url;
}

async function downloadSheet(answers) {
    const req = mathLastRequest || buildRequest();
    if (!req.topics.length) { status('Select at least one topic.', true); return; }
    status('Building PDF…');
    try {
        const url = await fetchPdf(req, answers);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${req.title.replace(/\s+/g, '-')}-${answers ? 'answers' : 'worksheet'}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
        status('Downloaded.');
    } catch (e) {
        status(e.message, true);
    }
}

async function saveSheet() {
    const req = mathLastRequest || buildRequest();
    if (!req.topics.length) { status('Select at least one topic.', true); return; }
    try {
        const r = await fetch('/math/worksheets', {
            method: 'POST', headers: jsonAuthHeaders(),
            body: JSON.stringify({ ...req, notes: '' }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        const rec = await r.json();
        status(`Saved as "${rec.title}" — reprints identically from seed ${rec.seed}.`);
    } catch (e) {
        status(e.message, true);
    }
}

/* ------------------------------------------------------------- side panel */

function openSide(title) {
    document.getElementById('math-side-title').textContent = title;
    document.getElementById('math-side').style.display = 'block';
    return document.getElementById('math-side-body');
}

async function openSaved() {
    const body = openSide('Saved worksheets');
    body.innerHTML = '<div style="color:#64748b;font-size:0.8rem">Loading…</div>';
    try {
        const r = await fetch('/math/worksheets', { headers: authHeaders() });
        const { worksheets } = await r.json();
        if (!worksheets.length) {
            body.innerHTML = '<div style="color:#64748b;font-size:0.8rem">Nothing saved yet.</div>';
            return;
        }
        body.innerHTML = '';
        worksheets.forEach(w => {
            const g = mathCurriculum.find(x => x.id === w.grade);
            const div = document.createElement('div');
            div.className = 'math-saved-item';
            div.innerHTML =
                `<div><strong></strong></div>` +
                `<div class="meta"></div>` +
                `<div class="btns">` +
                `<button data-act="sheet">⬇ Sheet</button>` +
                `<button data-act="key">⬇ Key</button>` +
                `<button data-act="load">Load</button>` +
                `<button data-act="del">Delete</button></div>`;
            div.querySelector('strong').textContent = w.title;
            div.querySelector('.meta').textContent =
                `${g ? levelText(g) : w.grade} · ${w.count} questions · seed ${w.seed}` +
                (w.student ? ` · ${w.student}` : '');
            div.querySelector('[data-act="sheet"]').onclick = () => downloadSaved(w.id, false);
            div.querySelector('[data-act="key"]').onclick = () => downloadSaved(w.id, true);
            div.querySelector('[data-act="load"]').onclick = () => loadSaved(w);
            div.querySelector('[data-act="del"]').onclick = () => deleteSaved(w.id);
            body.appendChild(div);
        });
    } catch (e) {
        body.innerHTML = `<div style="color:#f87171;font-size:0.8rem">${e.message}</div>`;
    }
}

async function downloadSaved(id, answers) {
    const r = await fetch(`/math/worksheets/${id}/pdf?answers=${answers}`,
                          { headers: authHeaders() });
    if (!r.ok) { status('Could not build that PDF.', true); return; }
    const url = URL.createObjectURL(await r.blob());
    const a = document.createElement('a');
    a.href = url;
    a.download = `${id}-${answers ? 'answers' : 'worksheet'}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
}

function loadSaved(w) {
    document.getElementById('math-grade').value = w.grade;
    onGradeChange();
    document.querySelectorAll('#math-topics input').forEach(cb => {
        cb.checked = w.topics.includes(cb.value);
    });
    updateTopicCount();
    document.getElementById('math-count').value = w.count;
    document.getElementById('math-seed').value = w.seed;
    document.getElementById('math-title').value = w.title;
    document.getElementById('math-student').value = w.student || '';
    document.getElementById('math-side').style.display = 'none';
    generateSheet();
}

async function deleteSaved(id) {
    if (!confirm('Delete this worksheet?')) return;
    await fetch(`/math/worksheets/${id}`, { method: 'DELETE', headers: authHeaders() });
    openSaved();
}

async function openProgress() {
    const body = openSide('Progress by topic');
    body.innerHTML = '<div style="color:#64748b;font-size:0.8rem">Loading…</div>';
    try {
        const r = await fetch('/math/progress', { headers: authHeaders() });
        const { topics } = await r.json();
        if (!topics.length) {
            body.innerHTML = '<div style="color:#64748b;font-size:0.8rem;line-height:1.5">' +
                'No marked sheets yet. Record results against a saved worksheet and ' +
                'the weakest topics will be listed here first.</div>';
            return;
        }
        body.innerHTML = '';
        topics.forEach(t => {
            const pct = Math.round(t.accuracy * 100);
            const cls = pct < 50 ? 'weak' : pct < 80 ? 'mid' : '';
            const div = document.createElement('div');
            div.className = 'math-saved-item';
            div.innerHTML = `<div><strong></strong></div>` +
                `<div class="meta">${t.correct}/${t.attempted} correct — ${pct}%</div>` +
                `<div class="math-bar ${cls}"><div style="width:${pct}%"></div></div>`;
            div.querySelector('strong').textContent = t.name;
            body.appendChild(div);
        });
    } catch (e) {
        body.innerHTML = `<div style="color:#f87171;font-size:0.8rem">${e.message}</div>`;
    }
}

/* ------------------------------------------------------------------- init */

async function loadMathCurriculum() {
    const r = await fetch('/math/curriculum', { headers: authHeaders() });
    if (!r.ok) throw new Error('Could not load the curriculum');
    const data = await r.json();
    mathCurriculum = data.grades;
    renderGrades();
    document.getElementById('math-grade').value = 'grade8';
    onGradeChange();
    if (!data.pdf_available) {
        status('pdflatex is not installed — printable PDFs are unavailable.', true);
    }
}

(async function initMath() {
    try {
        const cfg = await (await fetch('/auth/config')).json();
        inviteRequired = cfg.invite_required;
    } catch (e) { /* proceed */ }

    const token = localStorage.getItem('authToken');
    if (token) {
        try {
            const r = await fetch('/auth/me', { headers: { Authorization: 'Bearer ' + token } });
            if (!r.ok) setAuthState(null, null);
        } catch (e) { setAuthState(null, null); }
    }
    updateAuthUI();

    if (!authUser) {
        openAuthModal('login');
    } else {
        loadMathCurriculum().catch(e => status(e.message, true));
    }
})();

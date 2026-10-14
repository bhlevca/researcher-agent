// ══════════════════════════════════════════════
// ── Reading passages — comprehensible input ──
// ══════════════════════════════════════════════
//
// The tutor's other tabs are all *output*: quizzes, drills, roleplay. This tab
// is the input half. A passage is generated at the learner's CEFR level with
// their due vocabulary woven in, so review happens in context rather than on
// isolated flashcards — and it can be listened to as well as read.

let tutorPassageRaw = '';
let tutorPassageSeeded = [];

// Below this share of new words the passage is too easy to teach anything;
// above it, comprehension breaks down and the input stops being comprehensible.
const PASSAGE_IDEAL_MIN = 0.05;
const PASSAGE_IDEAL_MAX = 0.15;

// The ratio compares the passage against words you have SAVED, which is only a
// proxy for what you actually know. Below this many saved words the number is
// dominated by vocabulary you know but never recorded, so we decline to judge
// rather than always reporting "hard".
const PASSAGE_MIN_VOCAB_TO_JUDGE = 50;

function _passageFitLabel(ratio, knownCount) {
    if (typeof knownCount === 'number' && knownCount < PASSAGE_MIN_VOCAB_TO_JUDGE) {
        return {
            text: 'save more vocabulary to gauge difficulty (' +
                  knownCount + '/' + PASSAGE_MIN_VOCAB_TO_JUDGE + ')',
            color: '#94a3b8',
        };
    }
    if (ratio < PASSAGE_IDEAL_MIN) {
        return { text: 'easy — try a harder topic or a higher level', color: '#60a5fa' };
    }
    if (ratio > PASSAGE_IDEAL_MAX) {
        return { text: 'hard — over 15% new content words', color: '#f59e0b' };
    }
    return { text: 'well pitched (i+1)', color: '#4ade80' };
}

async function showDueCount() {
    const hint = document.getElementById('passage-due-hint');
    if (!hint || !tutorSessionId) return;
    try {
        const cfg = getTutorConfig();
        const resp = await fetch('/tutor/vocabulary/due?lang=' + encodeURIComponent(cfg.target_lang), {
            headers: tutorAuthHeaders(),
        });
        if (!resp.ok) return;
        const data = await resp.json();
        const n = (data.vocabulary || []).length;
        hint.textContent = n
            ? n + ' word(s) due for review will be woven into the passage.'
            : 'No words due right now — the passage will use level-appropriate vocabulary.';
    } catch (_) { /* hint is optional */ }
}

async function generatePassage() {
    if (!tutorSessionId) { tutorShowToast('Create or load a session first'); return; }

    const topic = document.getElementById('passage-topic').value.trim();
    const wordCount = parseInt(document.getElementById('passage-length').value, 10);
    const seed = document.getElementById('passage-seed').checked;

    const content = document.getElementById('passage-content');
    content.innerHTML = '';

    const liveWrapper = document.createElement('div');
    const label = document.createElement('div');
    label.className = 'tutor-streaming-label';
    label.innerHTML = '<span class="pulse"></span> Writing passage...';
    liveWrapper.appendChild(label);
    const liveBox = document.createElement('div');
    liveBox.className = 'tutor-streaming';
    liveWrapper.appendChild(liveBox);
    content.appendChild(liveWrapper);

    const reasoningLines = [];
    try {
        const resp = await fetch('/tutor/passages', {
            method: 'POST',
            headers: tutorJsonHeaders(),
            body: JSON.stringify({
                session_id: tutorSessionId,
                topic: topic,
                word_count: wordCount,
                seed_due_vocab: seed,
            }),
        });
        if (!resp.ok) {
            liveWrapper.remove();
            tutorShowToast('Failed to generate passage');
            return;
        }

        const result = await processTutorSSE(resp, liveBox, reasoningLines);
        collapseThinking(liveWrapper, liveBox, 'Thinking — passage');
        if (!result || !result.response) return;

        tutorPassageRaw = result.response;
        const viewer = document.createElement('div');
        viewer.className = 'lesson-viewer';
        viewer.innerHTML = renderContent(result.response);
        content.appendChild(viewer);

        const actions = document.createElement('div');
        actions.style.cssText = 'display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.75rem;align-items:center;';

        // Listening practice is nearly free: the TTS is already there, and the
        // research puts listening alongside reading as comprehensible input.
        const listenBtn = document.createElement('button');
        listenBtn.textContent = '🔊 Listen';
        listenBtn.title = 'Read the passage aloud in the target language';
        listenBtn.style.cssText = 'background:#0f3460;color:#e2e8f0;border:1px solid #1a3a6e;padding:0.45rem 1rem;border-radius:0.5rem;cursor:pointer;font-size:0.85rem;';
        listenBtn.onclick = () => speakPassage(result.response);
        actions.appendChild(listenBtn);

        const saveBtn = document.createElement('button');
        saveBtn.textContent = '💾 Save';
        saveBtn.style.cssText = 'background:#e94560;color:white;border:none;padding:0.45rem 1rem;border-radius:0.5rem;cursor:pointer;font-size:0.85rem;';
        saveBtn.onclick = () => savePassage(topic);
        actions.appendChild(saveBtn);

        content.appendChild(actions);
    } catch (e) {
        liveWrapper.remove();
        tutorShowToast('Error: ' + e.message);
    }
}

// Strip the glossary and questions before speaking — only the prose is meant
// to be listened to.
function _passageProseOnly(markdown) {
    const cut = markdown.search(/\n#{2,3}\s*(Glossary|Questions)/i);
    let prose = cut > 0 ? markdown.slice(0, cut) : markdown;
    return prose.replace(/^#{1,6}\s*/gm, '').replace(/[*_`|]/g, '').trim();
}

function speakPassage(markdown) {
    const text = _passageProseOnly(markdown);
    if (!text) { tutorShowToast('Nothing to read aloud'); return; }
    if (typeof speakText === 'function') {
        // Passages are always in the target language; say so rather than let the
        // server infer it from whatever voice happens to be selected.
        speakText(text, null, getTutorConfig().target_lang);
    } else {
        tutorShowToast('TTS unavailable');
    }
}

async function savePassage(topic) {
    if (!tutorPassageRaw) { tutorShowToast('No passage to save'); return; }
    try {
        const resp = await fetch('/tutor/passages/save', {
            method: 'POST',
            headers: tutorJsonHeaders(),
            body: JSON.stringify({
                session_id: tutorSessionId,
                topic: topic,
                content: tutorPassageRaw,
                seeded_words: tutorPassageSeeded,
            }),
        });
        if (!resp.ok) { tutorShowToast('Failed to save passage'); return; }
        const saved = await resp.json();
        const pct = Math.round((saved.unknown_ratio || 0) * 100);
        const fit = _passageFitLabel(saved.unknown_ratio || 0, saved.known_word_count);
        tutorShowToast('Saved — ' + pct + '% new words, ' + fit.text);
        loadPassages();
    } catch (e) { tutorShowToast('Error: ' + e.message); }
}

async function loadPassages() {
    if (!tutorSessionId) return;
    const list = document.getElementById('passage-list');
    if (!list) return;
    try {
        const resp = await fetch('/tutor/sessions/' + tutorSessionId + '/passages', {
            headers: tutorAuthHeaders(),
        });
        const data = await resp.json();
        if (!data.passages || data.passages.length === 0) {
            list.innerHTML = '<div style="color:#64748b;text-align:center;padding:1rem;">No saved passages yet</div>';
            return;
        }
        list.innerHTML = '';
        data.passages.forEach(p => {
            const pct = Math.round((p.unknown_ratio || 0) * 100);
            const fit = _passageFitLabel(p.unknown_ratio || 0);
            const card = document.createElement('div');
            card.className = 'lesson-card';
            card.innerHTML =
                '<div class="lesson-card-title">' + escapeHtml(p.title || p.topic) + '</div>' +
                '<div class="lesson-card-meta">' +
                    p.level + ' · ' + (p.word_count || 0) + ' words · ' +
                    '<span style="color:' + fit.color + '">' + pct + '% new — ' + fit.text + '</span>' +
                    ' · ' + new Date(p.created_at).toLocaleString() +
                '</div>';
            card.onclick = () => viewPassage(p.id);
            list.appendChild(card);
        });
    } catch (e) {
        list.innerHTML = '<div style="color:#ef4444;padding:1rem;">Failed to load passages</div>';
    }
}

async function viewPassage(id) {
    try {
        const resp = await fetch('/tutor/passages/' + id, { headers: tutorAuthHeaders() });
        if (!resp.ok) { tutorShowToast('Passage not found'); return; }
        const p = await resp.json();
        tutorPassageRaw = p.content;
        const content = document.getElementById('passage-content');
        content.innerHTML = '';
        const viewer = document.createElement('div');
        viewer.className = 'lesson-viewer';
        viewer.innerHTML = renderContent(p.content);
        content.appendChild(viewer);

        const actions = document.createElement('div');
        actions.style.cssText = 'display:flex;gap:0.5rem;margin-top:0.75rem;';
        const listenBtn = document.createElement('button');
        listenBtn.textContent = '🔊 Listen';
        listenBtn.style.cssText = 'background:#0f3460;color:#e2e8f0;border:1px solid #1a3a6e;padding:0.45rem 1rem;border-radius:0.5rem;cursor:pointer;font-size:0.85rem;';
        listenBtn.onclick = () => speakPassage(p.content);
        actions.appendChild(listenBtn);

        const delBtn = document.createElement('button');
        delBtn.textContent = '🗑 Delete';
        delBtn.style.cssText = 'background:none;border:1px solid #7f1d1d;color:#ef4444;padding:0.45rem 1rem;border-radius:0.5rem;cursor:pointer;font-size:0.85rem;';
        delBtn.onclick = async () => {
            const r = await fetch('/tutor/passages/' + id, { method: 'DELETE', headers: tutorAuthHeaders() });
            if (r.ok) { content.innerHTML = ''; loadPassages(); tutorShowToast('Passage deleted'); }
        };
        actions.appendChild(delBtn);
        content.appendChild(actions);
    } catch (e) { tutorShowToast('Error: ' + e.message); }
}

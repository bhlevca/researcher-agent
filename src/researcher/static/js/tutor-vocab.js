// ══════════════════════════════════════════════
// ── Vocabulary management ──
// ══════════════════════════════════════════════

let _vocabFilterTimer = null;

function _reviewDueLabel(nextReview) {
    if (!nextReview) return '';
    const today = new Date().toISOString().slice(0, 10);
    if (nextReview <= today) return '<span style="color:#f59e0b;font-size:0.68rem;" title="Due for review today">⚡ Due</span>';
    const diff = Math.round((new Date(nextReview) - new Date(today)) / 86400000);
    if (diff === 1) return '<span style="color:#64748b;font-size:0.68rem;">Tomorrow</span>';
    return '<span style="color:#64748b;font-size:0.68rem;">in ' + diff + 'd</span>';
}

async function loadVocabularyDueCount() {
    const lang = document.getElementById('tutor-target-lang')?.value;
    if (!lang) return;
    try {
        const resp = await fetch('/tutor/vocabulary/due?lang=' + encodeURIComponent(lang), { headers: tutorAuthHeaders() });
        if (!resp.ok) return;
        const data = await resp.json();
        const count = data.count || 0;
        const btn = document.getElementById('review-due-btn');
        const countEl = document.getElementById('review-due-count');
        if (btn && countEl) {
            countEl.textContent = count;
            btn.style.display = count > 0 ? '' : 'none';
        }
    } catch (e) { /* ignore */ }
}

async function loadVocabulary() {
    const lang = document.getElementById('tutor-target-lang')?.value;
    if (!lang) return;
    const container = document.getElementById('vocab-list');
    const countEl = document.getElementById('vocab-count');
    try {
        const resp = await fetch('/tutor/vocabulary?lang=' + encodeURIComponent(lang), { headers: tutorAuthHeaders() });
        const data = await resp.json();
        const items = data.vocabulary || [];
        countEl.textContent = items.length + ' words';

        if (items.length === 0) {
            container.innerHTML = '<div style="color:#64748b;text-align:center;padding:2rem;">No vocabulary saved yet. Chat with the tutor or generate lessons to build your vocabulary.</div>';
            loadVocabularyDueCount();
            return;
        }

        const filter = (document.getElementById('vocab-search')?.value || '').toLowerCase();
        const filtered = filter ? items.filter(v =>
            v.word.toLowerCase().includes(filter) ||
            (v.translation || '').toLowerCase().includes(filter)
        ) : items;

        const today = new Date().toISOString().slice(0, 10);
        let html = '<table class="vocab-table"><thead><tr>' +
            '<th>Word</th><th>Translation</th><th>Mastery</th><th>Reviewed</th><th>Next Review</th><th></th>' +
            '</tr></thead><tbody>';
        filtered.forEach(v => {
            const ml = Math.min(5, Math.max(0, v.mastery_level || 0));
            const due = (v.next_review && v.next_review <= today);
            html += '<tr' + (due ? ' style="background:rgba(245,158,11,0.06);"' : '') + '>' +
                '<td style="color:#e0e0e0;font-weight:500;">' + escapeHtml(v.word) + '</td>' +
                '<td>' + escapeHtml(v.translation || '') + '</td>' +
                '<td><span class="mastery mastery-' + ml + '">' + ml + '/5</span></td>' +
                '<td style="font-size:0.72rem;color:#64748b;">' + (v.times_reviewed || 0) + 'x</td>' +
                '<td style="font-size:0.72rem;">' + _reviewDueLabel(v.next_review) + '</td>' +
                '<td><button onclick="deleteVocab(\'' + v.id + '\')" style="background:none;border:none;color:#64748b;cursor:pointer;font-size:0.7rem;" title="Delete">✕</button></td>' +
                '</tr>';
        });
        html += '</tbody></table>';
        container.innerHTML = html;

        // Update due count badge
        loadVocabularyDueCount();
    } catch (e) {
        container.innerHTML = '<div style="color:#ef4444;text-align:center;">Error loading vocabulary</div>';
    }
}

function filterVocabulary() {
    clearTimeout(_vocabFilterTimer);
    _vocabFilterTimer = setTimeout(loadVocabulary, 300);
}

async function addVocabManual() {
    if (!tutorSessionId) { tutorShowToast('Create or load a session first'); return; }
    const word = prompt('Word or phrase:');
    if (!word) return;
    const translation = prompt('Translation:');
    const lang = document.getElementById('tutor-target-lang')?.value || '';
    try {
        const resp = await fetch('/tutor/vocabulary', {
            method: 'POST',
            headers: tutorJsonHeaders(),
            body: JSON.stringify({
                session_id: tutorSessionId,
                word, translation: translation || '',
                target_lang: lang,
            }),
        });
        if (resp.ok) {
            tutorShowToast('Added: ' + word);
            loadVocabulary();
        } else {
            tutorShowToast('Failed to add');
        }
    } catch (e) { tutorShowToast('Error: ' + e.message); }
}

async function deleteVocab(id) {
    try {
        const resp = await fetch('/tutor/vocabulary/' + id, {
            method: 'DELETE',
            headers: tutorAuthHeaders(),
        });
        if (resp.ok) loadVocabulary();
    } catch (e) { tutorShowToast('Error deleting word'); }
}

async function startDueReview() {
    const lang = document.getElementById('tutor-target-lang')?.value;
    if (!lang) { tutorShowToast('Select a target language first'); return; }

    try {
        const resp = await fetch('/tutor/vocabulary/due?lang=' + encodeURIComponent(lang), { headers: tutorAuthHeaders() });
        if (!resp.ok) { tutorShowToast('Could not load due words'); return; }
        const data = await resp.json();
        const due = data.due || [];
        if (due.length === 0) { tutorShowToast('No words due for review today!'); return; }

        // Switch to quiz tab and pre-fill a translation quiz seeded with due words
        switchTutorTab('quiz');

        // Build a simple flash-card review directly in the quiz panel
        _renderDueFlashcards(due);
    } catch (e) { tutorShowToast('Error: ' + e.message); }
}

function _renderDueFlashcards(words) {
    const container = document.getElementById('quiz-content');
    if (!container) return;

    let idx = 0;
    let correct = 0;
    const total = words.length;

    function showCard() {
        if (idx >= total) {
            container.innerHTML =
                '<div style="text-align:center;padding:2rem;">' +
                '<div style="font-size:2rem;margin-bottom:0.5rem;">🎉</div>' +
                '<div style="color:#e0e0e0;font-size:1.1rem;">Review complete!</div>' +
                '<div style="color:#64748b;margin-top:0.5rem;">' + correct + ' / ' + total + ' correct</div>' +
                '<button onclick="loadVocabulary();switchTutorTab(\'vocab\')" style="margin-top:1rem;">Back to Vocabulary</button>' +
                '</div>';
            // Hide submit bar
            const bar = document.getElementById('quiz-submit-bar');
            if (bar) bar.style.display = 'none';
            return;
        }
        const v = words[idx];
        container.innerHTML =
            '<div style="text-align:center;padding:1.5rem;">' +
            '<div style="color:#94a3b8;font-size:0.75rem;margin-bottom:0.5rem;">Word ' + (idx + 1) + ' of ' + total + '</div>' +
            '<div style="color:#38bdf8;font-size:1.6rem;font-weight:600;margin-bottom:1rem;">' + escapeHtml(v.word) + '</div>' +
            (v.phonetic ? '<div style="color:#64748b;font-size:0.85rem;margin-bottom:0.75rem;">' + escapeHtml(v.phonetic) + '</div>' : '') +
            '<input type="text" id="fc-answer" placeholder="Type the translation..." ' +
            'style="width:100%;max-width:320px;margin-bottom:0.75rem;padding:0.5rem;background:#1a1a2e;color:#e0e0e0;border:1px solid #0f3460;border-radius:0.4rem;" ' +
            'onkeydown="if(event.key===\'Enter\')checkFlashcard(' + JSON.stringify(v) + ')">' +
            '<br>' +
            '<button onclick="checkFlashcard(' + JSON.stringify(v) + ')">Check</button>' +
            '<button onclick="skipFlashcard()" style="margin-left:0.5rem;background:#0f3460;color:#94a3b8;">Skip</button>' +
            '</div>';
        setTimeout(() => document.getElementById('fc-answer')?.focus(), 50);
    }

    window.checkFlashcard = function(v) {
        const answer = (document.getElementById('fc-answer')?.value || '').trim().toLowerCase();
        const expected = (v.translation || '').trim().toLowerCase();
        const ok = answer === expected || expected.split(/[,;/]/).map(s => s.trim()).includes(answer);
        if (ok) correct++;
        const color = ok ? '#22c55e' : '#ef4444';
        const icon = ok ? '✅' : '❌';
        const container = document.getElementById('quiz-content');
        container.innerHTML =
            '<div style="text-align:center;padding:1.5rem;">' +
            '<div style="color:#38bdf8;font-size:1.4rem;font-weight:600;margin-bottom:0.5rem;">' + escapeHtml(v.word) + '</div>' +
            '<div style="font-size:1.8rem;margin-bottom:0.5rem;">' + icon + '</div>' +
            '<div style="color:' + color + ';font-size:1.1rem;margin-bottom:0.25rem;">' + (ok ? 'Correct!' : 'Incorrect') + '</div>' +
            '<div style="color:#e0e0e0;margin-bottom:0.25rem;">Answer: <strong>' + escapeHtml(v.translation) + '</strong></div>' +
            (v.context ? '<div style="color:#64748b;font-size:0.8rem;margin-bottom:0.75rem;">' + escapeHtml(v.context) + '</div>' : '') +
            '<button onclick="nextFlashcard()" style="margin-top:0.75rem;">Next →</button>' +
            '</div>';
        idx++;
    };

    window.skipFlashcard = function() { idx++; showCard(); };
    window.nextFlashcard = function() { showCard(); };

    const bar = document.getElementById('quiz-submit-bar');
    if (bar) bar.style.display = 'none';

    showCard();
}

// ══════════════════════════════════════════════
// ── Vocabulary management ──
// ══════════════════════════════════════════════

let _vocabFilterTimer = null;

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
            return;
        }

        const filter = (document.getElementById('vocab-search')?.value || '').toLowerCase();
        const filtered = filter ? items.filter(v =>
            v.word.toLowerCase().includes(filter) ||
            (v.translation || '').toLowerCase().includes(filter)
        ) : items;

        let html = '<table class="vocab-table"><thead><tr>' +
            '<th>Word</th><th>Translation</th><th>Mastery</th><th>Reviewed</th><th></th>' +
            '</tr></thead><tbody>';
        filtered.forEach(v => {
            const ml = Math.min(5, Math.max(0, v.mastery_level || 0));
            html += '<tr>' +
                '<td style="color:#e0e0e0;font-weight:500;">' + escapeHtml(v.word) + '</td>' +
                '<td>' + escapeHtml(v.translation || '') + '</td>' +
                '<td><span class="mastery mastery-' + ml + '">' + ml + '/5</span></td>' +
                '<td style="font-size:0.72rem;color:#64748b;">' + (v.times_reviewed || 0) + ' times</td>' +
                '<td><button onclick="deleteVocab(\'' + v.id + '\')" style="background:none;border:none;color:#64748b;cursor:pointer;font-size:0.7rem;" title="Delete">✕</button></td>' +
                '</tr>';
        });
        html += '</tbody></table>';
        container.innerHTML = html;
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

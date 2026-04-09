// ══════════════════════════════════════════════
// ── Stats & Appraisal ──
// ══════════════════════════════════════════════

async function loadStats() {
    if (!tutorSessionId) return;
    const grid = document.getElementById('stats-grid');
    try {
        const resp = await fetch('/tutor/sessions/' + tutorSessionId + '/stats', { headers: tutorAuthHeaders() });
        if (!resp.ok) { grid.innerHTML = ''; return; }
        const data = await resp.json();
        grid.innerHTML = '';

        const stats = [
            { label: 'Messages', value: data.message_count || 0 },
            { label: 'Vocabulary', value: data.vocabulary_count || 0 },
            { label: 'Vocab Accuracy', value: data.vocabulary_accuracy ? Math.round(data.vocabulary_accuracy * 100) + '%' : '—' },
            { label: 'Lessons', value: data.lesson_count || 0 },
            { label: 'Quizzes', value: data.quiz_count || 0 },
            { label: 'Avg Quiz Score', value: data.avg_quiz_score ? Math.round(data.avg_quiz_score * 100) + '%' : '—' },
        ];
        stats.forEach(s => {
            const card = document.createElement('div');
            card.className = 'stat-card';
            card.innerHTML = '<div class="stat-value">' + s.value + '</div><div class="stat-label">' + s.label + '</div>';
            grid.appendChild(card);
        });
    } catch (e) { /* ignore */ }
}

async function generateAppraisal() {
    if (!tutorSessionId) { tutorShowToast('Create or load a session first'); return; }

    const cfg = getTutorConfig();
    const content = document.getElementById('appraisal-content');
    content.innerHTML = '';

    const liveWrapper = document.createElement('div');
    const label = document.createElement('div');
    label.className = 'tutor-streaming-label';
    label.innerHTML = '<span class="pulse"></span> Generating appraisal...';
    liveWrapper.appendChild(label);
    const liveBox = document.createElement('div');
    liveBox.className = 'tutor-streaming';
    liveWrapper.appendChild(liveBox);
    content.appendChild(liveWrapper);

    let reasoningLines = [];
    try {
        const resp = await fetch('/tutor/appraisal', {
            method: 'POST',
            headers: tutorJsonHeaders(),
            body: JSON.stringify({
                session_id: tutorSessionId,
                target_language: cfg.target_lang,
                native_language: cfg.native_lang,
                level: cfg.level,
            }),
        });

        if (!resp.ok) {
            liveWrapper.remove();
            tutorShowToast('Failed to generate appraisal');
            return;
        }

        const result = await processTutorSSE(resp, liveBox, reasoningLines);
        liveWrapper.remove();

        if (result && result.response) {
            const viewer = document.createElement('div');
            viewer.className = 'appraisal-viewer';
            viewer.innerHTML = renderContent(result.response);
            content.appendChild(viewer);
        }
    } catch (e) {
        liveWrapper.remove();
        tutorShowToast('Error: ' + e.message);
    }
}

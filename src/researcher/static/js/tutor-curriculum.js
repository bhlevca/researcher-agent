// ══════════════════════════════════════════════
// ── Curriculum Progress Bar ──
// ══════════════════════════════════════════════

async function loadCurriculumProgress() {
    if (!tutorSessionId) return;
    const bar = document.getElementById('curriculum-bar');
    if (!bar) return;
    try {
        const resp = await fetch('/tutor/sessions/' + tutorSessionId + '/curriculum-progress',
            { headers: tutorAuthHeaders() });
        if (!resp.ok) { bar.style.display = 'none'; return; }
        const data = await resp.json();
        bar.style.display = 'block';
        const topicEl = document.getElementById('curriculum-topic');
        const fillEl = document.getElementById('curriculum-fill');
        const pctEl = document.getElementById('curriculum-pct');
        if (topicEl) topicEl.textContent = data.level + ': ' + (data.current_topic || 'Complete');
        if (fillEl) fillEl.style.width = (data.pct || 0) + '%';
        if (pctEl) pctEl.textContent = (data.topic_index || 0) + '/' + (data.total_topics || 0);
    } catch (e) {
        const bar = document.getElementById('curriculum-bar');
        if (bar) bar.style.display = 'none';
    }
}

async function advanceCurriculumTopic() {
    if (!tutorSessionId) return;
    const btn = document.getElementById('curriculum-advance-btn');
    if (btn) btn.disabled = true;
    try {
        const resp = await fetch('/tutor/sessions/' + tutorSessionId + '/advance-topic', {
            method: 'POST',
            headers: tutorJsonHeaders(),
        });
        if (resp.ok) {
            await loadCurriculumProgress();
        }
    } catch (e) { /* ignore */ }
    finally { if (btn) btn.disabled = false; }
}

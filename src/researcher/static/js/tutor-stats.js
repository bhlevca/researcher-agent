// ══════════════════════════════════════════════
// ── Stats & Appraisal ──
// ══════════════════════════════════════════════

let _chartMastery = null;
let _chartQuiz = null;

function _destroyCharts() {
    if (_chartMastery) { _chartMastery.destroy(); _chartMastery = null; }
    if (_chartQuiz) { _chartQuiz.destroy(); _chartQuiz = null; }
}

function _buildMasteryChart(dist) {
    const labels = ['0 – New', '1', '2', '3', '4', '5 – Mastered'];
    const values = labels.map((_, i) => dist[String(i)] || 0);
    const colors = ['#64748b','#f59e0b','#f97316','#22c55e','#16a34a','#15803d'];
    const ctx = document.getElementById('chart-mastery');
    if (!ctx) return;
    _chartMastery = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{ data: values, backgroundColor: colors, borderRadius: 4 }],
        },
        options: {
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, ticks: { stepSize: 1, color: '#94a3b8' }, grid: { color: '#1e293b' } },
                x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { display: false } },
            },
        },
    });
}

function _buildQuizChart(history) {
    if (!history || !history.length) return;
    const labels = history.map(h => h.date);
    const values = history.map(h => h.pct);
    const ctx = document.getElementById('chart-quiz');
    if (!ctx) return;
    _chartQuiz = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                data: values,
                borderColor: '#e94560',
                backgroundColor: 'rgba(233,69,96,0.15)',
                tension: 0.3,
                fill: true,
                pointRadius: 4,
                pointBackgroundColor: '#e94560',
            }],
        },
        options: {
            plugins: { legend: { display: false } },
            scales: {
                y: {
                    min: 0, max: 100,
                    ticks: { color: '#94a3b8', callback: v => v + '%' },
                    grid: { color: '#1e293b' },
                },
                x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { display: false } },
            },
        },
    });
}

async function loadStats() {
    if (!tutorSessionId) return;
    const grid = document.getElementById('stats-grid');
    _destroyCharts();
    try {
        const resp = await fetch('/tutor/sessions/' + tutorSessionId + '/stats', { headers: tutorAuthHeaders() });
        if (!resp.ok) { grid.innerHTML = ''; return; }
        const data = await resp.json();
        grid.innerHTML = '';

        const vocab = data.vocabulary || {};
        const quizzes = data.quizzes || {};
        const srsCount = data.srs_due_count ?? '—';

        const stats = [
            { label: 'Vocabulary', value: vocab.total_words || 0 },
            { label: 'Avg Mastery', value: vocab.avg_mastery !== undefined ? vocab.avg_mastery.toFixed(1) + ' / 5' : '—' },
            { label: 'Accuracy', value: vocab.accuracy !== undefined ? vocab.accuracy + '%' : '—' },
            { label: 'Lessons', value: data.lessons_completed || 0 },
            { label: 'Quizzes', value: quizzes.total_taken || 0 },
            { label: 'Avg Quiz', value: quizzes.avg_score !== undefined ? quizzes.avg_score + '%' : '—' },
            { label: 'SRS Due', value: srsCount },
            { label: 'Messages', value: data.total_interactions || 0 },
        ];
        stats.forEach(s => {
            const card = document.createElement('div');
            card.className = 'stat-card';
            card.innerHTML = '<div class="stat-value">' + s.value + '</div><div class="stat-label">' + s.label + '</div>';
            grid.appendChild(card);
        });

        if (data.mastery_distribution) _buildMasteryChart(data.mastery_distribution);
        if (data.quiz_history) _buildQuizChart(data.quiz_history);
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

        const result = await processTutorSSE(resp, liveBox, []);
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

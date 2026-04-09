// ══════════════════════════════════════════════
// ── Quiz generation, answering & grading ──
// ══════════════════════════════════════════════

// Parse quiz JSON from raw LLM response (handles fenced code blocks or bare JSON)
function tryParseQuizJson(text) {
    if (!text) return null;
    // Try fenced code block first
    const fenced = text.match(/```(?:json)?\s*\n?([\s\S]*?)```/);
    const raw = fenced ? fenced[1].trim() : text.trim();
    try {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed) && parsed.length > 0 && parsed[0].question) return parsed;
        if (parsed.questions && Array.isArray(parsed.questions)) return parsed.questions;
    } catch (e) { /* not valid JSON */ }
    // Try to find a JSON array in the text
    const arrMatch = text.match(/\[[\s\S]*\]/);
    if (arrMatch) {
        try {
            const arr = JSON.parse(arrMatch[0]);
            if (Array.isArray(arr) && arr.length > 0 && arr[0].question) return arr;
        } catch (e) { /* ignore */ }
    }
    return null;
}

async function generateQuiz() {
    if (!tutorSessionId) { tutorShowToast('Create or load a session first'); return; }

    const quizType = document.getElementById('quiz-type').value;
    const numQ = parseInt(document.getElementById('quiz-num').value, 10) || 5;
    const cfg = getTutorConfig();

    const content = document.getElementById('quiz-content');
    content.innerHTML = '';
    document.getElementById('quiz-submit-bar').style.display = 'none';

    // Show streaming
    const liveWrapper = document.createElement('div');
    const label = document.createElement('div');
    label.className = 'tutor-streaming-label';
    label.innerHTML = '<span class="pulse"></span> Generating quiz...';
    liveWrapper.appendChild(label);
    const liveBox = document.createElement('div');
    liveBox.className = 'tutor-streaming';
    liveWrapper.appendChild(liveBox);
    content.appendChild(liveWrapper);

    let reasoningLines = [];
    try {
        const resp = await fetch('/tutor/quiz/generate', {
            method: 'POST',
            headers: tutorJsonHeaders(),
            body: JSON.stringify({
                session_id: tutorSessionId,
                quiz_type: quizType,
                num_questions: numQ,
                target_language: cfg.target_lang,
                native_language: cfg.native_lang,
                level: cfg.level,
            }),
        });

        if (!resp.ok) {
            liveWrapper.remove();
            tutorShowToast('Failed to generate quiz');
            return;
        }

        const result = await processTutorSSE(resp, liveBox, reasoningLines);
        liveWrapper.remove();

        if (result && result.response) {
            tutorQuizRawResponse = result.response;

            // Try to parse questions client-side first
            let questions = tryParseQuizJson(result.response);

            // Try to save the quiz and get parsed questions from server
            try {
                const saveResp = await fetch('/tutor/quiz/save', {
                    method: 'POST',
                    headers: tutorJsonHeaders(),
                    body: JSON.stringify({
                        session_id: tutorSessionId,
                        quiz_type: quizType,
                        response: result.response,
                    }),
                });

                if (saveResp.ok) {
                    const saved = await saveResp.json();
                    if (saved.questions && saved.questions.length > 0) {
                        questions = saved.questions;
                        tutorQuizData = { id: saved.id, questions };
                    }
                }
            } catch (e) { /* save failed, use client-parsed questions */ }

            if (questions && questions.length > 0) {
                if (!tutorQuizData) tutorQuizData = { id: 'local', questions };
                renderQuizQuestions(questions);
                document.getElementById('quiz-submit-bar').style.display = 'flex';
                return;
            }
            // Fallback: show formatted response
            const viewer = document.createElement('div');
            viewer.className = 'lesson-viewer';
            viewer.innerHTML = renderContent(result.response);
            content.appendChild(viewer);
        }
    } catch (e) {
        liveWrapper.remove();
        tutorShowToast('Error: ' + e.message);
    }
}

function renderQuizQuestions(questions) {
    const content = document.getElementById('quiz-content');
    content.innerHTML = '';

    questions.forEach((q, i) => {
        const div = document.createElement('div');
        div.className = 'quiz-question';
        div.dataset.questionId = q.id || i;

        const num = document.createElement('div');
        num.className = 'quiz-question-num';
        num.textContent = 'Question ' + (i + 1) + ' of ' + questions.length;
        div.appendChild(num);

        const text = document.createElement('div');
        text.className = 'quiz-question-text';
        text.textContent = q.question;
        div.appendChild(text);

        if (q.options && Array.isArray(q.options) && q.options.length > 0) {
            q.options.forEach((opt, oi) => {
                const label = document.createElement('label');
                label.className = 'quiz-option';
                const radio = document.createElement('input');
                radio.type = 'radio';
                radio.name = 'quiz-q-' + i;
                radio.value = opt;
                radio.onchange = () => {
                    // Highlight selected
                    div.querySelectorAll('.quiz-option').forEach(o => o.classList.remove('selected'));
                    label.classList.add('selected');
                };
                label.appendChild(radio);
                label.appendChild(document.createTextNode(' ' + opt));
                div.appendChild(label);
            });
        } else {
            // Fill-in-the-blank or free text
            const inp = document.createElement('input');
            inp.type = 'text';
            inp.className = 'quiz-answer-input';
            inp.placeholder = 'Type your answer...';
            inp.name = 'quiz-q-' + i;
            div.appendChild(inp);
        }

        content.appendChild(div);
    });
}

async function submitQuiz() {
    if (!tutorQuizData || !tutorSessionId) return;

    const answers = [];
    tutorQuizData.questions.forEach((q, i) => {
        const name = 'quiz-q-' + i;
        const hasOptions = q.options && Array.isArray(q.options) && q.options.length > 0;
        if (hasOptions) {
            const selected = document.querySelector('input[name="' + name + '"]:checked');
            answers.push({
                question_id: typeof q.id === 'number' ? q.id : i,
                answer: selected ? selected.value : '',
            });
        } else {
            const inp = document.querySelector('input[name="' + name + '"]');
            answers.push({
                question_id: typeof q.id === 'number' ? q.id : i,
                answer: inp ? inp.value.trim() : '',
            });
        }
    });

    try {
        const resp = await fetch('/tutor/quiz/submit', {
            method: 'POST',
            headers: tutorJsonHeaders(),
            body: JSON.stringify({
                session_id: tutorSessionId,
                quiz_id: tutorQuizData.id,
                answers,
            }),
        });

        if (!resp.ok) {
            tutorShowToast('Failed to submit quiz');
            return;
        }

        const result = await resp.json();
        document.getElementById('quiz-submit-bar').style.display = 'none';

        // Show results
        showQuizResults(result, tutorQuizData.questions);
    } catch (e) { tutorShowToast('Error: ' + e.message); }
}

function showQuizResults(result, questions) {
    // Highlight correct/incorrect answers
    if (result.details) {
        result.details.forEach((d, i) => {
            const qDiv = document.querySelectorAll('.quiz-question')[i];
            if (!qDiv) return;
            if (d.is_correct) {
                // Mark correct
                const selected = qDiv.querySelector('.quiz-option.selected');
                if (selected) selected.classList.add('correct');
                // For text inputs, show checkmark
                const textInput = qDiv.querySelector('.quiz-answer-input');
                if (textInput) textInput.style.borderColor = '#22c55e';
            } else {
                // Mark incorrect
                const selected = qDiv.querySelector('.quiz-option.selected');
                if (selected) selected.classList.add('incorrect');
                // Highlight correct answer in multiple choice
                qDiv.querySelectorAll('.quiz-option').forEach(opt => {
                    if (opt.textContent.trim() === d.correct_answer) {
                        opt.classList.add('correct');
                    }
                });
                // For text inputs, show correct answer below
                const textInput = qDiv.querySelector('.quiz-answer-input');
                if (textInput) {
                    textInput.style.borderColor = '#ef4444';
                    const correction = document.createElement('div');
                    correction.style.cssText = 'color:#22c55e;font-size:0.82rem;margin-top:0.3rem;';
                    correction.textContent = '✓ Correct answer: ' + d.correct_answer;
                    textInput.parentNode.insertBefore(correction, textInput.nextSibling);
                }
            }
        });
    }

    // Score summary
    const content = document.getElementById('quiz-content');
    const summary = document.createElement('div');
    summary.className = 'quiz-results';
    summary.innerHTML = '<div class="quiz-score">' + result.score + '/' + result.total +
        ' (' + Math.round((result.score / Math.max(1, result.total)) * 100) + '%)</div>' +
        '<div style="text-align:center;color:#94a3b8;font-size:0.82rem;">Quiz completed!</div>';
    content.appendChild(summary);
}

async function loadQuizHistory() {
    if (!tutorSessionId) return;
    const content = document.getElementById('quiz-history');
    try {
        const resp = await fetch('/tutor/sessions/' + tutorSessionId + '/quizzes', { headers: tutorAuthHeaders() });
        const data = await resp.json();
        if (!data.quizzes || data.quizzes.length === 0) {
            content.innerHTML = '<div style="color:#64748b;font-size:0.78rem;">No quizzes taken yet</div>';
            return;
        }
        content.innerHTML = '<div style="font-size:0.78rem;color:#64748b;margin-bottom:0.5rem;">Past Quizzes</div>';
        data.quizzes.forEach(q => {
            const card = document.createElement('div');
            card.className = 'quiz-card';
            card.innerHTML = '<div style="font-size:0.82rem;color:#e0e0e0;">' + (q.quiz_type || 'Quiz') +
                (q.score !== undefined ? ' — ' + q.score + '/' + q.total : '') + '</div>' +
                '<div class="quiz-card-meta">' + new Date(q.created_at).toLocaleString() + '</div>';
            content.appendChild(card);
        });
    } catch (e) { /* ignore */ }
}

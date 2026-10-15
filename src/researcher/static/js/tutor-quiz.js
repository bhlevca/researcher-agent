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
        collapseThinking(liveWrapper, liveBox, 'Thinking — quiz');

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
                        tutorQuizData = { id: saved.id, quizType, questions };
                    }
                }
            } catch (e) { /* save failed, use client-parsed questions */ }

            if (questions && questions.length > 0) {
                if (!tutorQuizData) tutorQuizData = { id: 'local', quizType, questions };
                renderQuizQuestions(questions, quizType);
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

// ─────────────────────────────────────────────
// Render dispatcher
// ─────────────────────────────────────────────

// Map requested quiz_type → the canonical question type it produces
const _QUIZ_TYPE_FORCED = {
    'reorder':       'reorder',
    'matching':      'matching',
    'cloze':         'cloze',
    'fill_blank':    'cloze',
    'listening':     'listening',
    'pronunciation': 'pronunciation',
};

function renderQuizQuestions(questions, requestedQuizType) {
    const content = document.getElementById('quiz-content');
    content.innerHTML = '';
    const forcedType = _QUIZ_TYPE_FORCED[requestedQuizType] || null;

    questions.forEach((q, i) => {
        const div = document.createElement('div');
        div.className = 'quiz-question';
        div.dataset.questionId = q.id !== undefined ? q.id : i;

        // If server coercion failed (e.g. save errored), force the right type client-side
        const effectiveType = forcedType || q.type || 'vocabulary';
        // Patch q.type so submit logic and grading use the right type
        q.type = effectiveType;
        div.dataset.questionType = effectiveType;

        const num = document.createElement('div');
        num.className = 'quiz-question-num';
        num.textContent = 'Question ' + (i + 1) + ' of ' + questions.length
            + ' · ' + _quizTypeLabel(effectiveType);
        div.appendChild(num);

        const text = document.createElement('div');
        text.className = 'quiz-question-text';
        div.appendChild(text);

        switch (effectiveType) {
            case 'matching':      _renderMatching(q, i, div, text); break;
            case 'reorder':       _renderReorder(q, i, div, text); break;
            case 'listening':     _renderListening(q, i, div, text); break;
            case 'cloze':         _renderCloze(q, i, div, text); break;
            case 'pronunciation': _renderPronunciation(q, i, div, text); break;
            default:              _renderStandard(q, i, div, text); break;
        }

        content.appendChild(div);
    });
}

// ─────────────────────────────────────────────
// Pronunciation: Speak button + score display
// ─────────────────────────────────────────────

function _renderPronunciation(q, i, div, text) {
    // Recover sentence from target_sentence, correct_answer, or the question field itself
    let target = q.target_sentence || q.correct_answer || '';
    if (!target) {
        // LLM sometimes puts everything in q.question: "Read aloud: Bonjour!"
        const m = (q.question || '').match(/:\s*(.+)$/);
        if (m) target = m[1].trim();
    }

    // Build the question text safely using DOM (avoids innerHTML parsing surprises)
    text.textContent = (q.question || 'Read aloud:') + ' ';
    if (target) {
        const sentence = document.createElement('strong');
        sentence.style.cssText = 'font-size:1.1em;color:#e94560;display:inline;';
        sentence.textContent = target;
        text.appendChild(sentence);
    }

    // Hidden input stores last score as the "answer"
    const hidden = document.createElement('input');
    hidden.type = 'hidden';
    hidden.name = 'quiz-q-' + i;
    hidden.value = '';
    div.appendChild(hidden);

    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:0.6rem;align-items:center;margin-top:0.6rem;flex-wrap:wrap;';

    // 🔊 Play button — let student hear the correct pronunciation first
    const playBtn = document.createElement('button');
    playBtn.type = 'button';
    playBtn.textContent = '🔊 Hear it';
    playBtn.title = 'Listen to the correct pronunciation';
    playBtn.style.cssText = 'background:#0f3460;color:#e0e0e0;border:1px solid #1a3a6e;padding:0.3rem 0.7rem;border-radius:0.4rem;font-size:0.82rem;cursor:pointer;';
    playBtn.onclick = async () => {
        if (!target) return;
        // Use correct-language TTS voice, not the UI's selected English voice
        const cfg = typeof getTutorConfig === 'function' ? getTutorConfig() : {};
        const langCode = cfg.target_lang_code || 'fr';
        // Find best voice for the target language from loaded voices
        const voice = (typeof ttsVoices !== 'undefined' && ttsVoices.length)
            ? (ttsVoices.find(v => v.locale.toLowerCase().startsWith(langCode)) || ttsVoices[0]).name
            : null;
        if (voice && typeof tutorAuthHeaders === 'function') {
            try {
                const resp = await fetch('/tts/speak', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', ...tutorAuthHeaders() },
                    body: JSON.stringify({ text: target, voice }),
                });
                if (resp.ok) {
                    const blob = await resp.blob();
                    const url = URL.createObjectURL(blob);
                    const audio = new Audio(url);
                    audio.onended = () => URL.revokeObjectURL(url);
                    audio.play();
                    return;
                }
            } catch (_) { /* fall through to browser TTS */ }
        }
        const utt = new SpeechSynthesisUtterance(target);
        window.speechSynthesis.speak(utt);
    };

    const micBtn = document.createElement('button');
    micBtn.type = 'button';
    micBtn.textContent = '🎤 Speak';
    micBtn.title = 'Record your pronunciation';
    micBtn.style.cssText = 'background:#0f3460;color:#e0e0e0;border:1px solid #1a3a6e;padding:0.3rem 0.7rem;border-radius:0.4rem;font-size:0.82rem;cursor:pointer;';

    const feedback = document.createElement('span');
    feedback.style.cssText = 'font-size:0.8rem;color:#94a3b8;';
    feedback.textContent = 'Press 🔊 to hear, then 🎤 to record';

    if (target) row.appendChild(playBtn);
    row.appendChild(micBtn);
    row.appendChild(feedback);
    div.appendChild(row);

    async function startRecording() {
        const cfg = typeof getTutorConfig === 'function' ? getTutorConfig() : {};
        const lang = cfg.target_lang_code || 'fr';

        micBtn.disabled = true;
        micBtn.textContent = '🔴 Recording…';
        feedback.textContent = 'Speak now… (auto-stops in 10s)';

        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        } catch (e) {
            feedback.textContent = 'Microphone error: ' + e.message;
            micBtn.disabled = false;
            micBtn.textContent = '🎤 Speak';
            return;
        }

        const chunks = [];
        const rec = new MediaRecorder(stream);
        rec.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };

        micBtn.textContent = '⏹ Stop';
        micBtn.disabled = false;
        micBtn.onclick = () => rec.stop();

        rec.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            micBtn.disabled = true;
            micBtn.textContent = '⏳ Scoring…';
            feedback.textContent = 'Processing…';

            const blob = new Blob(chunks, { type: rec.mimeType });
            const form = new FormData();
            form.append('file', blob, 'pronunciation.webm');
            form.append('expected', target);
            form.append('language', lang);
            try {
                const resp = await fetch('/tutor/pronunciation/score',
                    { method: 'POST', body: form, headers: tutorAuthHeaders() });
                if (resp.ok) {
                    const data = await resp.json();
                    const pct = Math.round(data.score * 100);
                    const color = pct >= 90 ? '#22c55e' : pct >= 60 ? '#f59e0b' : '#e94560';
                    feedback.innerHTML =
                        'You said: <em>' + escapeHtml(data.transcript || '(silence)') + '</em> — ' +
                        '<strong style="color:' + color + ';">' + pct + '%</strong> — ' +
                        escapeHtml(data.feedback || '');
                    hidden.value = String(data.score);
                } else {
                    feedback.textContent = 'Scoring failed — try again.';
                }
            } catch (err) {
                feedback.textContent = 'Error: ' + err.message;
            }
            // Reset button so user can try again
            micBtn.disabled = false;
            micBtn.textContent = '🎤 Try again';
            micBtn.onclick = startRecording;
        };

        rec.start();
        const autoStop = setTimeout(() => { if (rec.state === 'recording') rec.stop(); }, 10000);
        rec.addEventListener('stop', () => clearTimeout(autoStop), { once: true });
    }

    micBtn.onclick = startRecording;
}

function _quizTypeLabel(type) {
    const labels = {
        vocabulary: 'Vocabulary', grammar: 'Grammar', translation: 'Translation',
        matching: 'Matching', reorder: 'Reorder', listening: 'Listening', cloze: 'Cloze',
        pronunciation: 'Pronunciation',
    };
    return labels[type] || type || '';
}

// ─────────────────────────────────────────────
// Standard: multiple choice or text input
// ─────────────────────────────────────────────

function _renderStandard(q, i, div, text) {
    text.textContent = q.question;

    if (q.options && Array.isArray(q.options) && q.options.length > 0) {
        q.options.forEach(opt => {
            const label = document.createElement('label');
            label.className = 'quiz-option';
            const radio = document.createElement('input');
            radio.type = 'radio';
            radio.name = 'quiz-q-' + i;
            radio.value = opt;
            radio.onchange = () => {
                div.querySelectorAll('.quiz-option').forEach(o => o.classList.remove('selected'));
                label.classList.add('selected');
            };
            label.appendChild(radio);
            label.appendChild(document.createTextNode(' ' + opt));
            div.appendChild(label);
        });
    } else {
        const inp = document.createElement('input');
        inp.type = 'text';
        inp.className = 'quiz-answer-input';
        inp.placeholder = 'Type your answer...';
        inp.name = 'quiz-q-' + i;
        div.appendChild(inp);
    }
}

// ─────────────────────────────────────────────
// Matching
// ─────────────────────────────────────────────

// Normalise a single pair to {left, right} regardless of what the LLM produced.
function _normalisePair(p) {
    if (!p) return null;
    // String: "word=translation" or "word: translation" or "word - translation"
    if (typeof p === 'string') {
        const m = p.match(/^(.+?)\s*[=:\-]\s*(.+)$/);
        if (m) return { left: m[1].trim(), right: m[2].trim() };
        return null;
    }
    if (typeof p !== 'object') return null;
    // Object with known key names
    if (p.left  !== undefined) return { left: String(p.left),  right: String(p.right  ?? '') };
    if (p.term  !== undefined) return { left: String(p.term),  right: String(p.definition ?? p.translation ?? '') };
    if (p.word  !== undefined) return { left: String(p.word),  right: String(p.translation ?? p.meaning ?? '') };
    // Arbitrary two-key object
    const keys = Object.keys(p);
    if (keys.length >= 2) return { left: String(p[keys[0]]), right: String(p[keys[1]]) };
    return null;
}

function _renderMatching(q, i, div, text) {
    text.textContent = q.question || 'Match each word to its translation.';

    const rawPairs = q.pairs || [];
    const pairs = rawPairs.map(_normalisePair).filter(Boolean);
    if (!pairs.length) { _renderStandard(q, i, div, text); return; }

    // Shuffle right-side options
    const rights = _shuffle(pairs.map(p => p.right));

    const grid = document.createElement('div');
    grid.className = 'quiz-matching-grid';
    grid.dataset.pairCount = pairs.length;

    pairs.forEach((p, pi) => {
        const leftEl = document.createElement('div');
        leftEl.className = 'quiz-matching-left';
        leftEl.textContent = p.left;

        const sel = document.createElement('select');
        sel.className = 'quiz-matching-select';
        sel.name = 'quiz-match-' + i + '-' + pi;
        sel.dataset.correctAnswer = p.right;

        const blank = document.createElement('option');
        blank.value = '';
        blank.textContent = '— select —';
        sel.appendChild(blank);
        rights.forEach(r => {
            const opt = document.createElement('option');
            opt.value = r;
            opt.textContent = r;
            sel.appendChild(opt);
        });

        grid.appendChild(leftEl);
        grid.appendChild(sel);
    });

    div.dataset.matchPairCount = pairs.length;
    div.appendChild(grid);

    // Hidden input to hold serialized answer
    const hidden = document.createElement('input');
    hidden.type = 'hidden';
    hidden.name = 'quiz-q-' + i;
    div.appendChild(hidden);
}

// ─────────────────────────────────────────────
// Reorder
// ─────────────────────────────────────────────

function _renderReorder(q, i, div, text) {
    // Show the hint (native-language cue), not a blank generic instruction
    text.textContent = q.question || 'Put the words in the correct order.';

    // Always derive the word pool from correct_answer to guarantee all
    // function words (articles, pronouns, prepositions) are present.
    // The LLM-provided words array is unreliable.
    const correctSrc = (q.correct_answer || '').trim();
    let rawWords = correctSrc ? correctSrc.split(/\s+/).filter(Boolean) : null;

    // Fallback only if correct_answer is empty
    if (!rawWords || rawWords.length === 0) {
        rawWords = q.words;
        if (typeof rawWords === 'string' && rawWords.trim()) {
            rawWords = rawWords.split(/\s+/).filter(Boolean);
        }
    }
    if (!Array.isArray(rawWords)) rawWords = [];
    if (rawWords.length === 0) {
        // Last resort: nothing usable — render a plain text input so it's not blank
        const inp = document.createElement('input');
        inp.type = 'text';
        inp.className = 'quiz-answer-input';
        inp.placeholder = 'Type the correct sentence...';
        inp.name = 'quiz-q-' + i;
        div.appendChild(inp);
        return;
    }
    const words = _shuffle(rawWords.slice());

    const ansLabel = document.createElement('div');
    ansLabel.className = 'quiz-reorder-answer-label';
    ansLabel.textContent = 'Your sentence:';
    div.appendChild(ansLabel);

    const answerArea = document.createElement('div');
    answerArea.className = 'quiz-reorder-answer';
    answerArea.dataset.quizIndex = i;
    div.appendChild(answerArea);

    const poolArea = document.createElement('div');
    poolArea.className = 'quiz-reorder-pool';
    div.appendChild(poolArea);

    // Hidden input for serialized answer
    const hidden = document.createElement('input');
    hidden.type = 'hidden';
    hidden.name = 'quiz-q-' + i;
    div.appendChild(hidden);

    words.forEach(word => {
        const chip = document.createElement('span');
        chip.className = 'quiz-word-chip';
        chip.textContent = word;
        chip.onclick = () => _reorderMoveToAnswer(chip, answerArea, poolArea, hidden, i);
        poolArea.appendChild(chip);
    });
}

function _reorderMoveToAnswer(chip, answerArea, poolArea, hidden, i) {
    chip.classList.add('in-answer');
    chip.onclick = () => _reorderMoveToPool(chip, answerArea, poolArea, hidden, i);
    answerArea.appendChild(chip);
    _updateReorderHidden(answerArea, hidden);
}

function _reorderMoveToPool(chip, answerArea, poolArea, hidden, i) {
    chip.classList.remove('in-answer');
    chip.onclick = () => _reorderMoveToAnswer(chip, answerArea, poolArea, hidden, i);
    poolArea.appendChild(chip);
    _updateReorderHidden(answerArea, hidden);
}

function _updateReorderHidden(answerArea, hidden) {
    const words = [...answerArea.querySelectorAll('.quiz-word-chip')].map(c => c.textContent);
    hidden.value = words.join(' ');
}

// ─────────────────────────────────────────────
// Listening
// ─────────────────────────────────────────────

function _renderListening(q, i, div, text) {
    text.textContent = q.question || 'Listen and write what you hear.';

    const audioText = q.audio_text || q.correct_answer || '';

    const btn = document.createElement('button');
    btn.className = 'quiz-listen-btn';
    btn.innerHTML = '🔊 Play';
    btn.onclick = () => {
        if (typeof speakText === 'function') {
            speakText(audioText);
        } else {
            // Fallback: browser TTS
            const utt = new SpeechSynthesisUtterance(audioText);
            window.speechSynthesis.speak(utt);
        }
    };
    div.appendChild(btn);

    const inp = document.createElement('input');
    inp.type = 'text';
    inp.className = 'quiz-answer-input';
    inp.placeholder = 'Type what you heard...';
    inp.name = 'quiz-q-' + i;
    div.appendChild(inp);
}

// ─────────────────────────────────────────────
// Cloze (gap fill)
// ─────────────────────────────────────────────

function _renderCloze(q, i, div, text) {
    const sentence = q.question || '';
    const parts = sentence.split('___');

    if (parts.length <= 1) {
        // No blank markers — fall back to text input
        text.textContent = sentence;
        const inp = document.createElement('input');
        inp.type = 'text';
        inp.className = 'quiz-answer-input';
        inp.placeholder = 'Type your answer...';
        inp.name = 'quiz-q-' + i;
        div.appendChild(inp);
        return;
    }

    const clozeSpan = document.createElement('span');
    clozeSpan.className = 'quiz-cloze-text';

    const inputs = [];
    parts.forEach((part, pi) => {
        clozeSpan.appendChild(document.createTextNode(part));
        if (pi < parts.length - 1) {
            const inp = document.createElement('input');
            inp.type = 'text';
            inp.className = 'quiz-cloze-input';
            inp.placeholder = '___';
            inp.dataset.blankIndex = pi;
            inputs.push(inp);
            clozeSpan.appendChild(inp);
        }
    });

    div.appendChild(clozeSpan);

    // Hidden input that collects all blanks joined with "|"
    const hidden = document.createElement('input');
    hidden.type = 'hidden';
    hidden.name = 'quiz-q-' + i;
    div.appendChild(hidden);

    const syncHidden = () => {
        hidden.value = inputs.map(inp => inp.value.trim()).join('|');
    };
    inputs.forEach(inp => inp.addEventListener('input', syncHidden));
}

// ─────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────

function _shuffle(arr) {
    for (let n = arr.length - 1; n > 0; n--) {
        const k = Math.floor(Math.random() * (n + 1));
        [arr[n], arr[k]] = [arr[k], arr[n]];
    }
    return arr;
}

// ─────────────────────────────────────────────
// Submit
// ─────────────────────────────────────────────

async function submitQuiz() {
    if (!tutorQuizData || !tutorSessionId) return;

    const answers = [];
    tutorQuizData.questions.forEach((q, i) => {
        const qid = typeof q.id === 'number' ? q.id : i;
        const name = 'quiz-q-' + i;
        const type = q.type || 'vocabulary';

        if (type === 'matching') {
            // Serialize all pair selections into hidden input first
            const div = document.querySelectorAll('.quiz-question')[i];
            if (div) {
                const grid = div.querySelector('.quiz-matching-grid');
                const hidden = div.querySelector('input[type=hidden][name="' + name + '"]');
                if (grid && hidden) {
                    const selects = grid.querySelectorAll('select');
                    const lefts = grid.querySelectorAll('.quiz-matching-left');
                    const parts = [];
                    selects.forEach((sel, si) => {
                        const left = lefts[si] ? lefts[si].textContent : '';
                        parts.push(left + '=' + (sel.value || ''));
                    });
                    hidden.value = parts.join('; ');
                }
            }
        }

        const inp = document.querySelector('[name="' + name + '"]');
        let answer = '';
        if (inp) {
            if (inp.type === 'radio') {
                const checked = document.querySelector('input[name="' + name + '"]:checked');
                answer = checked ? checked.value : '';
            } else {
                answer = inp.value ? inp.value.trim() : '';
            }
        } else {
            // Multiple-choice radio fallback
            const checked = document.querySelector('input[name="' + name + '"]:checked');
            if (checked) answer = checked.value;
        }

        answers.push({ question_id: qid, answer });
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
        showQuizResults(result, tutorQuizData.questions);
    } catch (e) { tutorShowToast('Error: ' + e.message); }
}

// ─────────────────────────────────────────────
// Results display
// ─────────────────────────────────────────────

function showQuizResults(result, questions) {
    const qDivs = document.querySelectorAll('.quiz-question');

    if (result.details) {
        result.details.forEach(d => {
            const idx = questions.findIndex(q =>
                (typeof q.id === 'number' ? q.id : questions.indexOf(q)) === d.question_id
            );
            const qDiv = qDivs[idx >= 0 ? idx : d.question_id];
            if (!qDiv) return;

            const score = typeof d.score === 'number' ? d.score : (d.is_correct ? 1 : 0);
            const type = qDiv.dataset.questionType || 'vocabulary';

            // Disable all inputs
            qDiv.querySelectorAll('input, select').forEach(el => { el.disabled = true; });

            // Visual feedback per type
            if (type === 'matching') {
                const grid = qDiv.querySelector('.quiz-matching-grid');
                if (grid) {
                    const selects = grid.querySelectorAll('select');
                    selects.forEach(sel => {
                        const isCorrect = sel.value === sel.dataset.correctAnswer;
                        sel.classList.add(isCorrect ? 'correct' : 'incorrect');
                    });
                }
            } else if (type === 'reorder') {
                const chips = qDiv.querySelectorAll('.quiz-word-chip.in-answer');
                chips.forEach(c => c.classList.add(d.is_correct ? 'correct' : 'incorrect'));
                if (!d.is_correct) _showCorrection(qDiv, d.correct_answer);
            } else if (type === 'cloze') {
                // Always trust the LLM/grader score — it knows valid alternatives.
                const inputs = qDiv.querySelectorAll('.quiz-cloze-input');
                if (inputs.length <= 1) {
                    if (inputs.length === 1) inputs[0].classList.add(d.is_correct ? 'correct' : 'incorrect');
                } else {
                    const expectedParts = (d.correct_answer || '').split('|');
                    inputs.forEach((inp, bi) => {
                        const expected = (expectedParts[bi] || '').trim().toLowerCase();
                        const ok = inp.value.trim().toLowerCase() === expected;
                        inp.classList.add(ok ? 'correct' : (d.is_correct ? 'correct' : 'incorrect'));
                    });
                }
                if (!d.is_correct) {
                    // Show the full reconstructed sentence, not just the missing word
                    const q = questions[idx >= 0 ? idx : d.question_id];
                    const template = q ? q.question : '';
                    const fullSentence = template && template.includes('___')
                        ? template.replace('___', d.correct_answer || '?')
                        : d.correct_answer;
                    _showCorrection(qDiv, fullSentence);
                }
            } else if (type === 'listening') {
                const inp = qDiv.querySelector('.quiz-answer-input');
                if (inp) inp.style.borderColor = d.is_correct ? '#34d399' : '#ef4444';
                if (!d.is_correct) _showCorrection(qDiv, d.correct_answer);
            } else {
                // Standard: radio or text input (includes translation)
                if (d.is_correct) {
                    const selected = qDiv.querySelector('.quiz-option.selected');
                    if (selected) selected.classList.add('correct');
                    const textInput = qDiv.querySelector('.quiz-answer-input');
                    if (textInput) textInput.style.borderColor = '#34d399';
                } else {
                    const selected = qDiv.querySelector('.quiz-option.selected');
                    if (selected) selected.classList.add('incorrect');
                    qDiv.querySelectorAll('.quiz-option').forEach(opt => {
                        if (opt.textContent.trim() === d.correct_answer) opt.classList.add('correct');
                    });
                    const textInput = qDiv.querySelector('.quiz-answer-input');
                    if (textInput) {
                        textInput.style.borderColor = '#ef4444';
                        // Show correct answer once (feedback intentionally omits it)
                        _showCorrection(qDiv, d.correct_answer);
                    }
                }
            }

            // Score badge + feedback
            const num = qDiv.querySelector('.quiz-question-num');
            if (num) {
                const badge = document.createElement('span');
                badge.className = 'quiz-score-badge ' + (score >= 1 ? 'full' : score >= 0.5 ? 'half' : 'wrong');
                badge.textContent = score >= 1 ? '✓' : score > 0 ? '½' : '✗';
                num.appendChild(badge);
            }
            if (d.feedback) {
                const fb = document.createElement('div');
                fb.className = 'quiz-feedback';
                fb.textContent = d.feedback;
                qDiv.appendChild(fb);
            }
        });
    }

    // Score summary
    const content = document.getElementById('quiz-content');
    const summary = document.createElement('div');
    summary.className = 'quiz-results';
    const scoreVal = typeof result.score === 'number' ? result.score : 0;
    const scoreDisplay = Number.isInteger(scoreVal) ? scoreVal : scoreVal.toFixed(1);
    summary.innerHTML =
        '<div class="quiz-score">' + scoreDisplay + ' / ' + result.total +
        ' (' + (result.percentage || Math.round(scoreVal / Math.max(1, result.total) * 100)) + '%)</div>' +
        '<div style="text-align:center;color:#94a3b8;font-size:0.82rem;">Quiz completed!</div>';
    content.appendChild(summary);
}

function _showCorrection(qDiv, correctAnswer) {
    const correction = document.createElement('div');
    correction.style.cssText = 'color:#34d399;font-size:0.8rem;margin-top:0.3rem;';
    correction.textContent = '✓ Correct: ' + correctAnswer;
    qDiv.appendChild(correction);
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
            const scoreVal = q.score !== undefined ? q.score : 0;
            const scoreDisplay = Number.isInteger(scoreVal) ? scoreVal : (+scoreVal).toFixed(1);
            card.innerHTML = '<div style="font-size:0.82rem;color:#e0e0e0;">' + (q.quiz_type || 'Quiz') +
                (q.score !== undefined ? ' — ' + scoreDisplay + '/' + q.total : '') + '</div>' +
                '<div class="quiz-card-meta">' + new Date(q.created_at).toLocaleString() + '</div>';
            content.appendChild(card);
        });
    } catch (e) { /* ignore */ }
}


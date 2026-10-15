// ══════════════════════════════════════════════
// ── Lesson generation & management ──
// ══════════════════════════════════════════════

async function downloadLesson(id, format) {
    // md and docx are documents to read; pdf and pdf-answers are the printable
    // worksheet and its key, split at the lesson's own Solutions section.
    const url = (format === 'pdf' || format === 'pdf-answers')
        ? '/tutor/lessons/' + id + '/sheet?answers=' + (format === 'pdf-answers')
        : '/tutor/lessons/' + id + '/export?format=' + format;
    try {
        const resp = await fetch(url, { headers: tutorAuthHeaders() });
        if (!resp.ok) {
            let detail = 'Export failed';
            try { detail = (await resp.json()).detail || detail; } catch (e) { /* keep */ }
            tutorShowToast(detail);
            return;
        }
        const blob = await resp.blob();
        const disposition = resp.headers.get('Content-Disposition') || '';
        const nameMatch = disposition.match(/filename="([^"]+)"/);
        const filename = nameMatch ? nameMatch[1] : ('lesson.' + format);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    } catch (e) { tutorShowToast('Export error: ' + e.message); }
}

async function generateLesson() {
    if (!tutorSessionId) { tutorShowToast('Create or load a session first'); return; }

    const topic = document.getElementById('lesson-topic').value.trim();
    if (!topic) { tutorShowToast('Enter a topic'); return; }
    const lessonType = document.getElementById('lesson-type').value;
    const cfg = getTutorConfig();

    const content = document.getElementById('lessons-content');
    content.innerHTML = '';

    // Show streaming
    const liveWrapper = document.createElement('div');
    const label = document.createElement('div');
    label.className = 'tutor-streaming-label';
    label.innerHTML = '<span class="pulse"></span> Generating lesson...';
    liveWrapper.appendChild(label);
    const liveBox = document.createElement('div');
    liveBox.className = 'tutor-streaming';
    liveWrapper.appendChild(liveBox);
    content.appendChild(liveWrapper);

    let reasoningLines = [];
    try {
        const resp = await fetch('/tutor/lessons', {
            method: 'POST',
            headers: tutorJsonHeaders(),
            body: JSON.stringify({
                session_id: tutorSessionId,
                topic, lesson_type: lessonType,
                target_language: cfg.target_lang,
                native_language: cfg.native_lang,
                level: cfg.level,
            }),
        });

        if (!resp.ok) {
            liveWrapper.remove();
            tutorShowToast('Failed to generate lesson');
            return;
        }

        const result = await processTutorSSE(resp, liveBox, reasoningLines);
        collapseThinking(liveWrapper, liveBox, 'Thinking — lesson');

        if (result && result.response) {
            tutorLessonRawResponse = result.response;
            // Render lesson
            const viewer = document.createElement('div');
            viewer.className = 'lesson-viewer';
            viewer.innerHTML = renderContent(result.response);
            content.appendChild(viewer);

            // Save button
            const saveBtn = document.createElement('button');
            saveBtn.textContent = '💾 Save Lesson';
            saveBtn.style.cssText = 'margin-top:0.75rem;background:#e94560;color:white;border:none;padding:0.45rem 1rem;border-radius:0.5rem;cursor:pointer;font-size:0.85rem;';
            saveBtn.onclick = () => saveLesson(topic, lessonType);
            content.appendChild(saveBtn);

            // Printing is an action on the lesson, not a second generator:
            // the markdown just produced is sent straight to the typesetter.
            content.appendChild(lessonPrintButtons(result.response, topic));
        }
    } catch (e) {
        liveWrapper.remove();
        tutorShowToast('Error: ' + e.message);
    }
}

async function saveLesson(topic, lessonType) {
    if (!tutorLessonRawResponse) { tutorShowToast('No lesson to save'); return; }
    try {
        const resp = await fetch('/tutor/lessons/save', {
            method: 'POST',
            headers: tutorJsonHeaders(),
            body: JSON.stringify({
                session_id: tutorSessionId,
                topic, lesson_type: lessonType,
                content: tutorLessonRawResponse,
            }),
        });
        if (resp.ok) {
            tutorShowToast('Lesson saved');
            loadLessons();
        } else {
            tutorShowToast('Failed to save lesson');
        }
    } catch (e) { tutorShowToast('Error: ' + e.message); }
}

async function loadLessons() {
    if (!tutorSessionId) return;
    const content = document.getElementById('lessons-list');
    try {
        const resp = await fetch('/tutor/sessions/' + tutorSessionId + '/lessons', { headers: tutorAuthHeaders() });
        const data = await resp.json();
        if (!data.lessons || data.lessons.length === 0) {
            content.innerHTML = '<div style="color:#64748b;text-align:center;padding:1rem;">No saved lessons yet</div>';
            return;
        }
        content.innerHTML = '';
        data.lessons.forEach(l => {
            const card = document.createElement('div');
            card.className = 'lesson-card';
            card.innerHTML = '<div class="lesson-card-title">' + escapeHtml(l.topic) + '</div>' +
                '<div class="lesson-card-meta">' + l.lesson_type + ' · ' + new Date(l.created_at).toLocaleString() + '</div>';
            card.onclick = () => viewLesson(l.id);
            content.appendChild(card);
        });
    } catch (e) {
        content.innerHTML = '<div style="color:#ef4444;text-align:center;">Error loading lessons</div>';
    }
}

async function viewLesson(id) {
    try {
        const resp = await fetch('/tutor/lessons/' + id, { headers: tutorAuthHeaders() });
        if (!resp.ok) { tutorShowToast('Failed to load lesson'); return; }
        const data = await resp.json();
        const content = document.getElementById('lessons-content');
        content.innerHTML = '';

        // Toolbar: Back + Export buttons
        const toolbar = document.createElement('div');
        toolbar.style.cssText = 'display:flex;gap:0.5rem;margin-bottom:0.75rem;flex-wrap:wrap;';

        const back = document.createElement('button');
        back.textContent = '← Back to lessons';
        back.style.cssText = 'background:#0f3460;color:#94a3b8;border:1px solid #1a3a6e;padding:0.3rem 0.7rem;border-radius:0.4rem;font-size:0.78rem;cursor:pointer;';
        back.onclick = () => { content.innerHTML = ''; loadLessons(); };
        toolbar.appendChild(back);

        const btnStyle = 'background:#0f3460;color:#94a3b8;border:1px solid #1a3a6e;padding:0.3rem 0.7rem;border-radius:0.4rem;font-size:0.78rem;cursor:pointer;';

        const exportMd = document.createElement('button');
        exportMd.textContent = '⬇ MD';
        exportMd.style.cssText = btnStyle;
        exportMd.onclick = () => downloadLesson(id, 'md');
        toolbar.appendChild(exportMd);

        const exportDocx = document.createElement('button');
        exportDocx.textContent = '⬇ DOCX';
        exportDocx.style.cssText = btnStyle;
        exportDocx.onclick = () => downloadLesson(id, 'docx');
        toolbar.appendChild(exportDocx);

        // MD and DOCX are documents to read; these two are the printable
        // worksheet and its key, typeset from the same lesson.
        const sheet = document.createElement('button');
        sheet.textContent = '🖨️ Sheet';
        sheet.style.cssText = btnStyle;
        sheet.onclick = () => downloadLesson(id, 'pdf');
        toolbar.appendChild(sheet);

        const key = document.createElement('button');
        key.textContent = '🖨️ Answer key';
        key.style.cssText = btnStyle;
        key.onclick = () => downloadLesson(id, 'pdf-answers');
        toolbar.appendChild(key);

        content.appendChild(toolbar);

        const viewer = document.createElement('div');
        viewer.className = 'lesson-viewer';
        const mdText = (data.content && data.content.markdown) ? data.content.markdown : JSON.stringify(data.content, null, 2);
        viewer.innerHTML = renderContent(mdText);
        content.appendChild(viewer);
    } catch (e) { tutorShowToast('Error: ' + e.message); }
}


// ══════════════════════════════════════════════
// ── Printing a lesson ──
// ══════════════════════════════════════════════
//
// The lesson the model writes already ends with a Solutions section, so the
// worksheet and its answer key are the same document cut in half. No second
// model call, and the two cannot disagree.

function lessonPrintButtons(markdown, topic) {
    const wrap = document.createElement('span');
    wrap.style.cssText = 'margin-left:0.5rem;display:inline-flex;gap:0.4rem;';
    [['🖨️ Sheet', false], ['🖨️ Answer key', true]].forEach(([label, answers]) => {
        const b = document.createElement('button');
        b.textContent = label;
        b.style.cssText = 'background:#0f3460;color:#e2e8f0;border:1px solid #1a3a6e;'
            + 'padding:0.45rem 0.9rem;border-radius:0.5rem;cursor:pointer;font-size:0.85rem;';
        b.onclick = () => printLessonMarkdown(markdown, topic, answers, b);
        wrap.appendChild(b);
    });
    return wrap;
}

async function printLessonMarkdown(markdown, topic, answers, btn) {
    const cfg = getTutorConfig();
    const was = btn ? btn.textContent : '';
    if (btn) { btn.disabled = true; btn.textContent = 'Typesetting…'; }
    try {
        const resp = await fetch(
            answers ? '/tutor/sheets/lesson/answers' : '/tutor/sheets/lesson/pdf', {
                method: 'POST',
                headers: tutorJsonHeaders(),
                body: JSON.stringify({
                    markdown,
                    title: topic || 'Lesson',
                    language: cfg.target_lang,
                    level: cfg.level,
                    student: '',
                    practice_space: true,
                    native: cfg.native_lang,
                }),
            });
        if (!resp.ok) {
            let detail = 'Could not build the PDF';
            try { detail = (await resp.json()).detail || detail; } catch (e) { /* keep */ }
            tutorShowToast(detail);
            return;
        }
        const url = URL.createObjectURL(await resp.blob());
        const a = document.createElement('a');
        a.href = url;
        a.download = (topic || 'lesson').replace(/[^\w -]/g, '') +
            (answers ? '-answers.pdf' : '.pdf');
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (e) {
        tutorShowToast('Print error: ' + e.message);
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = was; }
    }
}

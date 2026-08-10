// ══════════════════════════════════════════════
// ── File upload & management ──
// ══════════════════════════════════════════════
let attachedFiles = [];  // [{id, filename}] — chips currently shown
let sessionFileIds = new Set();  // all file IDs used in this session

async function handleFileSelect(event) {
    const fileList = event.target.files;
    if (!fileList || fileList.length === 0) return;
    const formData = new FormData();
    for (const f of fileList) formData.append('files', f);
    try {
        const resp = await fetch('/files/upload', {
            method: 'POST',
            headers: authHeaders(),
            body: formData,
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            showToast('Upload failed: ' + (err.detail || resp.statusText));
            return;
        }
        const data = await resp.json();
        for (const f of data.files) {
            if (f.error) {
                showToast(f.filename + ': ' + f.error);
                continue;
            }
            if (f.ocr && f.chars === 0) {
                showToast('No text could be read from ' + f.filename);
                continue;
            }

            if (f.ocr && f.export_url) {
                // A scan or photo is a CONVERSION job, not a conversation. Do NOT
                // attach it: attaching drags the whole transcription through the
                // agent's planning and ReAct loop on every later message, costing
                // minutes to hand back text that is already on disk. Deliver the
                // document and stop. It stays in the Files panel, where 📎 Attach
                // is there if the user genuinely wants to discuss it.
                showToast('OCR complete (' + f.chars + ' chars) — downloading ' +
                          f.export_format.toUpperCase() +
                          '. Not attached to the chat; use 📁 Files → 📎 Attach to discuss it.');
                downloadFileText(f.id, f.filename, f.export_format);
                continue;
            }

            attachedFiles.push({ id: f.id, filename: f.filename });
            sessionFileIds.add(f.id);
            addFileChip(f.id, f.filename);
        }
    } catch (e) {
        showToast('Upload error: ' + e.message);
    }
    event.target.value = '';
}

function addFileChip(id, filename) {
    const chips = document.getElementById('file-chips');
    const chip = document.createElement('span');
    chip.className = 'file-chip';
    chip.dataset.fileId = id;
    chip.innerHTML = '📄 ' + filename + ' <button onclick="removeFileChip(\'' + id + '\')">&times;</button>';
    chips.appendChild(chip);
}

function removeFileChip(id) {
    attachedFiles = attachedFiles.filter(f => f.id !== id);
    // Must also leave the session set, or the file keeps being sent with every
    // message even though its chip is gone.
    sessionFileIds.delete(id);
    const chip = document.querySelector('.file-chip[data-file-id="' + id + '"]');
    if (chip) chip.remove();
}

// Attach an already-uploaded file to the conversation.
//
// sessionFileIds lives only in page memory and is otherwise filled in just once,
// by the upload handler. That means a page reload silently detaches every file:
// it is still listed in the Files panel with its extracted text in the database,
// but no longer reaches the agent. This is how a file can look attached while
// the agent reports no attachment at all.
function attachExistingFile(id, filename) {
    if (sessionFileIds.has(id)) {
        showToast(filename + ' is already attached');
        return;
    }
    sessionFileIds.add(id);
    attachedFiles.push({ id: id, filename: filename });
    addFileChip(id, filename);
    showToast('Attached ' + filename);
    loadFilesList();
}

function getAttachedFileIds() {
    return [...sessionFileIds];
}

function clearFileChips() {
    attachedFiles = [];
    document.getElementById('file-chips').innerHTML = '';
}

function clearSessionFiles() {
    attachedFiles = [];
    sessionFileIds.clear();
    document.getElementById('file-chips').innerHTML = '';
}

// ── Files panel ──
function openFilesPanel() {
    document.getElementById('files-panel').classList.add('open');
    document.getElementById('files-panel-overlay').classList.add('open');
    loadFilesList();
}

function closeFilesPanel() {
    document.getElementById('files-panel').classList.remove('open');
    document.getElementById('files-panel-overlay').classList.remove('open');
}

// Fetch via XHR-with-auth rather than a plain link, since the endpoint needs
// the Authorization header; a bare <a href> would come back 401.
async function downloadFileText(fileId, filename, fmt) {
    try {
        const resp = await fetch('/files/' + fileId + '/export?format=' + fmt, {
            headers: authHeaders(),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            showToast(err.detail || 'Export failed');
            return;
        }
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename.replace(/\.[^.]+$/, '') + '.' + fmt;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    } catch (e) {
        showToast('Export error: ' + e.message);
    }
}

async function loadFilesList() {
    const list = document.getElementById('files-list');
    try {
        const resp = await fetch('/files', { headers: authHeaders() });
        const data = await resp.json();
        if (!data.files || data.files.length === 0) {
            list.innerHTML = '<div style="color:#64748b;text-align:center;padding:2rem;">No uploaded files</div>';
            return;
        }
        list.innerHTML = '';
        data.files.forEach(f => {
            const item = document.createElement('div');
            item.className = 'file-item';
            const info = document.createElement('div');
            info.innerHTML = '<div class="file-item-name">📄 ' + f.filename + '</div>' +
                '<div class="file-item-size">' + (f.size / 1024).toFixed(1) + ' KB · ' + f.extension + '</div>';
            item.appendChild(info);

            const actions = document.createElement('div');
            actions.style.cssText = 'display:flex;gap:0.3rem;align-items:center;';

            // Re-attach a previously uploaded file to this conversation.
            const isAttached = sessionFileIds.has(f.id);
            const attachBtn = document.createElement('button');
            attachBtn.textContent = isAttached ? '✓ Attached' : '📎 Attach';
            attachBtn.title = isAttached
                ? 'Already attached to this conversation'
                : 'Attach this file to the conversation';
            attachBtn.disabled = isAttached;
            attachBtn.style.cssText =
                'border:1px solid ' + (isAttached ? '#166534' : '#1a3a6e') + ';' +
                'background:' + (isAttached ? '#052e16' : '#0f3460') + ';' +
                'color:' + (isAttached ? '#4ade80' : '#94a3b8') + ';' +
                'cursor:' + (isAttached ? 'default' : 'pointer') + ';' +
                'font-size:0.65rem;padding:0.2rem 0.4rem;border-radius:0.3rem;';
            if (!isAttached) {
                attachBtn.onclick = () => attachExistingFile(f.id, f.filename);
            }
            actions.appendChild(attachBtn);
            // Download the extracted text (OCR included) as a real document.
            [['DOCX', 'docx'], ['ODT', 'odt']].forEach(([label, fmt]) => {
                const btn = document.createElement('button');
                btn.textContent = label;
                btn.title = 'Download extracted text as ' + label;
                btn.style.cssText = 'background:#0f3460;border:1px solid #1a3a6e;color:#94a3b8;' +
                    'cursor:pointer;font-size:0.65rem;padding:0.2rem 0.4rem;border-radius:0.3rem;';
                btn.onclick = () => downloadFileText(f.id, f.filename, fmt);
                actions.appendChild(btn);
            });

            const delBtn = document.createElement('button');
            delBtn.textContent = '✕';
            delBtn.style.cssText = 'background:none;border:none;color:#ef4444;cursor:pointer;font-size:1rem;';
            delBtn.onclick = () => deleteFileFromPanel(f.id, f.filename);
            actions.appendChild(delBtn);

            item.appendChild(actions);
            list.appendChild(item);
        });
    } catch (e) {
        list.innerHTML = '<div style="color:#ef4444;text-align:center;">Error loading files</div>';
    }
}

async function deleteFileFromPanel(id, name) {
    if (!confirm('Delete file "' + name + '"?')) return;
    try {
        const resp = await fetch('/files/' + id, { method: 'DELETE', headers: authHeaders() });
        if (resp.ok) {
            removeFileChip(id);
            loadFilesList();
            showToast('Deleted: ' + name);
        }
    } catch (e) { showToast('Error deleting file'); }
}

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
        const name = filename.replace(/\.[^.]+$/, '') + '.' + fmt;
        // Chrome/Edge on localhost or https offer a real "Save as" dialog;
        // elsewhere the browser downloads under this name instead.
        if (window.showSaveFilePicker) {
            try {
                const handle = await window.showSaveFilePicker({
                    suggestedName: name,
                    types: [{ description: fmt.toUpperCase() + ' document',
                              accept: { [blob.type || 'application/octet-stream']: ['.' + fmt] } }],
                });
                const w = await handle.createWritable();
                await w.write(blob);
                await w.close();
                showToast('Saved ' + handle.name);
                return;
            } catch (e) {
                if (e && e.name === 'AbortError') return;   // cancelled
            }
        }
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = name;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 10000);
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


// ══════════════════════════════════════════════
// ── Scanning a document ──
//
// One button. It asks which model to read with (the one best suited to pages
// is preselected), then streams a page at a time. The scan holds the GPU for
// its whole run, which is accepted rather than worked around: the dialog says
// so, and everyone else sees "scan — page 42 of 400" in the GPU panel.
// ══════════════════════════════════════════════

let _scanModels = [];

async function openScanDialog() {
    const dlg = document.getElementById('scan-dialog');
    const sel = document.getElementById('scan-model');
    sel.innerHTML = '<option>loading…</option>';
    dlg.style.display = 'flex';
    try {
        const r = await fetch('/files/scan/models', { headers: authHeaders() });
        const d = await r.json();
        _scanModels = d.models || [];
        sel.innerHTML = '';
        if (!_scanModels.length) {
            sel.innerHTML = '<option value="">no vision model installed</option>';
            document.getElementById('scan-model-why').textContent = d.hint || '';
            document.getElementById('scan-go').disabled = true;
            return;
        }
        for (const m of _scanModels) {
            const o = document.createElement('option');
            o.value = m.name;
            o.textContent = `${m.name} (${m.size_gb} GB)${m.recommended ? ' — recommended' : ''}`;
            sel.appendChild(o);
        }
        sel.value = d.recommended || _scanModels[0].name;
        sel.onchange = showScanWhy;
        showScanWhy();
        document.getElementById('scan-go').disabled = false;
    } catch (e) {
        sel.innerHTML = '<option value="">could not reach the server</option>';
    }
}

function showScanWhy() {
    const name = document.getElementById('scan-model').value;
    const m = _scanModels.find(x => x.name === name);
    document.getElementById('scan-model-why').textContent = m ? m.why : '';
}

function closeScanDialog() {
    document.getElementById('scan-dialog').style.display = 'none';
}

function chooseScanFile() {
    document.getElementById('scan-input').click();
}

// What to do with a finished scan: keep the text as a document, or use it.
// Buttons rather than an automatic dialog, because browsers only open a
// "Save as" dialog in answer to a click.
function scanResultActions(fileId, filename) {
    const row = document.createElement('div');
    row.className = 'scan-actions';
    const add = (label, title, onclick) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.textContent = label;
        b.title = title;
        b.onclick = onclick;
        row.appendChild(b);
    };
    add('💾 Save as DOCX', 'Save the scanned text as a Word document',
        () => downloadFileText(fileId, filename, 'docx'));
    add('💾 Save as ODT', 'Save the scanned text as a LibreOffice document',
        () => downloadFileText(fileId, filename, 'odt'));
    add('📎 Attach to chat', 'Ask questions about it in this conversation', (ev) => {
        attachExistingFile(fileId, filename);
        ev.target.disabled = true;
        ev.target.textContent = '✓ Attached';
    });
    return row;
}

async function startScan(event) {
    const f = event.target.files && event.target.files[0];
    event.target.value = '';                 // so the same file can be picked again
    if (!f) return;
    closeScanDialog();

    const model = document.getElementById('scan-model').value;
    const first = parseInt(document.getElementById('scan-first').value, 10) || 1;
    const last = parseInt(document.getElementById('scan-last').value, 10) || 0;

    const box = document.getElementById('scan-progress');
    box.style.display = 'block';
    box.textContent = `Scanning ${f.name}…`;

    const body = new FormData();
    body.append('file', f);
    const force = document.getElementById('scan-force').checked;
    const qs = `?model=${encodeURIComponent(model)}&first_page=${first}`
             + `&last_page=${last}&force_ocr=${force}`;

    try {
        const r = await fetch('/files/scan' + qs, {
            method: 'POST', headers: authHeaders(), body,
        });
        if (!r.ok) {
            let detail = r.statusText;
            try { detail = (await r.json()).detail || detail; } catch (e) { /* keep */ }
            box.textContent = 'Scan failed: ' + detail;
            return;
        }
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = '', pages = 0, started = Date.now();
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            while (buf.includes('\n\n')) {
                const idx = buf.indexOf('\n\n');
                const chunk = buf.slice(0, idx);
                buf = buf.slice(idx + 2);
                let ev = '', data = '';
                for (const line of chunk.split('\n')) {
                    if (line.startsWith('event: ')) ev = line.slice(7);
                    else if (line.startsWith('data: ')) data += line.slice(6);
                }
                if (!data) continue;
                const d = JSON.parse(data);
                if (ev === 'queue') {
                    box.textContent = d.text || 'Waiting for the GPU…';
                } else if (ev === 'start') {
                    // Say whether the file already carries text, so a long
                    // wait is not spent re-reading what was already readable.
                    let had = '';
                    if (d.sampled && d.with_text_layer !== undefined) {
                        const share = d.with_text_layer / d.sampled;
                        if (d.force_ocr && share > 0.5) {
                            had = ' — note: it already has a text layer, which you chose to ignore';
                        } else if (share > 0.9) {
                            had = ' — most pages already have text, so this will be quick';
                        } else if (share > 0.1) {
                            had = ` — about ${Math.round(share * 100)}% of pages already have text`;
                        }
                    }
                    box.textContent = `Reading ${d.file} with ${d.model}${had}…`;
                } else if (ev === 'page') {
                    pages++;
                    const each = (Date.now() - started) / 1000 / pages;
                    const leftN = Math.max(0, d.total - d.page);
                    const left = leftN ? ` · about ${Math.round(leftN * each)}s left` : '';
                    box.textContent = `Page ${d.page} of ${d.total}` +
                        (d.kind === 'ocr' ? ' (scanned)' : d.kind === 'text' ? ' (text layer)' : ' — unreadable') +
                        left;
                } else if (ev === 'done') {
                    const s = d.stats;
                    const bits = [`${s.ocr + s.text} page(s) read`];
                    if (s.text) bits.push(`${s.text} already had a text layer`);
                    if (s.ocr) bits.push(`${s.ocr} read by ${s.model}`);
                    if (s.empty) bits.push(`${s.empty} unreadable`);
                    box.textContent = `✅ ${d.filename}: ${bits.join(', ')} — ${d.chars.toLocaleString()} characters.`;
                    if (d.too_long_for_context) {
                        // Not a warning any more: attaching it works, the
                        // pages that answer each question are picked for it.
                        box.textContent += ' Attach it and ask away — it is too'
                            + ' long to read at once, so the pages relevant to'
                            + ' each question are picked automatically.';
                    }
                    box.appendChild(scanResultActions(d.file_id, d.filename));
                    if (typeof loadFilesList === 'function') loadFilesList();
                    showToast('Scan complete: ' + d.filename);
                } else if (ev === 'error') {
                    box.textContent = 'Scan failed: ' + d;
                }
            }
        }
    } catch (e) {
        box.textContent = 'Scan failed: ' + e.message;
    }
}

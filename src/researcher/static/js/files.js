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
            } else {
                attachedFiles.push({ id: f.id, filename: f.filename });
                sessionFileIds.add(f.id);
                addFileChip(f.id, f.filename);
            }
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
    const chip = document.querySelector('.file-chip[data-file-id="' + id + '"]');
    if (chip) chip.remove();
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
            const delBtn = document.createElement('button');
            delBtn.textContent = '✕';
            delBtn.style.cssText = 'background:none;border:none;color:#ef4444;cursor:pointer;font-size:1rem;';
            delBtn.onclick = () => deleteFileFromPanel(f.id, f.filename);
            item.appendChild(delBtn);
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

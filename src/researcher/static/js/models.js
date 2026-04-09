// ── Model dropdown ──
async function loadModels() {
    try {
        const resp = await fetch('/models', { headers: authHeaders() });
        const data = await resp.json();
        modelSelect.innerHTML = '';
        data.models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.name;
            opt.textContent = m.name + ' (' + m.params + ', ' + m.size_gb + ' GB)';
            const litellmName = 'ollama/' + m.name;
            if (litellmName === data.current || m.name === data.current) {
                opt.selected = true;
                modelInfo.textContent = m.family;
            }
            modelSelect.appendChild(opt);
        });
    } catch (e) {
        modelSelect.innerHTML = '<option>error loading models</option>';
    }
}

async function switchModel() {
    const chosen = modelSelect.value;
    modelSelect.disabled = true;
    modelInfo.textContent = 'switching...';
    try {
        const resp = await fetch('/model', {
            method: 'POST',
            headers: jsonAuthHeaders(),
            body: JSON.stringify({ model: chosen }),
        });
        if (resp.ok) {
            const data = await resp.json();
            modelInfo.textContent = 'active: ' + data.model;
            addAssistantMessage('Model switched to ' + data.model, [], {});
        } else {
            const err = await resp.json();
            modelInfo.textContent = 'switch failed';
            addSimpleMsg('error', 'Switch failed: ' + err.detail);
        }
    } catch (e) {
        modelInfo.textContent = 'switch failed';
        addSimpleMsg('error', 'Connection error: ' + e.message);
    } finally { modelSelect.disabled = false; }
}

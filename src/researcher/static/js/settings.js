// ── Memory depth toggle ──
const depthToggle = document.getElementById('depth-toggle');
let memoryDepth = 'shallow';

async function loadMemoryDepth() {
    try {
        const resp = await fetch('/memory-depth');
        const data = await resp.json();
        memoryDepth = data.depth;
        updateDepthUI();
    } catch (e) { /* ignore */ }
}

function updateDepthUI() {
    depthToggle.textContent = memoryDepth === 'deep' ? '🧠 Deep' : '🧠 Shallow';
    depthToggle.classList.toggle('deep', memoryDepth === 'deep');
    depthToggle.title = memoryDepth === 'deep'
        ? 'Memory: deep (thorough but slow) — click to switch to shallow'
        : 'Memory: shallow (fast) — click to switch to deep';
}

async function toggleMemoryDepth() {
    const newDepth = memoryDepth === 'shallow' ? 'deep' : 'shallow';
    depthToggle.disabled = true;
    try {
        const resp = await fetch('/memory-depth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ depth: newDepth }),
        });
        if (resp.ok) {
            memoryDepth = newDepth;
            updateDepthUI();
        }
    } catch (e) { /* ignore */ }
    finally { depthToggle.disabled = false; }
}

// ══════════════════════════════════════════════
// ── LLM Parameters panel ──
// ══════════════════════════════════════════════
const LLM_PARAM_KEYS = [
    'temperature', 'top_k', 'top_p', 'min_p',
    'repeat_penalty', 'frequency_penalty', 'presence_penalty',
    'num_predict', 'num_ctx', 'seed', 'planning_max_attempts'
];

function updateParamDisplay(key) {
    const el = document.getElementById('param-' + key);
    const valEl = document.getElementById('val-' + key);
    if (el && valEl) valEl.textContent = el.value;
}

function setParamUI(params) {
    for (const key of LLM_PARAM_KEYS) {
        if (params[key] !== undefined) {
            const el = document.getElementById('param-' + key);
            if (el) { el.value = params[key]; updateParamDisplay(key); }
        }
    }
}

function getParamValues() {
    const params = {};
    for (const key of LLM_PARAM_KEYS) {
        const el = document.getElementById('param-' + key);
        if (el) {
            params[key] = ['top_k', 'num_predict', 'num_ctx', 'seed', 'planning_max_attempts'].includes(key)
                ? parseInt(el.value, 10)
                : parseFloat(el.value);
        }
    }
    return params;
}

function openLlmSettings() {
    document.getElementById('llm-settings-overlay').classList.add('open');
    document.getElementById('llm-settings-panel').classList.add('open');
    // Load current LLM values from server
    fetch('/llm-params', { headers: authHeaders() })
        .then(r => r.ok ? r.json() : null)
        .then(data => { if (data && data.params) setParamUI(data.params); })
        .catch(() => {});
    // Load current Image values from server
    fetch('/image-params', { headers: authHeaders() })
        .then(r => r.ok ? r.json() : null)
        .then(data => { if (data && data.params) setImgParamUI(data.params); })
        .catch(() => {});
}

function closeLlmSettings() {
    document.getElementById('llm-settings-overlay').classList.remove('open');
    document.getElementById('llm-settings-panel').classList.remove('open');
}

// ── Image Parameters helpers ──
const IMG_PARAM_KEYS = [
    'hires_enabled', 'hires_width', 'hires_height',
    'tile_overlap', 'hires_strength', 'tile_width', 'tile_height'
];

function updateImgParamDisplay(key) {
    const el = document.getElementById('param-img-' + key);
    const valEl = document.getElementById('val-img-' + key);
    if (!el || !valEl) return;
    if (el.type === 'checkbox') {
        valEl.textContent = el.checked ? 'On' : 'Off';
    } else {
        valEl.textContent = el.value;
    }
}

function setImgParamUI(params) {
    for (const key of IMG_PARAM_KEYS) {
        if (params[key] !== undefined) {
            const el = document.getElementById('param-img-' + key);
            if (!el) continue;
            if (el.type === 'checkbox') {
                el.checked = !!params[key];
            } else {
                el.value = params[key];
            }
            updateImgParamDisplay(key);
        }
    }
}

function getImgParamValues() {
    const params = {};
    const floatKeys = new Set(['hires_strength']);
    for (const key of IMG_PARAM_KEYS) {
        const el = document.getElementById('param-img-' + key);
        if (!el) continue;
        if (el.type === 'checkbox') {
            params[key] = el.checked;
        } else if (floatKeys.has(key)) {
            params[key] = parseFloat(el.value);
        } else {
            params[key] = parseInt(el.value, 10);
        }
    }
    return params;
}

async function applyLlmParams() {
    const params = getParamValues();
    const imgParams = getImgParamValues();
    try {
        // Apply LLM params
        const resp = await fetch('/llm-params', {
            method: 'POST',
            headers: jsonAuthHeaders(),
            body: JSON.stringify({ params }),
        });
        // Apply image params
        const imgResp = await fetch('/image-params', {
            method: 'POST',
            headers: jsonAuthHeaders(),
            body: JSON.stringify({ params: imgParams }),
        });
        if (resp.ok) {
            const data = await resp.json();
            setParamUI(data.params);
        }
        if (imgResp.ok) {
            const imgData = await imgResp.json();
            setImgParamUI(imgData.params);
        }
        if (resp.ok && imgResp.ok) {
            showToast('Parameters applied');
            closeLlmSettings();
        } else {
            const err = await resp.json().catch(() => ({}));
            showToast('Error: ' + (err.detail || 'Failed to apply'));
        }
    } catch (e) {
        showToast('Error: ' + e.message);
    }
}

async function resetLlmParams() {
    try {
        const resp = await fetch('/llm-params/reset', {
            method: 'POST',
            headers: jsonAuthHeaders(),
        });
        const imgResp = await fetch('/image-params/reset', {
            method: 'POST',
            headers: jsonAuthHeaders(),
        });
        if (resp.ok) {
            const data = await resp.json();
            setParamUI(data.params);
        }
        if (imgResp.ok) {
            const imgData = await imgResp.json();
            setImgParamUI(imgData.params);
        }
        if (resp.ok) {
            showToast('Parameters reset to defaults');
        } else {
            showToast('Error: Failed to reset');
        }
    } catch (e) {
        showToast('Error: ' + e.message);
    }
}

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
    'num_predict', 'num_ctx', 'seed', 'planning_max_attempts', 'max_iter', 'crew_timeout_sec'
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
            params[key] = ['top_k', 'num_predict', 'num_ctx', 'seed', 'planning_max_attempts', 'max_iter', 'crew_timeout_sec'].includes(key)
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
    // Load OCR / speech / tutor / video values
    fetch('/subsystem-params', { headers: authHeaders() })
        .then(r => r.ok ? r.json() : null)
        .then(data => { if (data && data.params) setSubParamUI(data.params); })
        .catch(() => {});
}

function closeLlmSettings() {
    document.getElementById('llm-settings-overlay').classList.remove('open');
    document.getElementById('llm-settings-panel').classList.remove('open');
}

// ── Image Parameters helpers ──
const IMG_PARAM_KEYS = [
    'image_backend',
    'hires_enabled', 'hires_width', 'hires_height',
    'tile_overlap', 'hires_strength', 'realesrgan_enabled',
    'tile_width', 'tile_height',
    'refine_rounds', 'critic_model'
];

function updateImgParamDisplay(key) {
    const el = document.getElementById('param-img-' + key);
    const valEl = document.getElementById('val-img-' + key);
    if (!el || !valEl) return;
    if (el.type === 'checkbox') {
        if (key === 'realesrgan_enabled') {
            valEl.textContent = el.checked ? 'Real-ESRGAN' : 'High-Pass';
        } else {
            valEl.textContent = el.checked ? 'On' : 'Off';
        }
    } else if (el.tagName === 'SELECT') {
        valEl.textContent = el.value;
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
    const stringKeys = new Set(['image_backend', 'critic_model']);
    for (const key of IMG_PARAM_KEYS) {
        const el = document.getElementById('param-img-' + key);
        if (!el) continue;
        if (el.type === 'checkbox') {
            params[key] = el.checked;
        } else if (stringKeys.has(key)) {
            params[key] = el.value;
        } else if (floatKeys.has(key)) {
            params[key] = parseFloat(el.value);
        } else {
            params[key] = parseInt(el.value, 10);
        }
    }
    return params;
}


// ── Subsystem parameters (OCR / speech / tutor SRS / video) ──
// These are process-wide rather than per-user, so they come from one grouped
// endpoint instead of one call per subsystem.
const SUB_PARAM_GROUPS = {
    ocr:     ['ocr_enabled','ocr_model','ocr_dpi','ocr_max_dim','ocr_max_pages','ocr_timeout','ocr_auto_export'],
    whisper: ['whisper_model','whisper_device','whisper_compute_type','whisper_vad'],
    srs:     ['srs_algo','srs_desired_retention','srs_max_interval','srs_initial_ease'],
    video:   ['video_backend','width','height','fps','seconds','steps','guidance', 'clips'],
};
const SUB_FLOAT_KEYS = new Set(['srs_desired_retention','srs_initial_ease','guidance']);
const SUB_STRING_KEYS = new Set(['ocr_model','ocr_auto_export','whisper_model',
    'whisper_device','whisper_compute_type','srs_algo','video_backend']);

function updateSubParamDisplay(key) {
    const el = document.getElementById('param-sub-' + key);
    const valEl = document.getElementById('val-sub-' + key);
    if (!el || !valEl) return;
    valEl.textContent = el.type === 'checkbox' ? (el.checked ? 'On' : 'Off') : el.value;
}

function setSubParamUI(groups) {
    for (const [group, keys] of Object.entries(SUB_PARAM_GROUPS)) {
        const vals = groups[group];
        if (!vals) continue;
        for (const key of keys) {
            if (vals[key] === undefined) continue;
            const el = document.getElementById('param-sub-' + key);
            if (!el) continue;
            if (el.type === 'checkbox') el.checked = !!vals[key];
            else el.value = vals[key];
            updateSubParamDisplay(key);
        }
    }
}

function getSubParamValues() {
    const out = {};
    for (const [group, keys] of Object.entries(SUB_PARAM_GROUPS)) {
        const vals = {};
        for (const key of keys) {
            const el = document.getElementById('param-sub-' + key);
            if (!el) continue;
            if (el.type === 'checkbox') vals[key] = el.checked;
            else if (SUB_STRING_KEYS.has(key)) vals[key] = el.value;
            else if (SUB_FLOAT_KEYS.has(key)) vals[key] = parseFloat(el.value);
            else vals[key] = parseInt(el.value, 10);
        }
        if (Object.keys(vals).length) out[group] = vals;
    }
    return out;
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
        // Apply subsystem params (OCR / speech / tutor / video)
        const subResp = await fetch('/subsystem-params', {
            method: 'POST',
            headers: jsonAuthHeaders(),
            body: JSON.stringify({ params: getSubParamValues() }),
        });
        if (subResp.ok) {
            const subData = await subResp.json();
            if (subData && subData.params) setSubParamUI(subData.params);
        }
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

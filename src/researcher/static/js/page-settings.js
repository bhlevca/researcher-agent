// ══════════════════════════════════════════════
// ── Reusable LLM Settings panel for Tutor & Composer pages ──
// ══════════════════════════════════════════════
// Usage: call initPageSettings({ endpoint, authHeaders, jsonAuthHeaders, showToast })
// endpoint: e.g. '/tutor/llm-params' or '/composer/llm-params'

const PAGE_LLM_PARAM_KEYS = [
    'temperature', 'top_k', 'top_p', 'min_p',
    'repeat_penalty', 'frequency_penalty', 'presence_penalty',
    'num_predict', 'num_ctx', 'seed', 'max_iter', 'planning_max_attempts'
];

function _pgUpdateParamDisplay(key) {
    const el = document.getElementById('pg-param-' + key);
    const valEl = document.getElementById('pg-val-' + key);
    if (el && valEl) valEl.textContent = el.value;
}

function _pgSetParamUI(params) {
    for (const key of PAGE_LLM_PARAM_KEYS) {
        if (params[key] !== undefined) {
            const el = document.getElementById('pg-param-' + key);
            if (el) { el.value = params[key]; _pgUpdateParamDisplay(key); }
        }
    }
}

function _pgGetParamValues() {
    const params = {};
    const intKeys = new Set(['top_k', 'num_predict', 'num_ctx', 'seed', 'max_iter', 'planning_max_attempts']);
    for (const key of PAGE_LLM_PARAM_KEYS) {
        const el = document.getElementById('pg-param-' + key);
        if (el) {
            params[key] = intKeys.has(key) ? parseInt(el.value, 10) : parseFloat(el.value);
        }
    }
    return params;
}

let _pgSettings = {};

function openPageSettings() {
    document.getElementById('pg-settings-overlay').classList.add('open');
    document.getElementById('pg-settings-panel').classList.add('open');
    fetch(_pgSettings.endpoint, { headers: _pgSettings.authHeaders() })
        .then(r => r.ok ? r.json() : null)
        .then(data => { if (data && data.params) _pgSetParamUI(data.params); })
        .catch(() => {});
}

function closePageSettings() {
    document.getElementById('pg-settings-overlay').classList.remove('open');
    document.getElementById('pg-settings-panel').classList.remove('open');
}

async function applyPageSettings() {
    const params = _pgGetParamValues();
    try {
        const resp = await fetch(_pgSettings.endpoint, {
            method: 'POST',
            headers: _pgSettings.jsonAuthHeaders(),
            body: JSON.stringify({ params }),
        });
        if (resp.ok) {
            const data = await resp.json();
            _pgSetParamUI(data.params);
            _pgSettings.showToast('Parameters applied');
            closePageSettings();
        } else {
            const err = await resp.json().catch(() => ({}));
            _pgSettings.showToast('Error: ' + (err.detail || 'Failed to apply'));
        }
    } catch (e) {
        _pgSettings.showToast('Error: ' + e.message);
    }
}

async function resetPageSettings() {
    try {
        const resp = await fetch(_pgSettings.endpoint + '/reset', {
            method: 'POST',
            headers: _pgSettings.jsonAuthHeaders(),
        });
        if (resp.ok) {
            const data = await resp.json();
            _pgSetParamUI(data.params);
            _pgSettings.showToast('Parameters reset to defaults');
        } else {
            _pgSettings.showToast('Reset failed');
        }
    } catch (e) {
        _pgSettings.showToast('Error: ' + e.message);
    }
}

function initPageSettings(opts) {
    _pgSettings = opts;
}

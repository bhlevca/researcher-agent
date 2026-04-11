// ══════════════════════════════════════════════
// ── Application initialization ──
// ══════════════════════════════════════════════

// ── Auto-resize textarea ──
input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 128) + 'px';
});
input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); lastInputWasVoice = false; sendMessage(); }
});

// ── Model switch handler ──
modelSelect.addEventListener('change', switchModel);

// ── Voice select handler ──
document.getElementById('voice-select').addEventListener('change', (e) => {
    selectedVoiceName = e.target.value;
    localStorage.setItem('ttsVoice', selectedVoiceName);
    showToast('Voice: ' + selectedVoiceName);
});

// Close export menus on outside click
document.addEventListener('click', () => {
    document.querySelectorAll('.export-menu.open').forEach(m => m.classList.remove('open'));
});

// ── Auth initialization & startup ──
(async function init() {
    // Check if invite code is required
    try {
        const resp = await fetch('/auth/config');
        if (resp.ok) {
            const cfg = await resp.json();
            inviteRequired = cfg.invite_required;
        }
    } catch (e) { /* ignore */ }

    // Validate existing token
    if (authToken) {
        try {
            const resp = await fetch('/auth/me', { headers: authHeaders() });
            if (!resp.ok) { setAuthState(null, null); }
        } catch (e) { setAuthState(null, null); }
    }
    updateAuthUI();

    // If not logged in, force the login modal open
    if (!authUser) {
        openAuthModal('login');
    } else {
        loadModels();
    }

    // Load settings & voices
    loadMemoryDepth();
    loadVoices();
    // Sync image param sliders with server defaults
    fetch('/image-params', { headers: authHeaders() })
        .then(r => r.ok ? r.json() : null)
        .then(data => { if (data && data.params) setImgParamUI(data.params); })
        .catch(() => {});
})();

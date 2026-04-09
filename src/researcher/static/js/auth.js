// ── Auth state & helpers ──
let authToken = localStorage.getItem('authToken') || null;
let authUser = JSON.parse(localStorage.getItem('authUser') || 'null');
let authMode = 'login';
let inviteRequired = false;

function authHeaders() {
    const h = {};
    if (authToken) h['Authorization'] = 'Bearer ' + authToken;
    return h;
}

function jsonAuthHeaders() {
    return { 'Content-Type': 'application/json', ...authHeaders() };
}

function setAuthState(token, user) {
    authToken = token;
    authUser = user;
    if (token) {
        localStorage.setItem('authToken', token);
        localStorage.setItem('authUser', JSON.stringify(user));
    } else {
        localStorage.removeItem('authToken');
        localStorage.removeItem('authUser');
    }
    updateAuthUI();
}

function updateAuthUI() {
    const badge = document.getElementById('user-badge');
    const loginBtn = document.getElementById('login-btn');
    const inputArea = document.getElementById('input-area');
    const chatContainer = document.getElementById('chat-container');
    // Tutor page equivalents
    const tutorInputArea = document.getElementById('tutor-input-area');
    const tutorChat = document.getElementById('tutor-chat');
    if (authUser) {
        if (badge) badge.style.display = 'flex';
        const nameEl = document.getElementById('user-badge-name');
        if (nameEl) nameEl.textContent = authUser.username;
        if (loginBtn) loginBtn.style.display = 'none';
        if (inputArea) { inputArea.style.pointerEvents = ''; inputArea.style.opacity = ''; }
        if (chatContainer) chatContainer.style.opacity = '';
        if (tutorInputArea) { tutorInputArea.style.pointerEvents = ''; tutorInputArea.style.opacity = ''; }
        if (tutorChat) tutorChat.style.opacity = '';
    } else {
        if (badge) badge.style.display = 'none';
        if (loginBtn) loginBtn.style.display = '';
        // Grey out the UI when not logged in
        if (inputArea) { inputArea.style.pointerEvents = 'none'; inputArea.style.opacity = '0.4'; }
        if (chatContainer) chatContainer.style.opacity = '0.4';
        if (tutorInputArea) { tutorInputArea.style.pointerEvents = 'none'; tutorInputArea.style.opacity = '0.4'; }
        if (tutorChat) tutorChat.style.opacity = '0.4';
    }
}

function openAuthModal(mode) {
    authMode = mode || 'login';
    const overlay = document.getElementById('auth-overlay');
    const title = document.getElementById('auth-title');
    const submit = document.getElementById('auth-submit');
    const toggle = document.getElementById('auth-toggle');
    const invite = document.getElementById('auth-invite');
    const errEl = document.getElementById('auth-error');
    errEl.style.display = 'none';
    document.getElementById('auth-username').value = '';
    document.getElementById('auth-password').value = '';
    document.getElementById('auth-invite').value = '';
    document.getElementById('auth-website').value = '';

    if (authMode === 'register') {
        title.textContent = 'Register';
        submit.textContent = 'Register';
        toggle.innerHTML = 'Have an account? <a onclick="switchAuthMode()">Login</a>';
        invite.style.display = inviteRequired ? '' : 'none';
    } else {
        title.textContent = 'Login';
        submit.textContent = 'Login';
        toggle.innerHTML = 'No account? <a onclick="switchAuthMode()">Register</a>';
        invite.style.display = 'none';
    }
    // Hide cancel button if login is required (no user logged in)
    const cancelBtn = document.getElementById('auth-cancel-btn');
    if (cancelBtn) cancelBtn.style.display = authUser ? '' : 'none';
    overlay.classList.add('open');
    document.getElementById('auth-username').focus();
}

function closeAuthModal() {
    // Cannot close if not logged in
    if (!authUser) return;
    document.getElementById('auth-overlay').classList.remove('open');
}

function switchAuthMode() {
    openAuthModal(authMode === 'login' ? 'register' : 'login');
}

async function submitAuth() {
    const errEl = document.getElementById('auth-error');
    errEl.style.display = 'none';
    const username = document.getElementById('auth-username').value.trim();
    const password = document.getElementById('auth-password').value;
    const inviteCode = document.getElementById('auth-invite').value.trim();
    const website = document.getElementById('auth-website').value;  // honeypot

    if (!username || !password) {
        errEl.textContent = 'Username and password are required';
        errEl.style.display = 'block';
        return;
    }

    const url = authMode === 'register' ? '/auth/register' : '/auth/login';
    const body = { username, password, website };
    if (authMode === 'register') body.invite_code = inviteCode;

    try {
        const resp = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: 'Request failed' }));
            errEl.textContent = err.detail || 'Request failed';
            errEl.style.display = 'block';
            return;
        }
        const data = await resp.json();
        setAuthState(data.token, data.user);
        closeAuthModal();
        showToast('Welcome, ' + data.user.username + '!');
        // Reload models for the active page
        if (typeof loadModels === 'function') loadModels();
        if (typeof loadTutorModels === 'function') loadTutorModels();
        if (typeof loadVoices === 'function') loadVoices();
    } catch (e) {
        errEl.textContent = 'Connection error';
        errEl.style.display = 'block';
    }
}

function logout() {
    setAuthState(null, null);
    if (typeof chatHistory !== 'undefined') chatHistory = [];
    if (typeof currentSessionId !== 'undefined') currentSessionId = null;
    if (typeof currentSessionName !== 'undefined') currentSessionName = null;
    if (typeof chat !== 'undefined' && chat) chat.innerHTML = '';
    // Turn off dialog mode if active
    if (dialogMode) {
        dialogMode = false;
        const btn = document.getElementById('dialog-toggle');
        btn.classList.remove('active');
        btn.textContent = '🗣️ Dialog';
    }
    // Clear attached files
    if (typeof clearSessionFiles === 'function') clearSessionFiles();
    showToast('Logged out');
    openAuthModal('login');
}

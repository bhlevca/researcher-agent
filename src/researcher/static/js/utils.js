// ── Utilities ──
let _toastTimer = null;

function showToast(msg) {
    const t = document.getElementById('toast');
    clearTimeout(_toastTimer);
    t.innerHTML = '';
    const span = document.createElement('span');
    span.textContent = msg;
    t.appendChild(span);
    const btn = document.createElement('button');
    btn.className = 'toast-close';
    btn.innerHTML = '✕';
    btn.onclick = () => { clearTimeout(_toastTimer); t.classList.remove('show'); };
    t.appendChild(btn);
    t.classList.add('show');
}

function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

// Simple markdown rendering: code blocks, inline code, images, links, bold
function renderContent(text) {
    text = text.replace(/```(\w*)\n?([\s\S]*?)```/g,
        (_, lang, code) => '<pre><code>' + escapeHtml(code.trim()) + '</code></pre>');
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    text = text.replace(/!\[([^\]]*)\]\(([^)]+)\)/g,
        '<img src="$2" alt="$1" loading="lazy">');
    text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener">$1</a>');
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Bare image URLs (http and local /static/ paths)
    text = text.replace(/(^|[\s>])(https?:\/\/\S+\.(?:png|jpg|jpeg|gif|webp|svg))([\s<]|$)/gi,
        '$1<img src="$2" loading="lazy">$3');
    text = text.replace(/(^|[\s>])(\/static\/\S+\.(?:png|jpg|jpeg|gif|webp|svg))([\s<]|$)/gi,
        '$1<img src="$2" loading="lazy">$3');
    // Bare video paths — the tool returns a full <video> tag, but the model
    // sometimes echoes just the path.
    text = text.replace(/(^|[\s>])(\/static\/\S+\.(?:mp4|webm))([\s<]|$)/gi,
        '$1<video src="$2" controls loop muted playsinline style="max-width:100%;border-radius:0.5rem;"></video>$3');
    text = text.replace(/\n{2,}/g, '</p><p>');
    text = text.replace(/\n/g, '<br>');
    return '<p>' + text + '</p>';
}

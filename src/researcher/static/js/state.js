// ── Global state & DOM references ──
const chat = document.getElementById('chat-container');
const input = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const cancelBtn = document.getElementById('cancel-btn');
const modelSelect = document.getElementById('model-select');
const modelInfo = document.getElementById('model-info');

// ── Conversation state ──
let chatHistory = [];
let currentSessionId = null;
let currentSessionName = null;

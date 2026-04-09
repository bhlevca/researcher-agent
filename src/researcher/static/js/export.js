// ── Export ──
async function exportResponse(content, format) {
    try {
        const resp = await fetch('/export', {
            method: 'POST',
            headers: jsonAuthHeaders(),
            body: JSON.stringify({
                content: content,
                format: format,
                filename: 'response-' + new Date().toISOString().slice(0, 10),
            }),
        });
        if (!resp.ok) {
            showToast('Export failed');
            return;
        }
        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const ext = format === 'docx' ? 'docx' : format === 'xlsx' ? 'xlsx' : format === 'md' ? 'md' : 'txt';
        a.download = 'response-' + new Date().toISOString().slice(0, 10) + '.' + ext;
        a.click();
        URL.revokeObjectURL(url);
        showToast('Exported as ' + ext.toUpperCase());
    } catch (e) {
        showToast('Export error: ' + e.message);
    }
}

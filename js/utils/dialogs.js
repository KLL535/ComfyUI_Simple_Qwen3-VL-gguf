// js/utils/dialogs.js

// =========================================================================
// showConfirmDialog
// =========================================================================

export function showConfirmDialog(title, message, onConfirm) {
    const overlay = document.createElement('div');
    overlay.style.cssText =
        'position:fixed;top:0;left:0;width:100%;height:100%;' +
        'background:rgba(0,0,0,0.55);z-index:10000;' +
        'display:flex;align-items:center;justify-content:center;';

    const dialog = document.createElement('div');
    dialog.style.cssText =
        'background:var(--comfy-menu-bg,#2a2a2a);' +
        'border:1px solid var(--border-color,#555);' +
        'border-radius:8px;padding:20px;' +
        'min-width:400px;max-width:600px;' +
        'box-shadow:0 4px 24px rgba(0,0,0,0.6);' +
        'font-family:sans-serif;';

    const titleEl = document.createElement('h3');
    titleEl.textContent = title;
    titleEl.style.cssText =
        'margin:0 0 14px 0;' +
        'color:var(--input-text,#fff);' +
        'font-size:14px;font-weight:600;';

    const msgEl = document.createElement('div');
    msgEl.textContent = message;
    msgEl.style.cssText =
        'color:var(--input-text,#e0e0e0);' +
        'font-size:13px;line-height:1.45;' +
        'word-break:break-word;';

    const buttonRow = document.createElement('div');
    buttonRow.style.cssText =
        'margin-top:14px;display:flex;gap:8px;justify-content:flex-end;';

    const baseBtnStyle =
        'height:28px;padding:0 16px;' +
        'background:var(--comfy-input-bg,#333);' +
        'color:var(--input-text,#e0e0e0);' +
        'border:1px solid var(--border-color,#555);' +
        'border-radius:4px;cursor:pointer;' +
        'font-size:12px;font-family:sans-serif;' +
        'display:inline-flex;align-items:center;justify-content:center;' +
        'line-height:1;white-space:nowrap;outline:none;';

    const makeButton = (label, onClick, danger=false) => {
        const btn = document.createElement('button');
        btn.textContent = label;
        btn.style.cssText = baseBtnStyle;
        btn.addEventListener('mouseenter', () => {
            btn.style.background = danger ? '#e74c3c' : '#4a90e2';
            btn.style.color = '#fff';
            btn.style.borderColor = danger ? '#e74c3c' : '#4a90e2';
        });
        btn.addEventListener('mouseleave', () => {
            btn.style.background = 'var(--comfy-input-bg,#333)';
            btn.style.color = 'var(--input-text,#e0e0e0)';
            btn.style.borderColor = 'var(--border-color,#555)';
        });
        btn.addEventListener('click', (e) => { e.stopPropagation(); onClick(); });
        return btn;
    };

    const closeDialog = () => {
        if (document.body.contains(overlay)) document.body.removeChild(overlay);
    };

    const submit = () => {
        closeDialog();
        onConfirm();
    };

    buttonRow.appendChild(makeButton('✓ OK', submit, true));
    buttonRow.appendChild(makeButton('✗ Cancel', closeDialog));

    dialog.appendChild(titleEl);
    dialog.appendChild(msgEl);
    dialog.appendChild(buttonRow);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    // фокус на Cancel — безопаснее по умолчанию для деструктивного действия
    const cancelBtn = buttonRow.querySelector('button');
    if (cancelBtn) cancelBtn.focus();

    overlay.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); submit(); }
        else if (e.key === 'Escape') { e.preventDefault(); closeDialog(); }
    });

    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) closeDialog();
    });
}

// =========================================================================
// PromptDialog
// =========================================================================

export function showPromptDialog(title, defaultValue, onConfirm) {
    const overlay = document.createElement('div');
    overlay.style.cssText =
        'position:fixed;top:0;left:0;width:100%;height:100%;' +
        'background:rgba(0,0,0,0.55);z-index:10000;' +
        'display:flex;align-items:center;justify-content:center;';

    const dialog = document.createElement('div');
    dialog.style.cssText =
        'background:var(--comfy-menu-bg,#2a2a2a);' +
        'border:1px solid var(--border-color,#555);' +
        'border-radius:8px;padding:20px;' +
        'min-width:400px;max-width:600px;' +
        'box-shadow:0 4px 24px rgba(0,0,0,0.6);' +
        'font-family:sans-serif;';

    const titleEl = document.createElement('h3');
    titleEl.textContent = title;
    titleEl.style.cssText =
        'margin:0 0 14px 0;' +
        'color:var(--input-text,#fff);' +
        'font-size:14px;font-weight:600;';

    const input = document.createElement('input');
    input.type = 'text';
    input.value = defaultValue || '';
    input.style.cssText =
        'width:100%;height:32px;' +
        'background:var(--comfy-input-bg,#1a1a1a);' +
        'color:var(--input-text,#e0e0e0);' +
        'border:1px solid var(--border-color,#444);' +
        'border-radius:4px;padding:0 10px;' +
        'font-size:13px;box-sizing:border-box;outline:none;';
    input.addEventListener('focus', () => { input.style.borderColor = '#4a90e2'; });
    input.addEventListener('blur', () => { input.style.borderColor = 'var(--border-color,#444)'; });

    const buttonRow = document.createElement('div');
    buttonRow.style.cssText =
        'margin-top:14px;display:flex;gap:8px;justify-content:flex-end;';

    const baseBtnStyle =
        'height:28px;padding:0 16px;' +
        'background:var(--comfy-input-bg,#333);' +
        'color:var(--input-text,#e0e0e0);' +
        'border:1px solid var(--border-color,#555);' +
        'border-radius:4px;cursor:pointer;' +
        'font-size:12px;font-family:sans-serif;' +
        'display:inline-flex;align-items:center;justify-content:center;' +
        'line-height:1;white-space:nowrap;outline:none;';

    const makeButton = (label, onClick) => {
        const btn = document.createElement('button');
        btn.textContent = label;
        btn.style.cssText = baseBtnStyle;
        btn.addEventListener('mouseenter', () => {
            btn.style.background = '#4a90e2';
            btn.style.color = '#fff';
            btn.style.borderColor = '#4a90e2';
        });
        btn.addEventListener('mouseleave', () => {
            btn.style.background = 'var(--comfy-input-bg,#333)';
            btn.style.color = 'var(--input-text,#e0e0e0)';
            btn.style.borderColor = 'var(--border-color,#555)';
        });
        btn.addEventListener('click', (e) => { e.stopPropagation(); onClick(); });
        return btn;
    };

    const closeDialog = () => {
        if (document.body.contains(overlay)) document.body.removeChild(overlay);
    };

    const submit = () => {
        const val = input.value.trim();
        if (!val) return;
        closeDialog();
        onConfirm(val);
    };

    buttonRow.appendChild(makeButton('✓ OK', submit));
    buttonRow.appendChild(makeButton('✗ Cancel', closeDialog));

    dialog.appendChild(titleEl);
    dialog.appendChild(input);
    dialog.appendChild(buttonRow);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    input.focus();
    input.select();

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); submit(); }
        else if (e.key === 'Escape') { e.preventDefault(); closeDialog(); }
    });

    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) closeDialog();
    });
}
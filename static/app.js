/* ── Modals ──────────────────────────────────────────────────────────────────── */
function openModal(id) {
  document.getElementById(id).classList.add('open');
}
function closeModal(id) {
  document.getElementById(id).classList.remove('open');
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    document.querySelectorAll('.modal-overlay.open')
      .forEach(el => el.classList.remove('open'));
  }
});

/* ── Dropzone ────────────────────────────────────────────────────────────────── */
const dropzone  = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const fileList  = document.getElementById('file-list');
const uploadBtn = document.getElementById('upload-btn');

if (dropzone) {
  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', e => {
    e.preventDefault();
    dropzone.classList.add('over');
  });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('over'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('over');
    fileInput.files = e.dataTransfer.files;
    renderFileList(e.dataTransfer.files);
  });

  fileInput.addEventListener('change', () => renderFileList(fileInput.files));
}

function renderFileList(files) {
  if (!fileList) return;
  fileList.innerHTML = '';
  for (const f of files) {
    const item = document.createElement('div');
    item.className = 'file-item';
    item.innerHTML = `<span>${f.name}</span><span>${(f.size / 1024 / 1024).toFixed(1)} MB</span>`;
    fileList.appendChild(item);
  }
  if (uploadBtn) uploadBtn.disabled = files.length === 0;
}

/* ── Transcription ───────────────────────────────────────────────────────────── */
function transcribeClip(projectId, clipId) {
  fetch(`/projects/${projectId}/clips/${clipId}/transcribe`, { method: 'POST' })
    .then(r => r.json())
    .then(() => {
      updateBadge(clipId, 'transcribing');
      updateActions(projectId, clipId, 'transcribing');
      startPolling(projectId, clipId);
    });
}

function transcribeAll(projectId) {
  fetch(`/projects/${projectId}/transcribe-all`, { method: 'POST' })
    .then(() => location.reload());
}

/* ── Export ──────────────────────────────────────────────────────────────────── */
function openExportModal(projectId, clipId, clipName) {
  document.getElementById('export-clip-id').value = clipId;
  document.getElementById('export-clip-name').textContent = clipName;
  openModal('export-modal');
}

function submitExport(projectId) {
  const clipId    = document.getElementById('export-clip-id').value;
  const styleEl   = document.querySelector('#export-form input[name="style"]:checked');
  const style     = styleEl ? styleEl.value : '1';

  const body = new FormData();
  body.append('style', style);

  fetch(`/projects/${projectId}/clips/${clipId}/export`, { method: 'POST', body })
    .then(r => r.json())
    .then(() => {
      closeModal('export-modal');
      updateBadge(clipId, 'exporting');
      updateActions(projectId, clipId, 'exporting');
      startPolling(projectId, clipId);
    });
}

/* ── Status polling ──────────────────────────────────────────────────────────── */
const activePolls = {};

function startPolling(projectId, clipId) {
  if (activePolls[clipId]) return;
  activePolls[clipId] = setInterval(() => {
    fetch(`/projects/${projectId}/clips/${clipId}/status`)
      .then(r => r.json())
      .then(data => {
        updateBadge(clipId, data.status);
        const done = ['transcribed', 'exported', 'error'].includes(data.status);
        if (done) {
          clearInterval(activePolls[clipId]);
          delete activePolls[clipId];
          updateActions(projectId, clipId, data.status);
        }
      });
  }, 3000);
}

function updateBadge(clipId, status) {
  const badge = document.getElementById(`status-badge-${clipId}`);
  if (!badge) return;
  badge.className = `badge badge-${status}`;
  badge.textContent = status;
}

function updateActions(projectId, clipId, status) {
  const actionsCell = document.querySelector(`#clip-row-${clipId} .clip-actions`);
  if (!actionsCell) return;

  const filename = document.querySelector(`#clip-row-${clipId} .clip-name`)?.textContent || '';

  if (status === 'transcribing' || status === 'exporting') {
    actionsCell.innerHTML = `<span class="spinner-text">${status === 'transcribing' ? 'Transcribing' : 'Exporting'}…</span>`;
  } else if (status === 'transcribed') {
    actionsCell.innerHTML = `
      <a href="/projects/${projectId}/clips/${clipId}/editor" class="btn btn-sm btn-ghost">Edit Transcript</a>
      <button class="btn btn-sm btn-ghost"
        onclick="openExportModal(${projectId}, ${clipId}, '${filename.trim()}')">Export</button>
    `;
  } else if (status === 'exported') {
    actionsCell.innerHTML = `
      <a href="/projects/${projectId}/clips/${clipId}/editor" class="btn btn-sm btn-ghost">Edit Transcript</a>
      <button class="btn btn-sm btn-ghost"
        onclick="openExportModal(${projectId}, ${clipId}, '${filename.trim()}')">Export</button>
      <a href="/projects/${projectId}/clips/${clipId}/download" class="btn btn-sm btn-primary">Download</a>
    `;
  } else if (status === 'error') {
    actionsCell.innerHTML = `
      <button class="btn btn-sm btn-ghost" onclick="transcribeClip(${projectId}, ${clipId})">Retry</button>
    `;
  }
}

// Auto-start polling for in-progress clips on page load
document.querySelectorAll('[data-clip-id]').forEach(row => {
  const badge = row.querySelector('.badge');
  if (!badge) return;
  const status = badge.textContent.trim();
  if (status === 'transcribing' || status === 'exporting') {
    const projectId = parseInt(row.dataset.projectId);
    const clipId    = parseInt(row.dataset.clipId);
    startPolling(projectId, clipId);
  }
});

/* ── Transcript editor ───────────────────────────────────────────────────────── */
function seekVideo(seconds) {
  const video = document.getElementById('video-player');
  if (video) video.currentTime = seconds;
}

function saveTranscript() {
  const textareas = document.querySelectorAll('.segment-text');
  const segments  = [];

  textareas.forEach(ta => {
    segments.push({
      start: parseFloat(ta.dataset.start),
      end:   parseFloat(ta.dataset.end),
      text:  ta.value.trim(),
    });
  });

  const status = document.getElementById('save-status');
  if (status) status.textContent = 'Saving…';

  fetch(`/projects/${projectId}/clips/${clipId}/save`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ segments }),
  })
  .then(r => r.json())
  .then(() => {
    if (status) {
      status.textContent = 'Saved';
      setTimeout(() => { status.textContent = ''; }, 2000);
    }
  })
  .catch(() => {
    if (status) status.textContent = 'Error saving';
  });
}

// Ctrl+S / Cmd+S to save
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault();
    if (typeof saveTranscript === 'function' && document.getElementById('segments-container')) {
      saveTranscript();
    }
  }
});

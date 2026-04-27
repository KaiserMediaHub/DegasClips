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

/* ── Dropzone & Upload Queue ─────────────────────────────────────────────────── */
const CHUNK_SIZE = 50 * 1024 * 1024; // 50 MB

const dropzone  = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const fileList  = document.getElementById('file-list');
const uploadBtn = document.getElementById('upload-btn');

let queuedFiles = [];

if (dropzone) {
  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('over'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('over'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault();
    dropzone.classList.remove('over');
    setFiles(e.dataTransfer.files);
  });
  fileInput.addEventListener('change', () => setFiles(fileInput.files));
}

function setFiles(files) {
  queuedFiles = Array.from(files);
  renderFileList(queuedFiles);
}

function renderFileList(files) {
  if (!fileList) return;
  fileList.innerHTML = '';
  files.forEach((f, i) => {
    const item = document.createElement('div');
    item.className = 'file-item';
    item.id = `file-item-${i}`;
    item.innerHTML = `
      <span class="file-item-name">${f.name}</span>
      <span class="file-item-size">${(f.size / 1024 / 1024).toFixed(1)} MB</span>
      <span class="file-item-status" id="file-status-${i}">Queued</span>
    `;
    fileList.appendChild(item);
  });
  if (uploadBtn) uploadBtn.disabled = files.length === 0;
}

// Called by the Upload button in the modal
async function startUploadQueue(projectId) {
  if (!queuedFiles.length) return;

  uploadBtn.disabled = true;
  uploadBtn.textContent = 'Uploading…';

  const overall = document.getElementById('upload-overall');
  if (overall) overall.style.display = 'block';

  for (let i = 0; i < queuedFiles.length; i++) {
    const file = queuedFiles[i];
    const statusEl = document.getElementById(`file-status-${i}`);
    const overallEl = document.getElementById('upload-overall-text');

    if (overallEl) overallEl.textContent = `Uploading ${i + 1} of ${queuedFiles.length}: ${file.name}`;
    if (statusEl) statusEl.textContent = 'Uploading…';

    try {
      await uploadFileInChunks(file, projectId, (pct) => {
        if (statusEl) statusEl.textContent = `${pct}%`;
      });
      if (statusEl) statusEl.textContent = '✓ Done';
    } catch (err) {
      if (statusEl) statusEl.textContent = '✗ Failed';
      console.error('Upload failed for', file.name, err);
    }
  }

  if (overallEl) overallEl.textContent = 'All files uploaded! Reloading…';
  setTimeout(() => {
    closeModal('upload-modal');
    location.reload();
  }, 1200);
}

async function uploadFileInChunks(file, projectId, onProgress) {
  const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
  const fileUid = crypto.randomUUID();

  for (let i = 0; i < totalChunks; i++) {
    const start = i * CHUNK_SIZE;
    const end   = Math.min(start + CHUNK_SIZE, file.size);
    const chunk = file.slice(start, end);

    const form = new FormData();
    form.append('file_uid',     fileUid);
    form.append('chunk_index',  i);
    form.append('total_chunks', totalChunks);
    form.append('filename',     file.name);
    form.append('data',         chunk);

    const res = await fetch(`/projects/${projectId}/upload/chunk`, {
      method: 'POST',
      body: form,
    });

    if (!res.ok) throw new Error(`Chunk ${i} failed: ${res.status}`);

    const pct = Math.round(((i + 1) / totalChunks) * 100);
    onProgress(pct);
  }
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

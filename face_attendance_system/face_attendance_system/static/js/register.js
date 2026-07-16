const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ring = document.getElementById('ring');
const captureStatus = document.getElementById('capture-status');
const trainStatus = document.getElementById('train-status');
const captureBtn = document.getElementById('capture-btn');
const createUserBtn = document.getElementById('create-user-btn');
const trainBtn = document.getElementById('train-btn');

let currentUserId = null;
let stream = null;
const TARGET_SAMPLES = 30;

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 } });
        video.srcObject = stream;
    } catch (err) {
        captureStatus.textContent = 'Camera access denied: ' + err.message;
    }
}
startCamera();

function grabFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    return canvas.toDataURL('image/jpeg', 0.85);
}

createUserBtn.addEventListener('click', async () => {
    const name = document.getElementById('name').value.trim();
    const roll_no = document.getElementById('roll_no').value.trim();
    if (!name) {
        alert('Enter a name first.');
        return;
    }
    createUserBtn.disabled = true;
    const res = await fetch('/api/users', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, roll_no })
    });
    const data = await res.json();
    createUserBtn.disabled = false;
    if (!data.success) {
        alert(data.error || 'Could not create user.');
        return;
    }
    currentUserId = data.user_id;
    captureBtn.disabled = false;
    captureStatus.textContent = `User "${name}" created. Ready to capture.`;
    addUserChip(data.user_id, name, roll_no);
});

captureBtn.addEventListener('click', async () => {
    if (!currentUserId) return;
    captureBtn.disabled = true;
    let count = 0;
    ring.classList.add('pulse');

    while (count < TARGET_SAMPLES) {
        const frame = grabFrame();
        const res = await fetch('/api/register/capture', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: currentUserId, frame })
        });
        const data = await res.json();
        if (data.success) {
            count = data.sample_count;
            captureStatus.textContent = `Captured ${count} / ${TARGET_SAMPLES} samples. Move your head slightly between shots.`;
            ring.classList.remove('danger');
            ring.classList.add('ok');
        } else {
            captureStatus.textContent = data.error || 'No face detected — center your face.';
            ring.classList.remove('ok');
            ring.classList.add('danger');
        }
        await new Promise(r => setTimeout(r, 250));
    }

    ring.classList.remove('pulse', 'ok', 'danger');
    captureStatus.textContent = `Done. ${count} samples captured — you can train the model now.`;
    captureBtn.disabled = false;
});

trainBtn.addEventListener('click', async () => {
    trainBtn.disabled = true;
    trainStatus.textContent = 'Training model — this can take a minute…';
    const res = await fetch('/api/register/train', { method: 'POST' });
    const data = await res.json();
    trainBtn.disabled = false;
    if (data.success) {
        trainStatus.textContent = `Trained on ${data.num_classes} users, ${data.num_samples} samples. Validation accuracy: ${(data.val_accuracy * 100).toFixed(1)}%.`;
    } else {
        trainStatus.textContent = data.error;
    }
});

function addUserChip(id, name, roll_no) {
    const list = document.getElementById('user-list');
    const empty = list.querySelector('.empty-state');
    if (empty) empty.remove();
    const chip = document.createElement('div');
    chip.className = 'user-chip';
    chip.dataset.userId = id;
    chip.innerHTML = `
        <div>
            <div>${name}</div>
            <div class="meta">${roll_no || 'no roll no.'} · 0 samples · not trained</div>
        </div>
        <button class="btn danger delete-user-btn" data-id="${id}" style="padding: 6px 12px; font-size: 12px;">Remove</button>
    `;
    list.appendChild(chip);
    chip.querySelector('.delete-user-btn').addEventListener('click', () => deleteUser(id, chip));
}

document.querySelectorAll('.delete-user-btn').forEach(btn => {
    btn.addEventListener('click', () => deleteUser(btn.dataset.id, btn.closest('.user-chip')));
});

async function deleteUser(id, chipEl) {
    if (!confirm('Remove this user? This does not remove their captured images on disk.')) return;
    const res = await fetch(`/api/users/${id}`, { method: 'DELETE' });
    const data = await res.json();
    if (data.success) chipEl.remove();
}

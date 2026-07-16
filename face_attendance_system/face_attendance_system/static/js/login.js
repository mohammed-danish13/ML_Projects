const video = document.getElementById('video');
const ring = document.getElementById('ring');
const scanStatus = document.getElementById('scan-status');
const resultCard = document.getElementById('result-card');
const resultEyebrow = document.getElementById('result-eyebrow');
const resultName = document.getElementById('result-name');
const resultMeta = document.getElementById('result-meta');
const resultBadge = document.getElementById('result-badge');

let scanning = true;
let lastRecognizedId = null;
let cooldownUntil = 0;

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 } });
        video.srcObject = stream;
        scanStatus.textContent = 'Scanning for a face…';
        loop();
    } catch (err) {
        scanStatus.textContent = 'Camera access denied: ' + err.message;
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

async function loop() {
    while (scanning) {
        if (Date.now() < cooldownUntil) {
            await new Promise(r => setTimeout(r, 400));
            continue;
        }
        const frame = grabFrame();
        try {
            const res = await fetch('/api/attendance/recognize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frame })
            });
            const data = await res.json();
            handleResult(data);
        } catch (e) {
            scanStatus.textContent = 'Connection error — retrying…';
        }
        await new Promise(r => setTimeout(r, 700));
    }
}

function handleResult(data) {
    if (data.success) {
        showRecognized(data);
        cooldownUntil = Date.now() + 4000; // pause scanning briefly after a hit
        return;
    }
    ring.classList.remove('ok', 'warn');
    switch (data.error) {
        case 'no_face':
            scanStatus.textContent = 'No face in frame — step closer to the camera.';
            break;
        case 'not_recognized':
            scanStatus.textContent = `Face not recognized (confidence ${(data.confidence * 100).toFixed(0)}%). Try register if you're new.`;
            ring.classList.add('warn');
            break;
        case 'not_trained':
            scanStatus.textContent = 'Model not trained yet. Ask an admin to register and train users first.';
            break;
        default:
            scanStatus.textContent = 'Scanning for a face…';
    }
}

function showRecognized(data) {
    scanStatus.textContent = data.already_marked
        ? `Welcome back, ${data.name} — already checked in today at ${data.time}.`
        : `Checked in — ${data.name} at ${data.time}.`;

    ring.classList.remove('warn');
    ring.classList.add(data.status === 'ON-TIME' ? 'ok' : 'warn');

    resultCard.style.display = 'block';
    resultEyebrow.textContent = data.already_marked ? 'Already Checked In' : 'Check-In Recorded';
    resultName.textContent = data.name;
    resultMeta.textContent = `${data.roll_no || 'no roll no.'} · ${data.time} · confidence ${(data.confidence * 100).toFixed(0)}%`;
    resultBadge.textContent = data.status;
    resultBadge.className = 'badge ' + (data.status === 'ON-TIME' ? 'ok' : 'warn');
}

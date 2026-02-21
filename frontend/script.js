const video = document.getElementById("camera");
const canvas = document.getElementById("frameCanvas");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusPill = document.getElementById("statusPill");
const statusText = document.getElementById("statusText");
const overlayText = document.getElementById("overlayText");
const expressionLabel = document.getElementById("expressionLabel");
const expressionConfidence = document.getElementById("expressionConfidence");
const confidenceBar = document.getElementById("confidenceBar");
const logEl = document.getElementById("log");

let stream = null;
let intervalId = null;

const BACKEND_URL = "http://localhost:8000/api/predict-expression";

function log(msg) {
    const p = document.createElement("p");
    p.textContent = msg;
    logEl.prepend(p);
}

function setStatus(text, ready=false) {
    statusText.textContent = text;
    overlayText.textContent = text;
    if (ready) statusPill.classList.add("ready");
    else statusPill.classList.remove("ready");
}

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        setStatus("Camera on, analyzing...", true);

        intervalId = setInterval(captureAndSend, 800);
    } catch (err) {
        log("Camera error: " + err.message);
    }
}

function stopCamera() {
    if (intervalId) clearInterval(intervalId);
    if (stream) stream.getTracks().forEach(t => t.stop());

    video.srcObject = null;
    setStatus("Camera off", false);
}

function captureAndSend() {
    if (!video.videoWidth) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        try {
            const res = await fetch(BACKEND_URL, {
                method: "POST",
                body: formData
            });

            const data = await res.json();
            updateUI(data.expression, data.confidence);

        } catch (err) {
            log("Network error: Failed to fetch");
        }
    }, "image/jpeg");
}

function updateUI(expression, confidence) {
    const percent = Math.round(confidence * 100);
    expressionLabel.textContent = expression.toUpperCase();
    expressionConfidence.textContent = "Confidence: " + percent + "%";
    confidenceBar.style.width = percent + "%";
}

startBtn.onclick = startCamera;
stopBtn.onclick = stopCamera;
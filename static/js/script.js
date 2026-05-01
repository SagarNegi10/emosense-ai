const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");

const emoji = document.getElementById("emoji");
const emotion = document.getElementById("emotion");
const confidence = document.getElementById("confidence");
const autoBtn = document.getElementById("autoBtn");
const detectBtn = document.getElementById("detectBtn");
const confidenceBar = document.getElementById("confidenceBar");
const statusText = document.getElementById("statusText");
const statusMessage = document.getElementById("statusMessage");
const cameraStatus = document.getElementById("cameraStatus");
const processingBadge = document.getElementById("processingBadge");
const topPredictions = document.getElementById("topPredictions");

let autoMode = false;
let interval = null;
let inFlight = false;
let cameraReady = false;
let pausedForVisibility = false;
let activeController = null;

const LIVE_INTERVAL_MS = 1000;

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

detectBtn.disabled = true;
statusMessage.innerText = "Waiting for camera...";

function setStatus(type, text) {
    cameraStatus.classList.remove("success", "error");
    if (type) {
        cameraStatus.classList.add(type);
    }
    statusText.innerText = text;
}

function updateAutoButton() {
    autoBtn.classList.toggle("stop", autoMode);
    autoBtn.innerHTML = autoMode
        ? '<span class="btn-icon">■</span><span class="btn-text">Stop Live Detection</span>'
        : '<span class="btn-icon">▶</span><span class="btn-text">Start Live Detection</span>';
}

function syncCanvasSize() {
    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;
    overlay.width = w;
    overlay.height = h;
    captureCanvas.width = w;
    captureCanvas.height = h;
}

navigator.mediaDevices.getUserMedia({
    video: {
        width: 640,
        height: 480,
        facingMode: "user"
    },
    audio: false
})
.then(stream => {
    video.srcObject = stream;
    setStatus("success", "Camera ready");
    statusMessage.innerText = "Ready to detect";
    cameraReady = true;
    detectBtn.disabled = false;
    video.addEventListener("loadedmetadata", syncCanvasSize, { once: true });
    window.addEventListener("resize", syncCanvasSize);
    updateAutoButton();
})
.catch(() => {
    setStatus("error", "Camera blocked");
    statusMessage.innerText = "Enable camera permissions";
});

function detectEmotion(){
    if(inFlight || !cameraReady) return;

    inFlight = true;
    processingBadge.classList.add("active");
    statusMessage.innerText = "Analyzing...";

    syncCanvasSize();
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

    captureCanvas.toBlob(blob => {
        if (!blob) {
            statusMessage.innerText = "Capture failed";
            inFlight = false;
            processingBadge.classList.remove("active");
            return;
        }

        const formData = new FormData();
        formData.append("image", blob, "frame.jpg");

        if (activeController) {
            activeController.abort();
        }
        activeController = new AbortController();

        fetch("/predict", {
            method: "POST",
            body: formData,
            signal: activeController.signal
        })
        .then(res => {
            if (!res.ok) {
                throw new Error("Prediction request failed");
            }
            return res.json();
        })
        .then(data => {

            ctx.clearRect(0, 0, overlay.width, overlay.height);

            if(data.success){
                emoji.innerText = data.emoji;
                emotion.innerText = data.emotion;
                confidence.innerText = data.confidence + "%";
                confidenceBar.style.width = data.confidence + "%";
                statusMessage.innerText = "Face detected";
                renderTopPredictions(data.top_predictions || []);

                if (data.box) {
                    const scaleX = overlay.width / captureCanvas.width;
                    const scaleY = overlay.height / captureCanvas.height;
                    ctx.strokeStyle = "#22c55e";
                    ctx.lineWidth = 3;
                    ctx.strokeRect(
                        data.box.x * scaleX,
                        data.box.y * scaleY,
                        data.box.w * scaleX,
                        data.box.h * scaleY
                    );
                }
            } else {
                emoji.innerText = "😕";
                emotion.innerText = "No Face";
                confidence.innerText = "";
                confidenceBar.style.width = "0%";
                renderTopPredictions([]);
                statusMessage.innerText = "Adjust position / lighting";
            }
        })
        .catch(err => {
            if (err.name === "AbortError") {
                return;
            }
            statusMessage.innerText = "Prediction error";
        })
        .finally(() => {
            inFlight = false;
            processingBadge.classList.remove("active");
            activeController = null;
        });

    }, "image/jpeg", 0.75);

}

function toggleAuto(){
    if(!cameraReady) return;

    autoMode = !autoMode;
    updateAutoButton();

    if(autoMode){
        interval = setInterval(detectEmotion, LIVE_INTERVAL_MS);
    } else {
        clearInterval(interval);
        interval = null;
    }
}

function renderTopPredictions(preds) {
    if (!preds.length) {
        topPredictions.innerHTML = "";
        return;
    }

    topPredictions.innerHTML = preds.map(item => `
        <div class="top-row">
            <span class="top-label">${item.emoji} ${item.label}</span>
            <span class="top-value">${item.confidence}%</span>
        </div>
    `).join("");
}

document.addEventListener("visibilitychange", () => {
    if (document.hidden && autoMode) {
        pausedForVisibility = true;
        toggleAuto();
        statusMessage.innerText = "Live mode paused (tab hidden)";
    } else if (!document.hidden && pausedForVisibility) {
        pausedForVisibility = false;
        toggleAuto();
        statusMessage.innerText = "Live mode resumed";
    }
});
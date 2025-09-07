$(document).ready(function() {
    // --- Canvas Setup ---
    let canvas = document.getElementById("canvas");
    let ctx = canvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    let drawing = false;
    let timeout;

    // --- Camera Setup ---
    let video = document.getElementById("video-feed");
    let snapshotCanvas = document.getElementById("snapshot-canvas");
    let stream;

    // --- DOM Element Selectors ---
    const drawArea = $('#draw-area');
    const cameraArea = $('#camera-area');
    const drawModeBtn = $('#draw-mode-btn');
    const cameraModeBtn = $('#camera-mode-btn');

    // --- Mode Switching ---
    drawModeBtn.click(function() {
        cameraArea.addClass('d-none');
        drawArea.removeClass('d-none');
        drawModeBtn.addClass('active btn-primary').removeClass('btn-outline-primary');
        cameraModeBtn.removeClass('active btn-primary').addClass('btn-outline-primary');
        stopCamera();
    });

    cameraModeBtn.click(function() {
        drawArea.addClass('d-none');
        cameraArea.removeClass('d-none');
        cameraModeBtn.addClass('active btn-primary').removeClass('btn-outline-primary');
        drawModeBtn.removeClass('active btn-primary').addClass('btn-outline-primary');
        startCamera();
    });

    // --- Camera Functions ---
    async function startCamera() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                video.play();
            } catch (err) {
                console.error("Error accessing camera: ", err);
                alert("Could not access the camera. Please grant permission and try again.");
            }
        } else {
            alert("Your browser does not support camera access.");
        }
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
    }

    // --- Core Drawing Functionality ---
    function draw(x, y) {
        if (!drawing) return;
        clearTimeout(timeout);
        ctx.fillStyle = "white";
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, 2 * Math.PI);
        ctx.fill();
    }

    // --- Event Listeners for Drawing Canvas ---
    canvas.addEventListener("mousedown", () => drawing = true);
    canvas.addEventListener("mouseup", () => {
        drawing = false;
        clearTimeout(timeout);
        timeout = setTimeout(predictFromCanvas, 500);
    });
    canvas.addEventListener("mousemove", (e) => draw(e.offsetX, e.offsetY));
    canvas.addEventListener("touchstart", (e) => { e.preventDefault(); drawing = true; let t = e.touches[0]; draw(t.clientX-canvas.offsetLeft, t.clientY-canvas.offsetTop); }, { passive: false });
    canvas.addEventListener("touchend", (e) => { e.preventDefault(); drawing = false; clearTimeout(timeout); timeout = setTimeout(predictFromCanvas, 500); }, { passive: false });
    canvas.addEventListener("touchmove", (e) => { e.preventDefault(); if (!drawing) return; let t = e.touches[0]; draw(t.clientX-canvas.offsetLeft, t.clientY-canvas.offsetTop); }, { passive: false });

    // --- Prediction Logic ---
    function predictFromCanvas() {
        sendPredictionRequest(canvas.toDataURL("image/png"));
    }

    function predictFromCamera() {
        let snapshotCtx = snapshotCanvas.getContext('2d');
        // The video is mirrored, so we must flip the canvas when drawing the snapshot
        snapshotCtx.translate(snapshotCanvas.width, 0);
        snapshotCtx.scale(-1, 1);
        snapshotCtx.drawImage(video, 0, 0, snapshotCanvas.width, snapshotCanvas.height);
        // Reset the transformation
        snapshotCtx.setTransform(1, 0, 0, 1, 0, 0);
        sendPredictionRequest(snapshotCanvas.toDataURL("image/png"));
    }
    
    // Reusable AJAX function
    function sendPredictionRequest(imageDataURL) {
        $.ajax({
            type: "POST",
            url: "/predict",
            contentType: "application/json",
            data: JSON.stringify({ image: imageDataURL }),
            success: function(res) {
                $("#result").text(res.digit);
                $("#prob").attr("src", "/static/" + res.prob_img + "?t=" + new Date().getTime());
                $('#result-placeholder').addClass('d-none');
                $('#result-display').removeClass('d-none');
                confetti({ particleCount: 100, spread: 70, origin: { y: 0.6 } });
            },
            error: function() {
                alert("Prediction request failed. Please try again.");
            }
        });
    }

    // --- Button Click Handlers ---
    $("#clear").click(() => {
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        $('#result-placeholder').removeClass('d-none');
        $('#result-display').addClass('d-none');
        $("#result").text("");
        $("#prob").attr("src", "");
    });

    $("#predict").click(predictFromCanvas);
    $("#snap").click(predictFromCamera);
});
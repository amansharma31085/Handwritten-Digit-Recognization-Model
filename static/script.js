$(document).ready(function() {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    let isDrawing = false;
    let fullWord = ""; 

    function resetCanvas() {
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        $("#result-display").addClass("d-none");
        $("#placeholder").removeClass("d-none");
    }
    resetCanvas();

    $(canvas).mousedown(() => isDrawing = true);
    $(window).mouseup(() => isDrawing = false);
    $(canvas).mousemove(function(e) {
        if (!isDrawing) return;
        ctx.fillStyle = "white";
        ctx.beginPath();
        ctx.arc(e.offsetX, e.offsetY, 12, 0, Math.PI * 2);
        ctx.fill();
    });

    $("#clear").click(resetCanvas);

    $("#reset-word").click(function() {
        fullWord = "";
        $("#word-display").text("");
        $("#word-info").text("");
    });

    // Manual Wikipedia Define Button
    $("#define-word").click(function() {
        if (fullWord.length > 0) {
            $("#word-info").text("Searching Wikipedia...");
            $.ajax({
                type: "POST",
                url: "/get_info",
                contentType: "application/json",
                data: JSON.stringify({ word: fullWord }),
                success: function(response) {
                    $("#word-info").text(response.info);
                }
            });
        } else {
            alert("Please build a word first!");
        }
    });

    $("#predict").click(function() {
        const imageData = canvas.toDataURL("image/png");
        const currentMode = $("#mode-selector").val();

        $.ajax({
            type: "POST",
            url: "/predict",
            contentType: "application/json",
            data: JSON.stringify({ image: imageData, mode: currentMode }),
            success: function(response) {
                $("#placeholder").addClass("d-none");
                $("#result-display").removeClass("d-none");
                $("#result").text(response.result);
                $("#prob-img").attr("src", "/static/" + response.prob_img + "?t=" + new Date().getTime());
                
                if (currentMode === 'letters') {
                    fullWord += response.result;
                    $("#word-display").text(fullWord);
                }
            }
        });
    });
});
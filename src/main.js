// Copyright 2023 The MediaPipe Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//      http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Modifications copyright (C) 2024 sirococoa.

import {
    HandLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

let handLandmarker = undefined;
window.webcamRunning = false;
window.detectionRunning = false

window.videoWidth = 0;
window.videoHeight = 0;

let landmarks = new Array();

window.getLandmarks = function() {
    return landmarks;
}

const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "CPU"
        },
        runningMode: "VIDEO",
        numHands: 1
    });
};
await createHandLandmarker();

const video = document.getElementById("webcam");

const hasGetUserMedia = () => !!navigator.mediaDevices?.getUserMedia;

if (hasGetUserMedia()) {
    const constraints = {
        video: {
            width: {
              min: 1280,
              max: 1920,
            },
            height: {
              min: 720,
              max: 1080,
            },
            facingMode: 'user'
        }
    };

    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });

    window.webcamRunning = true;
} else {
    console.warn("getUserMedia() is not supported by your browser");
}

let lastVideoTime = -1;
let results = undefined;
async function predictWebcam() {
    window.videoWidth = video.videoWidth;
    window.videoHeight = video.videoHeight;

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = handLandmarker.detectForVideo(video, startTimeMs);
    }
    if (results.landmarks) {
        landmarks = results.landmarks;
    }

    if (window.webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }

    window.detectionRunning = true
}

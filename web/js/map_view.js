const CONFIG = {
    initialView: {
        center: [20.0, -35.0],
        zoom: 3.2
    },
    pollIntervalMs: 120,
    transition: {
        flyDurationSeconds: 0.95,
        flyLockMs: 950,
        bannerMs: 520
    },
    smoothing: {
        zoomLerpFactor: 0.24,
        centerLerpFactor: 0.28,
        zoomEpsilon: 0.02,
        centerEpsilonMeters: 1
    }
};

const map = L.map("map", {
    zoomControl: true,
    attributionControl: true,
    worldCopyJump: true,
    zoomSnap: 0.1,
    zoomDelta: 0.1
}).setView(CONFIG.initialView.center, CONFIG.initialView.zoom)

L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    {
        maxZoom: 20,
        attribution:
            "Sources: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
    }
).addTo(map);

const $ = (id) => document.getElementById(id);

const elements = {
    mode: $("mode"),
    zoom: $("zoom"),
    center: $("center"),
    satelliteName: $("satelliteName"),
    satelliteCaption: $("satelliteCaption"),
    satelliteTrack: $("satelliteTrack"),
    transitionBanner: $("transitionBanner"),
    transitionTitle: $("transitionTitle"),
    classificationOverlay: $("classificationOverlay"),
    classificationStatus: $("classificationStatus"),
    classificationResult: $("classificationResult"),
    classificationConfidence: $("classificationConfidence")
};

const appState = {
    lastUpdatedAt: 0,
    lastTransitionNonce: 0,
    lastClassificationGestureNonce: 0,
    classificationState: "idle",
    applyingRemoteUpdate: false,
    syncFrameHandle: null,
    desiredMapState: null
};

function formatZoom(value) {
    return Number(value).toFixed(1);
}

function formatCenter(lat, lng) {
    return `${Number(lat).toFixed(4)}, ${Number(lng).toFixed(4)}`;
}

function setApplyingRemoteUpdate(value) {
    appState.applyingRemoteUpdate = value;
}

function renderTrack(activeIndex, total) {
    const fragment = document.createDocumentFragment();

    for(let i = 0; i < total; i++) {
        const dot = document.createElement("div");
        dot.className = `dot${i === activeIndex ? " active" : ""}`;
        fragment.appendChild(dot);
    }

    elements.satelliteTrack.replaceChildren(fragment);
}

function playTransition(direction, title) {
    elements.transitionBanner.className = `transition-banner ${direction}`;
    elements.transitionTitle.textContent = title;

    requestAnimationFrame(() => {
        elements.transitionBanner.classList.add("active");
    });

    setTimeout(() => {
        elements.transitionBanner.classList.remove("active");
    }, CONFIG.transition.bannerMs);
}

function syncStatus(state) {
    elements.mode.textContent = state.active_mode.toUpperCase();
    elements.zoom.textContent = formatZoom(state.zoom_level);
    elements.center.textContent = formatCenter(state.center_lat, state.center_lng);
    elements.satelliteName.textContent = state.satellite_name;
    elements.satelliteCaption.textContent = state.satellite_caption;

    renderTrack(Number(state.satellite_index), Number(state.satellite_count));
}

function getTargetMapState(state) {
    return {
        center: L.latLng(Number(state.center_lat), Number(state.center_lng)),
        zoom: Number(state.zoom_level)
    };
}

function hasMapMoved(targetCenter, targetZoom) {
    const currentCenter = map.getCenter();
    const currentZoom = map.getZoom();

    return (
        currentCenter.distanceTo(targetCenter) > CONFIG.smoothing.centerEpsilonMeters ||
            Math.abs(currentZoom - targetZoom) > CONFIG.smoothing.zoomEpsilon
    );
}

function stopSmoothSync() {
    if (appState.syncFrameHandle === null) {
        return;
    }

    cancelAnimationFrame(appState.syncFrameHandle);
    appState.syncFrameHandle = null;
}

function smoothSyncMap() {
    const desired = appState.desiredMapState;

    if (desired === null) {
        appState.syncFrameHandle = null;
        return;
    }

    const currentCenter = map.getCenter();
    const currentZoom = map.getZoom();
    const targetCenter = L.latLng(desired.centerLat, desired.centerLng);
    const targetZoom = desired.zoom;

    const zoomDelta = targetZoom - currentZoom;
    const centerDistance = currentCenter.distanceTo(targetCenter);

    const zoomSettled = Math.abs(zoomDelta) <= CONFIG.smoothing.zoomEpsilon;
    const centerSettled = centerDistance <= CONFIG.smoothing.centerEpsilonMeters;

    if (zoomSettled && centerSettled) {
        map.setView(targetCenter, targetZoom, {animate: false});
        appState.desiredMapState = null;
        appState.syncFrameHandle = null;
        setApplyingRemoteUpdate(false);
        return;
    }

    const nextZoom = zoomSettled
        ? targetZoom
        : currentZoom + zoomDelta * CONFIG.smoothing.zoomLerpFactor;

    const nextLat = centerSettled
        ? targetCenter.lat
        : currentCenter.lat + (targetCenter.lat - currentCenter.lat) * CONFIG.smoothing.centerLerpFactor;

    const nextLng = centerSettled
        ? targetCenter.lng
        : currentCenter.lng + (targetCenter.lng - currentCenter.lng) * CONFIG.smoothing.centerLerpFactor;

    map.setView([nextLat, nextLng], nextZoom, {animate: false});
    appState.syncFrameHandle = requestAnimationFrame(smoothSyncMap);
}

function flyToMapState(targetCenter, targetZoom) {
    appState.desiredMapState = null;
    stopSmoothSync();
    setApplyingRemoteUpdate(true);

    map.flyTo(targetCenter, targetZoom, {
        animate: true,
        duration: CONFIG.transition.flyDurationSeconds,
        easeLinearity: 0.18
    });

    setTimeout(() => {
        setApplyingRemoteUpdate(false);
    }, CONFIG.transition.flyLockMs);
}

function smoothToMapState(targetCenter, targetZoom) {
    appState.desiredMapState = {
        centerLat: targetCenter.lat,
        centerLng: targetCenter.lng,
        zoom: targetZoom
    };

    setApplyingRemoteUpdate(true);

    if (appState.syncFrameHandle === null) {
        appState.syncFrameHandle = requestAnimationFrame(smoothSyncMap);
    }
}

function syncMapPosition(state, transitionTriggered) {
    const { center, zoom } = getTargetMapState(state);

    if (!hasMapMoved(center, zoom)) {
        return;
    }

    if (transitionTriggered) {
        flyToMapState(center, zoom);
        return;
    }

    smoothToMapState(center, zoom);
}

function handleTransition(state) {
    const transitionNonce = Number(state.transition_nonce);
    const triggered = transitionNonce > appState.lastTransitionNonce;

    if (!triggered) {
        return false;
    }

    appState.lastTransitionNonce = transitionNonce;
    playTransition(state.transition_direction, state.satellite_name);
    return true;
}

async function pollState() {
    try {
        const response = await fetch("/state", { cache: "no-store" });

        if (!response.ok) {
            throw new Error(`State request failed: ${response.status}`);
        }

        const state = await response.json();

        if (state.updated_at <= appState.lastUpdatedAt) {
            return;
        }

        appState.lastUpdatedAt = state.updated_at;
        syncStatus(state);

        const transitionTriggered = handleTransition(state);
        handleClassificationGesture(state);
        syncMapPosition(state, transitionTriggered);
    } catch {
        elements.mode.textContent = "DISCONNECTED";
    }
}

function syncManualMapStatus() {
    if (appState.applyingRemoteUpdate) {
        return;
    }

    const center = map.getCenter();
    elements.zoom.textContent = formatZoom(map.getZoom());
    elements.center.textContent = formatCenter(center.lat, center.lng);
}

map.on("moveend zoomend", syncManualMapStatus);

function setClassificationState(status, result = "No result yet", confidence = "") {
    elements.classificationOverlay.classList.add("active");
    elements.classificationStatus.textContent = status;
    elements.classificationResult.textContent = result;
    elements.classificationConfidence.textContent = confidence;
}

function clearClassificationOverlay() {
    appState.classificationState = "idle";
    elements.classificationOverlay.classList.remove("active");
    elements.classificationStatus.textContent = "Ready to classify";
    elements.classificationResult.textContent = "No result yet";
    elements.classificationConfidence.textContent = "";
}

function handleClassificationGesture(state) {
    const gestureNonce = Number(state.classification_gesture_nonce);

    if (gestureNonce <= appState.lastClassificationGestureNonce) {
        return;
    }

    appState.lastClassificationGestureNonce = gestureNonce;

    if (appState.classificationState === "complete") {
        clearClassificationOverlay();
        return;
    }

    if (appState.classificationState === "idle") {
        captureMapCenterCrop();
    }
}

function getPngBlob(canvas) {
    return new Promise((resolve, reject) => {
        canvas.toBlob((blob) => {
            if (blob === null) {
                reject(new Error("Could not create image"));
                return;
            }

            resolve(blob);
        }, "image/png");
    });
}

async function captureMapCenterCrop() {
    const mapElement = document.getElementById("map");

    if (!mapElement) {
        console.error("Map element not found");
        return;
    }

    appState.classificationState = "running";
    setClassificationState("Capturing map area...");

    try {
        const canvas = await html2canvas(mapElement, {
            useCORS: true,
            backgroundColor: null
        });

        const cropSize = 350;

        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;

        const cropX = centerX - cropSize / 2;
        const cropY = centerY - cropSize / 2;

        const cropCanvas = document.createElement("canvas");
        cropCanvas.width = cropSize;
        cropCanvas.height = cropSize;

        const ctx = cropCanvas.getContext("2d");

        ctx.drawImage(
            canvas,
            cropX,
            cropY,
            cropSize,
            cropSize,
            0,
            0,
            cropSize,
            cropSize
        );

        const blob = await getPngBlob(cropCanvas);
        const saveResponse = await fetch("/screenshots/latest_map_crop.png", {
            method: "POST",
            body: blob
        });

        if (!saveResponse.ok) {
            throw new Error(`Save failed: ${saveResponse.status}`);
        }

        setClassificationState("Image saved. Classification in progress...");

        const classifyResponse = await fetch("/classify/latest_map_crop.png", {
            method: "POST"
        });

        if (!classifyResponse.ok) {
            throw new Error(`Classification failed: ${classifyResponse.status}`);
        }

        const result = await classifyResponse.json();
        const confidence = `${(Number(result.confidence) * 100).toFixed(1)}% confidence`;

        appState.classificationState = "complete";
        setClassificationState("Classification complete", result.category, confidence);
    } catch (error) {
        appState.classificationState = "complete";
        setClassificationState("Classification failed", "Try again", error.message);
    }
}

renderTrack(0, 5);
pollState();
setInterval(pollState, CONFIG.pollIntervalMs);
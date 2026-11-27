
const API = {
    pods: "/api/kubernetes/pods",
    metrics: "/api/kubernetes/metrics",
    cpuStart: "/api/anomalies/cpu/start",
    cpuStop: "/api/anomalies/cpu/stop",
    cpuStatus: "/api/anomalies/cpu/status"
};

function getElement(id) {
    return document.getElementById(id);
}

function setText(id, value) {
    const element = getElement(id);
    if (element) element.textContent = value;
}

function escapeHTML(value) {
    return String(value).replace(/[&<>"']/g, character => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
    })[character]);
}

function showConnection(message, connected) {
    const element = getElement("connection-status");
    if (!element) return;

    element.textContent = message;
    element.className = connected ? "connected" : "disconnected";
}

async function fetchJSON(url, options = {}) {
    const response = await fetch(url, options);

    let data;

    try {
        data = await response.json();
    } catch {
        throw new Error("The server returned an invalid response.");
    }

    if (!response.ok) {
        throw new Error(data.detail || data.message || "Request failed.");
    }

    return data;
}

// Load Kubernetes pod information
async function loadPods() {
    const data = await fetchJSON(API.pods);

    const pods = Array.isArray(data) ? data : (data.pods || []);

    const running = pods.filter(pod =>
        String(pod.status).toLowerCase() === "running" && pod.ready
    ).length;

    const unhealthy = pods.length - running;

    const restarts = pods.reduce(
        (total, pod) => total + Number(pod.restarts || 0),
        0
    );

    setText("total-pods", pods.length);
    setText("running-pods", running);
    setText("unhealthy-pods", unhealthy);
    setText("total-restarts", restarts);

    const table = getElement("pods-table");

    if (!table) return;

    if (pods.length === 0) {
        table.innerHTML = '<tr><td colspan="4">No pods found.</td></tr>';
        return;
    }

    table.innerHTML = pods.map(pod => `
        <tr>
            <td>${escapeHTML(pod.name)}</td>
            <td>${escapeHTML(pod.status || "Unknown")}</td>
            <td>${pod.ready ? "Ready" : "Not Ready"}</td>
            <td>${Number(pod.restarts || 0)}</td>
        </tr>
    `).join("");
}

// Load actual Kubernetes CPU and memory metrics
async function loadMetrics() {
    const data = await fetchJSON(API.metrics);

    // Correctly read the backend's "metrics" property
    const metrics = Array.isArray(data)
        ? data
        : (data.metrics || data.pods || []);

    const totalCPU = metrics.reduce(
        (sum, pod) => sum + Number(pod.cpu_millicores || 0),
        0
    );

    const totalMemory = metrics.reduce(
        (sum, pod) => sum + Number(pod.memory_mib || 0),
        0
    );

    setText("total-cpu", `${totalCPU.toFixed(2)} m`);
    setText("total-memory", `${totalMemory.toFixed(2)} MiB`);
    setText("metrics-pod-count", metrics.length);

    const table = getElement("metrics-table");

    if (!table) return;

    if (metrics.length === 0) {
        table.innerHTML =
            '<tr><td colspan="3">No metrics available.</td></tr>';
        return;
    }

    table.innerHTML = metrics.map(pod => `
        <tr>
            <td>${escapeHTML(pod.name)}</td>
            <td>${Number(pod.cpu_millicores || 0).toFixed(2)} m</td>
            <td>${Number(pod.memory_mib || 0).toFixed(2)} MiB</td>
        </tr>
    `).join("");
}

// Check CPU anomaly test status
async function loadCPUStatus() {
    const data = await fetchJSON(API.cpuStatus);
    updateCPUStatus(data);
}

function updateCPUStatus(data) {
    const status = getElement("cpu-test-status");
    const startButton = getElement("start-cpu-test");
    const stopButton = getElement("stop-cpu-test");

    if (!status || !startButton || !stopButton) return;

    if (data.running) {
        status.textContent = "CPU test is running.";
        status.className = "test-status running";
        startButton.disabled = true;
        stopButton.disabled = false;
    } else if (data.exists && data.phase) {
        status.textContent = `Test pod status: ${data.phase}.`;
        status.className = "test-status";
        startButton.disabled = false;
        stopButton.disabled = false;
    } else {
        status.textContent = "CPU test is stopped.";
        status.className = "test-status stopped";
        startButton.disabled = false;
        stopButton.disabled = true;
    }
}

// Start CPU stress test
async function startCPUTest() {
    const button = getElement("start-cpu-test");
    const status = getElement("cpu-test-status");

    button.disabled = true;
    status.textContent = "Starting CPU test...";

    try {
        const data = await fetchJSON(API.cpuStart, {
            method: "POST"
        });

        status.textContent = data.message || "CPU test start requested.";

        await loadCPUStatus();
        await refreshDashboard();

    } catch (error) {
        status.textContent = `Could not start test: ${error.message}`;
        button.disabled = false;
    }
}

// Stop CPU stress test
async function stopCPUTest() {
    const button = getElement("stop-cpu-test");
    const status = getElement("cpu-test-status");

    button.disabled = true;
    status.textContent = "Stopping CPU test...";

    try {
        const data = await fetchJSON(API.cpuStop, {
            method: "POST"
        });

        status.textContent = data.message || "CPU test stop requested.";

        await loadCPUStatus();
        await refreshDashboard();

    } catch (error) {
        status.textContent = `Could not stop test: ${error.message}`;
        button.disabled = false;
    }
}

// Refresh dashboard data
async function refreshDashboard() {
    try {
        await Promise.all([
            loadPods(),
            loadMetrics()
        ]);

        showConnection("Connected", true);

        setText(
            "last-updated",
            `Last updated: ${new Date().toLocaleTimeString()}`
        );

    } catch (error) {
        showConnection("Connection error", false);
        console.error("Dashboard refresh failed:", error);
    }
}

// Initialize dashboard
async function initializeDashboard() {
    getElement("refresh-button")?.addEventListener(
        "click",
        refreshDashboard
    );

    getElement("start-cpu-test")?.addEventListener(
        "click",
        startCPUTest
    );

    getElement("stop-cpu-test")?.addEventListener(
        "click",
        stopCPUTest
    );

    await refreshDashboard();

    try {
        await loadCPUStatus();
    } catch (error) {
        setText(
            "cpu-test-status",
            `Unable to check test status: ${error.message}`
        );
    }

    setInterval(refreshDashboard, 10000);

    setInterval(() => {
        loadCPUStatus().catch(error =>
            console.error("CPU status check failed:", error)
        );
    }, 5000);
}

document.addEventListener(
    "DOMContentLoaded",
    initializeDashboard
);

/* ── State ────────────────────────────────────────────────── */
let chatHistory = [];
let totalMsgs = 0;
let isProcessing = false;

/* ── Init ─────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  checkStatus();
  loadDocuments();
  setupDragDrop();
  document.getElementById("fileInput").addEventListener("change", handleFileSelect);
});

/* ── Status Check ─────────────────────────────────────────── */
async function checkStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const dot = document.getElementById("statusDot");
    const txt = document.getElementById("statusText");
    if (data.api_key_set) {
      dot.className = "status-dot ok";
      txt.textContent = "Gemini Connected";
    } else {
      dot.className = "status-dot err";
      txt.textContent = "API Key Missing";
    }
  } catch {
    document.getElementById("statusText").textContent = "Server Error";
  }
}

/* ── Drag & Drop ──────────────────────────────────────────── */
function setupDragDrop() {
  const zone = document.getElementById("uploadZone");
  zone.addEventListener("dragover", e => { e.preventDefault(); zone.classList.add("drag-over"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const files = Array.from(e.dataTransfer.files).filter(f => f.type === "application/pdf");
    if (files.length) uploadFiles(files);
    else showToast("Only PDF files are accepted.", "error");
  });
  zone.addEventListener("click", e => {
    if (e.target.tagName !== "BUTTON") document.getElementById("fileInput").click();
  });
}

function handleFileSelect(e) {
  const files = Array.from(e.target.files);
  if (files.length) uploadFiles(files);
  e.target.value = "";
}

/* ── Upload ───────────────────────────────────────────────── */
async function uploadFiles(files) {
  const progress = document.getElementById("uploadProgress");
  const fill = document.getElementById("progressFill");
  const progressText = document.getElementById("progressText");

  progress.style.display = "block";
  fill.style.width = "10%";
  progressText.textContent = `Processing ${files.length} file(s)…`;

  const formData = new FormData();
  files.forEach(f => formData.append("files", f));

  try {
    fill.style.width = "40%";
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    fill.style.width = "80%";
    const data = await res.json();

    if (!res.ok) {
      showToast(data.error || "Upload failed.", "error");
    } else {
      fill.style.width = "100%";
      progressText.textContent = data.message;
      data.results.forEach(r => {
        if (r.success) showToast(`✅ ${r.filename} — ${r.chunks} chunks`, "success");
        else showToast(`❌ ${r.filename}: ${r.error}`, "error");
      });
      renderDocList(data.documents);
      updateStats({ docs: data.documents.length });
      await refreshChunkCount();
    }
  } catch (e) {
    showToast("Network error during upload.", "error");
    progressText.textContent = "Upload failed.";
  }

  setTimeout(() => { progress.style.display = "none"; fill.style.width = "0"; }, 2000);
}

/* ── Documents ────────────────────────────────────────────── */
async function loadDocuments() {
  try {
    const res = await fetch("/api/documents");
    const data = await res.json();
    renderDocList(data.documents || []);
    updateStats({ docs: (data.documents || []).length, chunks: data.total_chunks || 0 });
  } catch { /* silent */ }
}

function renderDocList(docs) {
  const list = document.getElementById("docList");
  const btn = document.getElementById("summaryBtn");
  document.getElementById("docCount").textContent = docs.length;

  if (!docs || docs.length === 0) {
    list.innerHTML = '<div class="empty-docs">No documents yet.<br/>Upload a PDF to begin.</div>';
    btn.disabled = true;
    return;
  }

  btn.disabled = false;
  list.innerHTML = docs.map(doc => `
    <div class="doc-item" id="doc-${CSS.escape(doc.filename)}">
      <span class="doc-icon">📄</span>
      <div class="doc-info">
        <div class="doc-name" title="${esc(doc.filename)}">${esc(doc.filename)}</div>
        <div class="doc-meta">${doc.pages || '?'} pages · ${doc.chunks || '?'} chunks</div>
      </div>
      <button class="doc-delete" title="Remove document" onclick="deleteDoc('${esc(doc.filename)}')">✕</button>
    </div>`).join("");
}

async function deleteDoc(filename) {
  if (!confirm(`Remove "${filename}" from the knowledge base?`)) return;
  try {
    const res = await fetch(`/api/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
    const data = await res.json();
    if (res.ok) {
      showToast(`Removed ${filename}`, "success");
      renderDocList(data.documents || []);
      updateStats({ docs: (data.documents || []).length });
      await refreshChunkCount();
    } else {
      showToast(data.error || "Delete failed.", "error");
    }
  } catch { showToast("Network error.", "error"); }
}

async function refreshChunkCount() {
  try {
    const res = await fetch("/api/documents");
    const data = await res.json();
    updateStats({ chunks: data.total_chunks || 0 });
  } catch { /* silent */ }
}

function updateStats({ docs, chunks, msgs } = {}) {
  if (docs !== undefined) document.getElementById("statDocs").textContent = docs;
  if (chunks !== undefined) document.getElementById("statChunks").textContent = chunks;
  if (msgs !== undefined) document.getElementById("statMsgs").textContent = msgs;
}

/* ── Chat ─────────────────────────────────────────────────── */
function handleKey(e) {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); askQuestion(); }
}

function resizeInput(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 120) + "px";
}

async function askQuestion() {
  if (isProcessing) return;
  const input = document.getElementById("questionInput");
  const question = input.value.trim();
  if (!question) return;

  hideWelcome();
  isProcessing = true;
  toggleSendBtn(false);
  input.value = "";
  input.style.height = "auto";

  addUserMessage(question);
  chatHistory.push({ role: "user", content: question });
  totalMsgs++;
  updateStats({ msgs: totalMsgs });

  const thinkId = addThinking();

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    removeThinking(thinkId);

    if (!res.ok || data.error) {
      addErrorMessage(data.error || "Something went wrong.");
    } else {
      addAIMessage(data);
      chatHistory.push({ role: "assistant", content: data.answer });
      totalMsgs++;
      updateStats({ msgs: totalMsgs });
    }
  } catch (e) {
    removeThinking(thinkId);
    addErrorMessage("Network error. Check your connection and try again.");
  }

  isProcessing = false;
  toggleSendBtn(true);
  scrollBottom();
  input.focus();
}

function toggleSendBtn(enabled) {
  document.getElementById("sendBtn").disabled = !enabled;
}

function hideWelcome() {
  const w = document.getElementById("welcomeScreen");
  if (w) w.style.display = "none";
}

function addUserMessage(text) {
  document.getElementById("messages").insertAdjacentHTML("beforeend", `
    <div class="msg-user">
      <div class="user-bubble">${esc(text)}</div>
    </div>`);
  scrollBottom();
}

function addThinking() {
  const id = "think-" + Date.now();
  document.getElementById("messages").insertAdjacentHTML("beforeend", `
    <div class="msg-ai thinking-wrap" id="${id}">
      <div class="ai-avatar">🔬</div>
      <div class="ai-body-wrap">
        <div class="ai-meta">
          <span class="ai-label">ResearchRAG</span>
        </div>
        <div class="ai-answer">
          <div class="dots"><span></span><span></span><span></span></div>
          <span class="thinking-label">Running LangGraph pipeline…</span>
        </div>
      </div>
    </div>`);
  scrollBottom();
  return id;
}

function removeThinking(id) {
  document.getElementById(id)?.remove();
}

function addAIMessage(data) {
  const time = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const sourcesHtml = buildSourcesHtml(data.sources || []);
  const msgId = "msg-" + Date.now();

  document.getElementById("messages").insertAdjacentHTML("beforeend", `
    <div class="msg-ai" id="${msgId}">
      <div class="ai-avatar">🔬</div>
      <div class="ai-body-wrap">
        <div class="ai-meta">
          <span class="ai-label">ResearchRAG</span>
          <span class="ai-time">${time}</span>
          ${data.chunks_used ? `<span class="chunks-badge">${data.chunks_used} chunks</span>` : ""}
        </div>
        <div class="ai-answer">${renderMd(data.answer)}</div>
        ${sourcesHtml}
        <div class="ai-footer">
          <button class="copy-btn" onclick="copyText('${msgId}')">Copy answer</button>
        </div>
      </div>
    </div>`);
}

function addErrorMessage(msg) {
  document.getElementById("messages").insertAdjacentHTML("beforeend", `
    <div class="msg-ai">
      <div class="ai-avatar">⚠️</div>
      <div class="ai-body-wrap">
        <div class="ai-meta"><span class="ai-label">Error</span></div>
        <div class="error-bubble">${esc(msg)}</div>
      </div>
    </div>`);
}

function buildSourcesHtml(sources) {
  if (!sources || sources.length === 0) return "";
  const chips = sources.map(s => `
    <div class="source-chip">
      <span>📄</span>
      <span>${esc(s.source)}</span>
      <span class="chip-page">p.${s.page}</span>
      <span class="chip-score">${(s.relevance * 100).toFixed(0)}%</span>
      <div class="tooltip"><strong>${esc(s.source)} — Page ${s.page}</strong><br/><br/>${esc(s.snippet)}</div>
    </div>`).join("");
  return `<div class="sources-panel"><div class="sources-title">Sources Used</div><div class="source-chips">${chips}</div></div>`;
}

function copyText(msgId) {
  const el = document.getElementById(msgId)?.querySelector(".ai-answer");
  if (el) {
    navigator.clipboard.writeText(el.innerText).then(() => {
      showToast("Copied to clipboard!", "success");
    });
  }
}

function clearChat() {
  if (!confirm("Clear the chat history?")) return;
  document.getElementById("messages").innerHTML = "";
  document.getElementById("welcomeScreen").style.display = "block";
  chatHistory = [];
  totalMsgs = 0;
  updateStats({ msgs: 0 });
  fetch("/api/chat/clear", { method: "POST" });
}

function downloadChat() {
  if (chatHistory.length === 0) { showToast("No chat to download yet.", "error"); return; }
  const lines = chatHistory.map(m => `[${m.role.toUpperCase()}]\n${m.content}`).join("\n\n" + "─".repeat(60) + "\n\n");
  const blob = new Blob([`RAG Research Assistant — Chat History\n${"=".repeat(60)}\n\n${lines}`], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `research-chat-${Date.now()}.txt`;
  a.click();
}

/* ── Summary Modal ────────────────────────────────────────── */
async function generateSummary() {
  const modal = document.getElementById("summaryModal");
  const body = document.getElementById("summaryBody");
  modal.classList.add("open");
  body.innerHTML = `<div class="summary-loading"><div class="spinner"></div><span>Analysing your documents…</span></div>`;

  try {
    const res = await fetch("/api/summarize", { method: "POST" });
    const data = await res.json();
    if (!res.ok || data.error) {
      body.innerHTML = `<div class="error-bubble">${esc(data.error || "Summary failed.")}</div>`;
    } else {
      body.innerHTML = `<div class="summary-content">${renderMd(data.summary)}</div>`;
    }
  } catch {
    body.innerHTML = `<div class="error-bubble">Network error generating summary.</div>`;
  }
}

function closeSummary() {
  document.getElementById("summaryModal").classList.remove("open");
}

document.addEventListener("keydown", e => { if (e.key === "Escape") closeSummary(); });

/* ── Markdown Renderer ────────────────────────────────────── */
function renderMd(text) {
  if (!text) return "";
  let html = text
    .replace(/```[\w]*\n?([\s\S]*?)```/g, (_, c) => `<pre><code>${esc(c.trim())}</code></pre>`)
    .replace(/`([^`]+)`/g, (_, c) => `<code>${esc(c)}</code>`)
    .replace(/^#### (.+)$/gm, "<h4>$1</h4>")
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm, "<h2>$1</h2>")
    .replace(/^# (.+)$/gm, "<h1>$1</h1>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/^---+$/gm, "<hr>")
    .replace(/^> (.+)$/gm, "<blockquote>$1</blockquote>")
    .replace(/^[\-\*] (.+)$/gm, "<li>$1</li>")
    .replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

  // Wrap consecutive <li> blocks in <ul>
  html = html.replace(/(<li>.*<\/li>(\n|$))+/g, m => `<ul>${m}</ul>`);

  // Paragraphs
  html = html.split(/\n\n+/).map(block => {
    const trimmed = block.trim();
    if (!trimmed) return "";
    if (/^<(h[1-4]|ul|ol|pre|blockquote|hr)/.test(trimmed)) return trimmed;
    return `<p>${trimmed.replace(/\n/g, "<br>")}</p>`;
  }).join("\n");

  return html;
}

/* ── Helpers ──────────────────────────────────────────────── */
function esc(s) {
  return String(s || "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function scrollBottom() {
  const w = document.getElementById("chatWindow");
  setTimeout(() => w.scrollTop = w.scrollHeight, 60);
}

let toastTimer;
function showToast(msg, type = "success") {
  let toast = document.getElementById("toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "toast";
    toast.style.cssText = `
      position:fixed;bottom:24px;right:24px;z-index:999;
      padding:10px 18px;border-radius:10px;font-size:13px;font-weight:500;
      max-width:320px;line-height:1.5;transition:opacity .3s;
      font-family:'Inter',sans-serif;box-shadow:0 8px 24px rgba(0,0,0,0.4);`;
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.style.opacity = "1";
  toast.style.background = type === "error" ? "rgba(248,113,113,0.15)" : "rgba(52,211,153,0.15)";
  toast.style.border = type === "error" ? "1px solid rgba(248,113,113,0.4)" : "1px solid rgba(52,211,153,0.4)";
  toast.style.color = type === "error" ? "#F87171" : "#34D399";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.style.opacity = "0"; }, 3500);
}

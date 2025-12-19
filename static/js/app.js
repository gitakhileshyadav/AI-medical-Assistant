/* ===========================
   MedVision – Production App.js
   =========================== */

document.addEventListener("DOMContentLoaded", () => {
  const chat = document.getElementById("chat");
  const fileInput = document.getElementById("file");
  const preview = document.getElementById("preview");
  const previewEmpty = document.getElementById("preview-empty");
  const queryInput = document.getElementById("query");
  const sendBtn = document.getElementById("send");
  const resetBtn = document.getElementById("reset");

  let selectedFile = null;
  let isSending = false;

  /* ---------- Utilities ---------- */

  const escapeHTML = (str) =>
    str.replace(/[&<>"']/g, (m) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[m])
    );

  const scrollToBottom = () => {
    chat.scrollTop = chat.scrollHeight;
  };

  const spacer = () => {
    const div = document.createElement("div");
    div.style.height = "14px";
    return div;
  };

  /* ---------- UI Builders ---------- */

  function addUserMessage(text) {
    const msg = document.createElement("div");
    msg.className = "msg user";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;

    msg.appendChild(bubble);
    chat.appendChild(msg);
    chat.appendChild(spacer());
    scrollToBottom();
  }

  function addAssistantMessage(text) {
    const msg = document.createElement("div");
    msg.className = "msg bot";

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const blocks = text.split("•");
    blocks.forEach((b, i) => {
      const clean = b.trim();
      if (!clean) return;

      const p = document.createElement("p");
      p.textContent = i === 0 ? clean : "• " + clean;
      bubble.appendChild(p);
    });

    /* Copy Button */
    const copyBtn = document.createElement("button");
    copyBtn.textContent = "Copy";
    copyBtn.style.marginTop = "10px";
    copyBtn.style.fontSize = "12px";
    copyBtn.onclick = () => {
      navigator.clipboard.writeText(text);
      copyBtn.textContent = "Copied ✓";
      setTimeout(() => (copyBtn.textContent = "Copy"), 1500);
    };

    bubble.appendChild(copyBtn);
    msg.appendChild(bubble);
    chat.appendChild(msg);
    chat.appendChild(spacer());
    scrollToBottom();
  }

  function addLoader() {
    const loader = document.createElement("div");
    loader.id = "loader";
    loader.className = "msg bot";
    loader.innerHTML = `<div class="bubble">Analyzing image…</div>`;
    chat.appendChild(loader);
    scrollToBottom();
  }

  function removeLoader() {
    const loader = document.getElementById("loader");
    if (loader) loader.remove();
  }

  /* ---------- Image Handling ---------- */

  function showPreview(file) {
    preview.innerHTML = "";
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    img.onload = () => URL.revokeObjectURL(img.src);
    preview.appendChild(img);
  }

  function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    selectedFile = file;
    previewEmpty.style.display = "none";
    showPreview(file);
  }

  fileInput.addEventListener("change", (e) => {
    handleFile(e.target.files[0]);
  });

  /* Drag & Drop */
  preview.addEventListener("dragover", (e) => {
    e.preventDefault();
    preview.style.borderColor = "#38bdf8";
  });

  preview.addEventListener("dragleave", () => {
    preview.style.borderColor = "";
  });

  preview.addEventListener("drop", (e) => {
    e.preventDefault();
    preview.style.borderColor = "";
    handleFile(e.dataTransfer.files[0]);
  });

  /* ---------- Send Message ---------- */

  async function sendMessage() {
    if (isSending) return;

    const query = queryInput.value.trim();
    if (!query) return;

    isSending = true;
    addUserMessage(query);
    queryInput.value = "";
    addLoader();

    const formData = new FormData();
    formData.append("query", query);
    if (selectedFile) {
      formData.append("image_file", selectedFile);
      selectedFile = null; // image only once
    }

    try {
      const res = await fetch("/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      removeLoader();

      if (!res.ok) {
        addAssistantMessage(data.detail || "Server error");
      } else {
        addAssistantMessage(data.answer);
      }
    } catch (err) {
      removeLoader();
      addAssistantMessage("Network error. Please retry.");
    } finally {
      isSending = false;
    }
  }

  sendBtn.addEventListener("click", sendMessage);

  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
  });

  /* ---------- Reset ---------- */

  resetBtn.addEventListener("click", async () => {
    await fetch("/reset", { method: "POST" });
    chat.innerHTML = "";
    preview.innerHTML = `<div class="muted" id="preview-empty">No image selected</div>`;
    selectedFile = null;
  });
});

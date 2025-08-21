// static/js/app.js
// Dynamically load external script and return Promise when loaded
function loadScript(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) {
      // already added
      resolve();
      return;
    }
    const s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.onload = () => resolve();
    s.onerror = (e) => reject(new Error('Failed to load ' + src));
    document.head.appendChild(s);
  });
}

async function startVanta() {
  try {
    // load three + vanta
    await loadScript('https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js');
    await loadScript('https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.globe.min.js');
    // init vanta
    if (window.VANTA && window.VANTA.GLOBE) {
      window.vantaEffect = window.VANTA.GLOBE({
        el: "#vanta-bg",
        mouseControls: true,
        touchControls: true,
        minHeight: 200.00,
        minWidth: 200.00,
        scale: 1.0,
        scaleMobile: 1.0,
        color: 0x7c3aed,
        color2: 0x06b6d4,
        backgroundColor: 0x021026,
        size: 1.1
      });
    }
  } catch (err) {
    // fail silently - Vanta is nice-to-have but not required
    console.warn('Vanta init failed:', err);
  }
}

/* ---------- App behavior: chat + file handling ---------- */
document.addEventListener('DOMContentLoaded', async () => {
  await startVanta();

  const fileInput = document.getElementById('file');
  const preview = document.getElementById('preview');
  const previewEmpty = document.getElementById('preview-empty');
  const resetBtn = document.getElementById('reset');
  const chat = document.getElementById('chat');
  const sendBtn = document.getElementById('send');
  const queryInput = document.getElementById('query');

  let firstRequest = true;

  function addMessage(text, who='bot') {
    const container = document.createElement('div');
    container.className = 'msg ' + (who === 'user' ? 'user' : 'bot');
    const bubble = document.createElement('div');
    bubble.className = 'bubble ' + (who === 'user' ? 'user' : 'bot');
    bubble.textContent = text;
    container.appendChild(bubble);
    chat.appendChild(container);
    chat.scrollTop = chat.scrollHeight;
  }

  function resizeFileToBlob(file, maxWidth = 1024, quality = 0.78) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = reject;
      reader.onload = () => {
        const img = new Image();
        img.onerror = reject;
        img.onload = () => {
          const scale = Math.min(1, maxWidth / img.width);
          const canvas = document.createElement('canvas');
          canvas.width = Math.round(img.width * scale);
          canvas.height = Math.round(img.height * scale);
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          canvas.toBlob((blob) => {
            if (!blob) reject(new Error('Canvas toBlob failed'));
            else resolve(blob);
          }, 'image/jpeg', quality);
        };
        img.src = reader.result;
      };
      reader.readAsDataURL(file);
    });
  }

  fileInput.addEventListener('change', (e) => {
    const f = e.target.files[0];
    if (!f) { preview.innerHTML = '<div class="muted">No image selected</div>'; return; }
    const url = URL.createObjectURL(f);
    preview.innerHTML = '<img src="' + url + '" alt="Uploaded image preview" />';
  });

  resetBtn.addEventListener('click', async () => {
    await fetch('/reset', { method: 'POST' }).catch(()=>{});
    preview.innerHTML = '<div class="muted">No image selected</div>';
    chat.innerHTML = '';
    firstRequest = true;
    addMessage('Conversation reset. Upload a new image to begin.', 'bot');
  });

  sendBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;
    const file = fileInput.files.length ? fileInput.files[0] : null;
    if (firstRequest && !file) {
      addMessage('Please upload an image for the first request.', 'bot');
      return;
    }

    addMessage(query, 'user');

    // loading bubble
    const loadingBubble = document.createElement('div');
    loadingBubble.className = 'msg bot';
    loadingBubble.innerHTML = '<div class="bubble bot">Analyzing… <span style="margin-left:8px">⏳</span></div>';
    chat.appendChild(loadingBubble);
    chat.scrollTop = chat.scrollHeight;

    const form = new FormData();
    form.append('query', query);

    if (firstRequest && file) {
      try {
        const blob = await resizeFileToBlob(file, 1024, 0.78);
        form.append('image_file', blob, 'upload.jpg');
      } catch (err) {
        loadingBubble.remove();
        addMessage('Image processing failed: ' + err.message, 'bot');
        return;
      }
    }

    sendBtn.disabled = true;
    try {
      const res = await fetch('/analyze', { method: 'POST', body: form });
      const text = await res.text();
      loadingBubble.remove();
      if (!res.ok) {
        addMessage('Server error: ' + text, 'bot');
        sendBtn.disabled = false;
        return;
      }
      const data = JSON.parse(text);
      addMessage(data.answer || 'No answer returned', 'bot');
      firstRequest = false;
    } catch (err) {
      loadingBubble.remove();
      addMessage('Network error: ' + err.message, 'bot');
    } finally {
      sendBtn.disabled = false;
      queryInput.value = '';
    }
  });

  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendBtn.click();
    }
  });

  // initial greeting
  addMessage('Welcome! Upload an image and ask a question to start analysis.', 'bot');
});

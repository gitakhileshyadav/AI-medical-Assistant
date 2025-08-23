# app.py (production-ready)
"""
Production-ready FastAPI app for the GenAI Medical Assistant.

Key production-ready features included here:
- CORS support via ALLOWED_ORIGINS environment variable
- Simple request-size guard for uploads (MAX_IMAGE_MB)
- Improved logging and error handling
- Health endpoint
- In-memory session store with clear comments and guidance to replace with Redis for multi-instance deployments
- Graceful Groq API call with retries and controlled timeouts
- Static files + Jinja2 templates mount as before

To run locally:
    pip install -r requirements.txt
    export GROQ_API_KEY=<your_key>           # optional if you only test locally without calling Groq
    export ALLOWED_ORIGINS=http://localhost:3000,https://your-vercel-domain.vercel.app
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

For production (render/gunicorn):
    gunicorn -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT --timeout 120

IMPORTANT: The in-memory `sessions` dict is not suitable for multi-instance production. Replace it
with a Redis store (or database) if you deploy to more than one process or machine.
"""

import os
import io
import time
import uuid
import base64
import logging
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

import requests
from PIL import Image

load_dotenv()

# ----------------- CONFIG -----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("genai")

GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Max upload size guard (in megabytes)
MAX_IMAGE_MB = float(os.getenv("MAX_IMAGE_MB", "6"))
MAX_IMAGE_BYTES = int(MAX_IMAGE_MB * 1024 * 1024)

# Session cookie settings
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "session_id")
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "true").lower() in ("1", "true", "yes")
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")

# CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else []
# If empty, allow all origins for local dev. Be explicit in production.
ALLOW_ALL_ORIGINS = len([o for o in ALLOWED_ORIGINS if o.strip()]) == 0

# ----------------- APP INIT -----------------
os.makedirs("static", exist_ok=True)
app = FastAPI(title="GenAI Medical Assistant")

# Add useful middlewares
app.add_middleware(GZipMiddleware, minimum_size=500)

if ALLOW_ALL_ORIGINS:
    logger.warning("ALLOWED_ORIGINS is empty — allowing all origins (development only).")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    origins_cleaned = [o.strip() for o in ALLOWED_ORIGINS if o.strip()]
    logger.info("CORS allowed origins: %s", origins_cleaned)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins_cleaned,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory session store. Replace with Redis for production multi-instance deployments.
# sessions: { session_id: {"history": [...], "last_image": "data:image/..;base64,...", "created": ts } }
sessions: Dict[str, Dict[str, Any]] = {}

# ----------------- HELPERS -----------------

def new_session() -> str:
    sid = str(uuid.uuid4())
    sessions[sid] = {"history": [], "last_image": None, "created": time.time()}
    logger.debug("Created session %s", sid)
    return sid


def get_session_id(request: Request) -> str:
    sid = request.cookies.get(SESSION_COOKIE_NAME)
    if sid and sid in sessions:
        return sid
    return new_session()


def assert_image_size_ok(bytes_len: int) -> None:
    if bytes_len > MAX_IMAGE_BYTES:
        raise ValueError(f"Image too large ({bytes_len} bytes). Max allowed {MAX_IMAGE_BYTES} bytes.")


def validate_image_bytes(image_bytes: bytes) -> None:
    assert_image_size_ok(len(image_bytes))
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except Exception as e:
        raise ValueError("Invalid image format") from e


def bytes_to_data_url(image_bytes: bytes, content_type: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{content_type};base64,{b64}"


def shrink_data_image(data_url: str, max_width: int = 1024, quality: int = 75) -> str:
    try:
        header, b64 = data_url.split(",", 1)
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        if img.width <= max_width:
            return data_url
        ratio = max_width / float(img.width)
        new_h = int(img.height * ratio)
        img = img.resize((max_width, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        new_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{new_b64}"
    except Exception as e:
        logger.warning("shrink_data_image failed, returning original: %s", e)
        return data_url


def prepare_user_message(query: str, data_url: Optional[str] = None):
    if data_url:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    return {"role": "user", "content": query}


# ----------------- External LLM call (Groq) -----------------

def call_groq_api(messages: List[Dict[str, Any]], max_retries: int = 3, timeout: int = 60) -> requests.Response:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Set the key in .env or env variables.")

    # shrink images inside messages (server-side guard)
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for chunk in content:
                if chunk.get("type") == "image_url":
                    url = chunk.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        chunk["image_url"]["url"] = shrink_data_image(url, max_width=1024, quality=75)

    payload = {
        "model": os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        "messages": messages,
        "max_tokens": int(os.getenv("GROQ_MAX_TOKENS", "300")),
        "temperature": float(os.getenv("GROQ_TEMPERATURE", "0.2")),
    }

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    backoff = 1.0
    last_resp = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Calling Groq (attempt %d)...", attempt)
            resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
            last_resp = resp
            logger.info("Groq returned status %s", resp.status_code)
            if 200 <= resp.status_code < 300:
                return resp
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                # client error -> return immediately
                logger.error("Client error from Groq: %s %s", resp.status_code, resp.text[:1000])
                return resp
            logger.warning("Server error from Groq (will retry if attempts left): %s %s", resp.status_code, resp.text[:1000])
        except requests.RequestException as e:
            logger.warning("Network error calling Groq: %s", e)
            last_resp = None

        if attempt < max_retries:
            time.sleep(backoff)
            backoff *= 2

    raise RuntimeError(
        f"Groq API failed after {max_retries} attempts. last_status={getattr(last_resp, 'status_code', None)} last_body={getattr(last_resp, 'text', None)[:2000] if last_resp is not None else None}"
    )


# ----------------- ROUTES -----------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    sid = get_session_id(request)
    resp = templates.TemplateResponse("index.html", {"request": request})
    # Cookie attributes: httponly, samesite, secure when in production
    resp.set_cookie(SESSION_COOKIE_NAME, sid, httponly=True, samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE)
    return resp


@app.get("/health", response_class=PlainTextResponse)
async def health():
    return "ok"


@app.post("/analyze")
async def analyze(request: Request, query: str = Form(...), image_file: Optional[UploadFile] = File(None)):
    """
    - If image_file provided -> treat as first request. Store image (base64 data URL) in session and reset history.
    - Otherwise -> follow-up using session history.
    """
    try:
        sid = get_session_id(request)
        session = sessions.setdefault(sid, {"history": [], "last_image": None})

        provided_data_url = None

        # If file uploaded, read and validate
        if image_file is not None:
            contents = await image_file.read()
            try:
                validate_image_bytes(contents)
            except ValueError as ve:
                logger.info("Invalid upload from client: %s", ve)
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
            provided_data_url = bytes_to_data_url(contents, content_type=image_file.content_type or "image/jpeg")

        # If a new image was uploaded -> reset history and store image
        if provided_data_url:
            session["last_image"] = provided_data_url
            # Reset history and add helpful system prompt
            session["history"] = [
                {
                    "role": "system",
                    "content": "You are a careful medical assistant. Provide likely diagnoses from the image, quantify uncertainty, and always recommend clinical confirmation.",
                }
            ]
            user_msg = prepare_user_message(query, provided_data_url)
            session["history"].append(user_msg)
        else:
            # follow-up: require an existing session history
            if not session.get("history"):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No image in session. Upload an image on the first request.")
            user_msg = prepare_user_message(query)
            session["history"].append(user_msg)

        # Call Groq with retry + server-side shrink of image if needed
        try:
            resp = call_groq_api(session["history"])
        except RuntimeError as re:
            logger.exception("Groq call failed after retries")
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(re))

        if not (200 <= resp.status_code < 300):
            logger.error("LLM API error: %s %s", resp.status_code, resp.text[:2000])
            return JSONResponse(content={"detail": f"LLM API returned {resp.status_code}: {resp.text}"}, status_code=500)

        result = resp.json()
        try:
            answer = result["choices"][0]["message"]["content"]
        except Exception:
            logger.exception("Unexpected LLM response")
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Unexpected LLM response structure")

        # Save assistant reply to history
        session["history"].append({"role": "assistant", "content": answer})

        # Make sure cookie is present for clients that didn't have it yet
        response = JSONResponse({"answer": answer})
        response.set_cookie(SESSION_COOKIE_NAME, sid, httponly=True, samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE)
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error in /analyze")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/reset")
async def reset(request: Request):
    sid = request.cookies.get(SESSION_COOKIE_NAME)
    if sid and sid in sessions:
        del sessions[sid]
    new_sid = new_session()
    resp = JSONResponse({"status": "reset"})
    resp.set_cookie(SESSION_COOKIE_NAME, new_sid, httponly=True, samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE)
    return resp


# ----------------- START (for local development) -----------------
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_flag = os.getenv("DEV_RELOAD", "true").lower() in ("1", "true", "yes")

    uvicorn.run("app:app", host=host, port=port, reload=reload_flag, log_level=LOG_LEVEL.lower())

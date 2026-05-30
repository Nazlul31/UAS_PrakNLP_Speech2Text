import json
import os
import re
import time
from datetime import date
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import TypeAdapter

try:
    from .utils import normalize_text
except ImportError:
    from utils import normalize_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(DOTENV_PATH, override=True)

STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")
os.makedirs(STORAGE_DIR, exist_ok=True)
CHAT_HISTORY_FILE = os.path.join(STORAGE_DIR, "chat_history.json")
RATE_STATE_FILE = os.path.join(STORAGE_DIR, "rate_state.json")

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemma-4-26b-a4b-it")
RPM_LIMIT = int(os.getenv("GEMINI_RPM_LIMIT", "10"))
RPD_LIMIT = int(os.getenv("GEMINI_RPD_LIMIT", "1000"))
REQUEST_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "60"))
MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "3"))

if not GOOGLE_API_KEY:
    raise RuntimeError("GEMINI_API_KEY belum ditemukan. Buat file .env di root project.")

# Memperketat instruksi agar model langsung menjawab tanpa analisis tersembunyi
system_instruction = """
You are a responsive, intelligent, and fluent virtual assistant.
You answer voice-chat input from multilingual code-switching speech.

Rules:
- Default language: Indonesian.
- Answer DIRECTLY to the point. Do NOT output any analysis, thoughts, goals, or translation blocks.
- Keep answers polite, clear, and complete, maximum 2-3 sentences.
- Do not repeat the user's question.
- If the input is unclear, ask one short clarification question.
- If you do not know the answer, say honestly that you do not know.
""".strip()

client = genai.Client(api_key=GOOGLE_API_KEY)

# Menaikkan max_output_tokens menjadi 512 agar teks jawaban tidak terpotong di tengah kalimat
chat_config = types.GenerateContentConfig(
    system_instruction=system_instruction,
    temperature=0.4,  # Menurunkan temperatur sedikit agar jawaban lebih fokus dan konsisten
    max_output_tokens=512,
    http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000),
)
history_adapter = TypeAdapter(list[types.Content])


def _read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return default
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return default


def _write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _wait_for_rate_limit() -> None:
    """Pembatas sederhana agar tidak melebihi RPM dan RPD lokal."""
    now = time.time()
    today = date.today().isoformat()
    state = _read_json(RATE_STATE_FILE, {"date": today, "daily_count": 0, "timestamps": []})

    if state.get("date") != today:
        state = {"date": today, "daily_count": 0, "timestamps": []}

    if state.get("daily_count", 0) >= RPD_LIMIT:
        raise RuntimeError(f"RPD lokal tercapai ({RPD_LIMIT}). Coba lagi besok atau naikkan limit di .env.")

    timestamps = [t for t in state.get("timestamps", []) if now - float(t) < 60]
    if len(timestamps) >= RPM_LIMIT:
        sleep_time = 60 - (now - min(timestamps)) + 1
        print(f"[INFO] RPM lokal tercapai. Sleep {sleep_time:.1f} detik...")
        time.sleep(max(1, sleep_time))
        now = time.time()
        timestamps = [t for t in timestamps if now - float(t) < 60]

    timestamps.append(now)
    state["timestamps"] = timestamps
    state["daily_count"] = state.get("daily_count", 0) + 1
    _write_json(RATE_STATE_FILE, state)


def _extract_retry_delay_seconds(error: Exception) -> int:
    message = str(error)
    match = re.search(r"retry in ([0-9.]+)s", message, re.IGNORECASE)
    if match:
        return max(1, int(float(match.group(1))) + 1)
    if "429" in message or "quota" in message.lower() or "rate" in message.lower():
        return 60
    return 5


def export_chat_history(chat) -> str:
    return history_adapter.dump_json(chat.get_history()).decode("utf-8")


def save_chat_history(chat) -> None:
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as file:
        file.write(export_chat_history(chat))


def load_chat_history():
    if not os.path.exists(CHAT_HISTORY_FILE) or os.path.getsize(CHAT_HISTORY_FILE) == 0:
        return client.chats.create(model=MODEL, config=chat_config)

    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as file:
            json_str = file.read().strip()
        if not json_str:
            return client.chats.create(model=MODEL, config=chat_config)
        history = history_adapter.validate_json(json_str)
        return client.chats.create(model=MODEL, config=chat_config, history=history)
    except Exception as exc:
        print(f"[ERROR] Gagal load history chat: {exc}")
        return client.chats.create(model=MODEL, config=chat_config)


# Memaksa pembuatan sesi chat baru yang bersih
chat = client.chats.create(model=MODEL, config=chat_config)


def generate_response(prompt: str) -> str:
    normalized_prompt = normalize_text(prompt)
    if not normalized_prompt:
        return "Maaf, saya belum menerima teks yang jelas dari audio."

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _wait_for_rate_limit()
            print(f"[DEBUG-LLM] Attempt {attempt}/{MAX_RETRIES}: Sending to Gemini API...")
            response = chat.send_message(normalized_prompt)
            print(f"[DEBUG-LLM] API returned response object: {type(response)}")
            
            # Try multiple extraction paths
            resp_text = None
            
            # Path 1: Direct response.text
            if hasattr(response, 'text') and response.text:
                resp_text = response.text
                print(f"[DEBUG-LLM] [PATH-1] Got text from response.text: '{resp_text[:100]}'")
            
            # Path 2: response.candidates[0].content.parts[0].text
            elif hasattr(response, 'candidates') and response.candidates:
                try:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        if candidate.content.parts:
                            part = candidate.content.parts[0]
                            if hasattr(part, 'text') and part.text:
                                resp_text = part.text
                                print(f"[DEBUG-LLM] [PATH-2] Got text from candidates[0].content.parts[0].text: '{resp_text[:100]}'")
                except (IndexError, AttributeError) as e:
                    print(f"[DEBUG-LLM] [PATH-2] Failed to extract: {e}")
            
            # Path 3: Check finish_reason for blocks
            if not resp_text and hasattr(response, 'candidates'):
                try:
                    candidate = response.candidates[0]
                    finish_reason = candidate.finish_reason if hasattr(candidate, 'finish_reason') else None
                    print(f"[DEBUG-LLM] [PATH-3] Response blocked? finish_reason={finish_reason}")
                    if hasattr(candidate, 'safety_ratings'):
                        print(f"[DEBUG-LLM] [PATH-3] safety_ratings: {candidate.safety_ratings}")
                except Exception as e:
                    print(f"[DEBUG-LLM] [PATH-3] Error checking finish_reason: {e}")
            
            # Debug: Print full response structure
            if not resp_text:
                print(f"[DEBUG-LLM] Full response object:")
                print(f"  - dir(response): {[x for x in dir(response) if not x.startswith('_')]}")
                print(f"  - response.__dict__: {response.__dict__ if hasattr(response, '__dict__') else 'N/A'}")
            
            if resp_text and resp_text.strip():
                # ========================================================
                # PIPELINE PEMBERSIHAN (CLEANING) UNTUK MENGAMANKAN TTS
                # ========================================================
                
                # 1. Jika model mengembalikan format panjang bercampur analisis, coba ambil teks setelah salam "Waalaikumsalam"
                if "Waalaikumsalam" in resp_text:
                    match_salam = re.search(r"(Waalaikumsalam.*)", resp_text, re.DOTALL | re.IGNORECASE)
                    if match_salam:
                        resp_text = match_salam.group(1)
                
                # 2. Bersihkan karakter markdown aneh (*, #, ") agar TTS tidak crash
                resp_text = resp_text.replace("*", "").replace("#", "").replace('"', '').replace('(', '').replace(')', '')
                
                # 3. Potong jika teks jawaban melebihi batasan kalimat wajar (Maksimal 4 kalimat awal)
                sentences = re.split(r'(?<=[.!?])\s+', resp_text)
                resp_text = " ".join(sentences[:4])
                
                # ========================================================

                normalized = normalize_text(resp_text)
                print(f"[DEBUG-LLM] After clean & normalize_text: '{normalized[:100]}'")
                save_chat_history(chat)
                return normalized
            else:
                print(f"[DEBUG-LLM] response.text is None/empty, will retry...")
                
        except Exception as exc:
            last_error = exc
            delay = _extract_retry_delay_seconds(exc)
            print(f"[WARNING] Gemini gagal attempt {attempt}/{MAX_RETRIES}: {exc}")
            import traceback
            traceback.print_exc()
            if attempt < MAX_RETRIES:
                time.sleep(delay)

    # Fallback response jika semua attempt gagal atau response kosong
    fallback = f"Maaf, saya tidak bisa memproses permintaan Anda saat ini."
    print(f"[DEBUG-LLM] All attempts exhausted, returning fallback: '{fallback}'")
    return fallback
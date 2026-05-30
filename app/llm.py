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

system_instruction = """
You are a direct conversational virtual assistant.
Task: Answer the user's question immediately.

STRICT RULES:
1. Output ONLY the final conversational answer in Indonesian.
2. Absolutely NO explanations, NO drafts, NO thoughts, NO format logs, NO self-corrections, and NO analysis text.
3. Maximum 2-3 sentences.
""".strip()

client = genai.Client(api_key=GOOGLE_API_KEY)

chat_config = types.GenerateContentConfig(
    system_instruction=system_instruction,
    temperature=0.3,
    max_output_tokens=1024,
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


def generate_response(prompt: str) -> str:
    normalized_prompt = normalize_text(prompt)
    if not normalized_prompt:
        return "Maaf, saya belum menerima teks yang jelas dari audio."

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _wait_for_rate_limit()
            print(f"[DEBUG-LLM] Attempt {attempt}/{MAX_RETRIES}: Sending to Gemini API (Stateless)...")
            
            # Eksekusi secara stateless via generate_content agar ingatan audio terbilas sempurna
            response = client.models.generate_content(
                model=MODEL,
                contents=normalized_prompt,
                config=chat_config
            )
            print(f"[DEBUG-LLM] API returned response object: {type(response)}")
            
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
            
            if resp_text and resp_text.strip():
                # ========================================================
                # PIPELINE PEMBERSIHAN
                # ========================================================
                
                # 1. Bersihkan karakter markdown pengganggu
                resp_text = resp_text.replace("*", "").replace("#", "").replace('"', '').replace('(', '').replace(')', '')
                
                # 2. PECAH PER BARIS & CEK JAWABAN DARI BAWAH (BOTTOM-UP FILTER)
                lines = [line.strip() for line in resp_text.split('\n') if line.strip()]
                
                final_answer = ""
                # Kita periksa baris dari urutan paling bawah (paling akhir)
                for line in reversed(lines):
                    line_lower = line.lower()
                    
                    # Abaikan baris jika mengandung tanda titik dua pembuka log/draf (seperti 'Option satu:', 'Context:', dll)
                    # ATAU mengandung kata kunci draf yang sudah kita ketahui
                    if ":" in line or any(x in line_lower for x in [
                        'user input', 'intent', 'goal', 'constraints', 'language', 
                        'draft', 'refining', 'self-correction', 'user asks', 
                        'option', 'direct?', 'no analysis', 'polite/clear', 'context', 'translation'
                    ]):
                        continue
                    
                    # Baris pertama yang bersih dari bawah adalah JAWABAN ASLI yang kita cari
                    final_answer = line
                    break
                
                # Fallback jika semua baris ternyata tercemar log (sangat jarang terjadi)
                if not final_answer and lines:
                    final_answer = lines[-1]
                
                resp_text = final_answer
                
                # ========================================================
                # PENANGANAN KALIMAT GANTUNG (ANTI-TRUNCATION)
                # ========================================================
                if resp_text:
                    # Cari tanda baca akhir terakhir (. atau ? atau !)
                    last_punctuation_index = max(
                        resp_text.rfind('.'), 
                        resp_text.rfind('?'), 
                        resp_text.rfind('!')
                    )
                    # Jika ada tanda baca ditemukan, potong teks tepat di tanda baca tersebut
                    if last_punctuation_index != -1:
                        resp_text = resp_text[:last_punctuation_index + 1].strip()

                # ========================================================

                normalized = normalize_text(resp_text)
                print(f"[DEBUG-LLM] After clean & normalize_text: '{normalized[:100]}'")
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

    # Fallback response jika semua attempt gagal atau kosong
    fallback = f"Maaf, saya tidak bisa memproses permintaan Anda saat ini."
    print(f"[DEBUG-LLM] All attempts exhausted, returning fallback: '{fallback}'")
    return fallback
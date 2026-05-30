import os
import re
from typing import Optional
from num2words import num2words

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)


def convert_numbers_to_text(text: str) -> str:
    """
    Mengubah angka numerik di dalam string menjadi kata terbilang (Bahasa Indonesia).
    Contoh: "450" -> "empat ratus lima puluh"
    """
    number_pattern = re.compile(r'\d+')
    
    def replace_with_words(match):
        number_str = match.group(0)
        try:
            return num2words(int(number_str), lang='id')
        except Exception:
            return number_str

    return number_pattern.sub(replace_with_words, text)


def normalize_text(text: str) -> str:
    """Normalisasi sebelum teks dikirim ke LLM dan TTS."""
    if not text:
        return ""

    text = text.strip()
    
    # Tambahan: Konversi angka numerik menjadi kata terbilang bahasa Indonesia
    text = convert_numbers_to_text(text)
    
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", text)
    return text


def safe_delete(path: Optional[str]) -> None:
    """Hapus file sementara tanpa menghentikan aplikasi jika gagal."""
    if not path:
        return
    try:
        if os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
    except Exception as exc:
        print(f"[WARNING] Gagal menghapus file sementara {path}: {exc}")


def get_file_ext(filename: str, default: str = ".wav") -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext if ext else default
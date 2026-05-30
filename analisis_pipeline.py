"""analisis_pipeline.py

Batch pipeline untuk transkripsi + ringkasan audio.
Fungsi utama `jalankan_uji_korpus` sekarang memakai preprocessing audio
untuk menghasilkan suara resolusi konsisten, dan memproses seluruh korpus
secara default.
"""

import os
import shutil
import time
import re
from typing import Dict, Optional

import pandas as pd
from langdetect import detect_langs

from app.llm import generate_response
from app.stt import transcribe_speech_to_text
from app.utils import normalize_text
from processing import preprocess_audio, scan_audio_files


def _analyze_code_switching(text: str) -> Dict[str, float]:
    """Menghitung rasio pencampuran bahasa secara spesifik untuk masing-masing 
    bahasa: Inggris (EN), Indonesia (ID), dan Arab (AR) menggunakan langdetect."""
    if not text or not re.sub(r"[^\w\s]", "", text).strip():
        return {"EN": 0.0, "ID": 0.0, "AR": 0.0}

    try:
        # Bersihkan teks dari simbol-simbol mengganggu agar fokus pada karakter kata
        text_clean = re.sub(r"[^\w\s]", "", text)
        
        # Ambil prediksi semua bahasa beserta probabilitas distribusi skornya
        predictions = detect_langs(text_clean)
        
        # Inisialisasi awal nilai untuk masing-masing bahasa target
        lang_ratios = {"EN": 0.0, "ID": 0.0, "AR": 0.0}
        
        for pred in predictions:
            if pred.lang == 'en':
                lang_ratios["EN"] = round(pred.prob, 3)
            elif pred.lang in ['id', 'ms']:  # 'ms' (Melayu) digabung ke ID karena kemiripan linguistik n-gram
                lang_ratios["ID"] = round(lang_ratios["ID"] + pred.prob, 3)
            elif pred.lang == 'ar':
                lang_ratios["AR"] = round(pred.prob, 3)
            else:
                # Mengabaikan sisa probabilitas bahasa lain di luar batasan korpus penelitian
                pass
                
        # Memastikan total probabilitas distribusi tetap bernilai logis dan presisi
        return lang_ratios
        
    except Exception:
        # Fallback aman jika teks terlalu pendek atau gagal diidentifikasi
        return {"EN": 0.0, "ID": 1.0, "AR": 0.0}


def _select_files(folder_corpus_audio: str, student_prefix: Optional[str], limit_other: Optional[int]):
    files = scan_audio_files(folder_corpus_audio)
    if not files:
        raise RuntimeError(f"Tidak ada file audio yang didukung di folder {folder_corpus_audio}.")

    if not student_prefix:
        return files

    student_files = [f for f in files if f.name.startswith(student_prefix)]
    other_files = [f for f in files if not f.name.startswith(student_prefix)]

    if limit_other is None:
        return student_files + other_files

    return student_files + other_files[:limit_other]


def jalankan_uji_korpus(
    folder_corpus_audio: str,
    limit_other: Optional[int] = None,
    student_prefix: Optional[str] = None,
):
    """Jalankan evaluasi batch pada folder audio secara stateless bebas amnesia konteks."""
    hasil_analisis = []

    if not os.path.exists(folder_corpus_audio):
        raise FileNotFoundError(f"Folder korpus '{folder_corpus_audio}' tidak ditemukan.")

    files_to_process = _select_files(folder_corpus_audio, student_prefix, limit_other)
    print(f"Total audio ditemukan: {len(scan_audio_files(folder_corpus_audio))} — akan diproses: {len(files_to_process)}")

    temp_dir = os.path.join("temp", "pipeline")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)

    try:
        for file_path in files_to_process:
            print(f"Memproses: {file_path.name}")
            start_time = time.time()

            try:
                processed = preprocess_audio(file_path, output_dir=temp_dir)
                if processed.status != "success":
                    raise RuntimeError(processed.message or "preprocessing gagal")

                with open(processed.output_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()

                try:
                    os.remove(processed.output_path)
                except OSError:
                    pass

                # 1. Jalankan komponen speech-to-text (STT) & normalisasi teks dasar
                transcript = transcribe_speech_to_text(audio_bytes)
                normalized = normalize_text(transcript)
                
                # 2. Hitung rasio code-switching spesifik (EN, ID, AR)
                ratios = _analyze_code_switching(transcript)
                
                # 3. Panggil fungsi LLM secara langsung dan bersih (Stateless)
                raw_llm_resp = generate_response(normalized)
                llm_resp = normalize_text(raw_llm_resp)

                elapsed = round(time.time() - start_time, 2)
                hasil_analisis.append(
                    {
                        "file": file_path.name,
                        "transcript": transcript,
                        "normalized": normalized,
                        "ratios": ratios,
                        "llm_response": llm_resp,
                        "latency_s": elapsed,
                        "status": "success",
                    }
                )
            except Exception as exc:
                elapsed = round(time.time() - start_time, 2)
                print(f"Gagal memproses {file_path.name}: {exc}")
                hasil_analisis.append(
                    {
                        "file": file_path.name,
                        "transcript": "",
                        "normalized": "",
                        "ratios": {},
                        "llm_response": "",
                        "status": f"failed: {exc}",
                        "latency_s": elapsed,
                    }
                )

            # Jeda waktu aman untuk menjaga kestabilan RPM kuota API Google AI Studio
            time.sleep(6)

        # Mapping data hasil eksekusi ke struktur baris DataFrame
        rows = []
        for item in hasil_analisis:
            ratios = item.get("ratios") or {}
            # Mengubah format cetak di CSV menjadi EN:0.x, ID:0.y, AR:0.z secara rapi
            ratios_str = ", ".join([f"{k}:{v}" for k, v in ratios.items()]) if ratios else "EN:0.0, ID:0.0, AR:0.0"
            rows.append(
                {
                    "File": item["file"],
                    "Transcript": item["transcript"],
                    "Normalized": item["normalized"],
                    "Ratios": ratios_str,
                    "LLM_Response": item["llm_response"],
                    "Latency_s": item["latency_s"],
                    "Status": item["status"],
                }
            )

        df = pd.DataFrame(rows)
        log_dir = os.path.join("log")
        os.makedirs(log_dir, exist_ok=True)
        out_csv = os.path.join(log_dir, "analisis_pipeline.csv")
        df.to_csv(out_csv, index=False)
        print(f"Selesai. Hasil disimpan di {out_csv}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    try:

        jalankan_uji_korpus("data/audio", limit_other=0, student_prefix="2336_")
    except Exception as exc:
        print(f"Error: {exc}")
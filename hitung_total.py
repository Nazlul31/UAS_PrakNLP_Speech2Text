import os
import re
import pandas as pd

from bersihkan_data import REFERENCE_TRANSCRIPTS

def _levenshtein(s1: list, s2: list) -> int:
    """Menghitung jarak edit Levenshtein antara dua list."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = range(len(s2) + 1)
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _clean_for_wer(text: str) -> str:
    """Standardisasi teks dasar (hapus harakat & tanda baca) agar perhitungan adil."""
    if not text:
        return ""
    # Hapus harakat Arab jika ada
    text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)
    # Hapus tanda baca umum
    text = re.sub(r'[.,!?؛،؟"\'\-_]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


def hitung_skor_keseluruhan_dari_csv(path_csv: str):
    if not os.path.exists(path_csv):
        print(f"[ERROR] File '{path_csv}' tidak ditemukan!")
        print("Pastikan file CSV hasil running pipeline sudah ada di folder log.")
        return

    df = pd.read_csv(path_csv)
    
    # Filter hanya data yang sukses diproses
    df_success = df[df["Status"].str.lower() == "success"]

    if df_success.empty:
        print("[WARNING] Tidak ada data dengan status 'success' di dalam file CSV.")
        return

    total_word_distance = 0
    total_word_ref_length = 0

    total_char_distance = 0
    total_char_ref_length = 0

    for _, row in df_success.iterrows():
        file_name = row["File"]
        
        # Cari pola audioXX (misal audio01) dari nama file di CSV
        match_key = re.search(r"(audio\d+)", str(file_name).lower())
        if not match_key:
            continue
        key_jawaban = match_key.group(1)
        
        ref_text = REFERENCE_TRANSCRIPTS.get(key_jawaban, "")
        hyp_text = str(row["Normalized"]) if pd.notna(row["Normalized"]) else ""

        if not ref_text:
            continue

        # 1. Akumulasi Jarak Kata (Untuk WER Total)
        r_words = _clean_for_wer(ref_text).split()
        h_words = _clean_for_wer(hyp_text).split()
        
        total_word_distance += _levenshtein(r_words, h_words)
        total_word_ref_length += len(r_words)

        # 2. Akumulasi Jarak Karakter (Untuk CER Total)
        r_chars = list(_clean_for_wer(ref_text).replace(" ", ""))
        h_chars = list(_clean_for_wer(hyp_text).replace(" ", ""))
        
        total_char_distance += _levenshtein(r_chars, h_chars)
        total_char_ref_length += len(r_chars)

    # Rumus Akumulasi Riil Korpus (Total Jarak / Total Panjang Ref)
    wer_total = round(total_word_distance / total_word_ref_length, 4) if total_word_ref_length else 0.0
    cer_total = round(total_char_distance / total_char_ref_length, 4) if total_char_ref_length else 0.0

    # Batasi maksimal error di angka 1.0 (100% error)
    wer_total = min(1.0, wer_total)
    cer_total = min(1.0, cer_total)

    print("\n=======================================================")
    print("      HASIL AKURASI KESELURUHAN (TOTAL KORPUS)         ")
    print("=======================================================")
    print(f" Sampel Sukses Diolah : {len(df_success)} file audio")
    print("-------------------------------------------------------")
    print(f" Total Kata Referensi  : {total_word_ref_length} kata")
    print(f" TOTAL WER KORPUS      : {wer_total} ({round(wer_total * 100, 2)}%)")
    print("-------------------------------------------------------")
    print(f" Total Char Referensi  : {total_char_ref_length} karakter")
    print(f" TOTAL CER KORPUS      : {cer_total} ({round(cer_total * 100, 2)}%)")
    print("=======================================================\n")


if __name__ == "__main__":
    # Jalankan mengarah ke file hasil CSV pipeline utama kamu
    PATH_LAPORAN = os.path.join("log", "analisis_pipeline.csv")
    hitung_skor_keseluruhan_dari_csv(PATH_LAPORAN)
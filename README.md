# Sistem Asisten Suara Interaktif (Multilingual Speech-to-Speech)

Repositori ini berisi implementasi proyek UAS untuk mata kuliah Pemrosesan Bahasa Alami (NLP). Sistem ini mengintegrasikan rangkaian pipeline pemrosesan ucapan end-to-end yang berjalan secara real-time, meliputi transkripsi suara, analisis linguistik, penalaran kontekstual, hingga sintesis vokal otomatis.

## Kapabilitas Sistem & Alur Pipeline

Aplikasi ini mengolah input audio melalui beberapa tahapan modular yang divisualisasikan langsung pada dashboard:
1. Speech-to-Text (STT): Mentranskrip rekaman ucapan secara lokal menggunakan arsitektur whisper.cpp (OpenAI Whisper biner).
2. Normalisasi Teks & Analisis Bahasa: Memperbaiki kata-kata kolokial/tidak baku menjadi leksikon formal, serta mendeteksi persentase bahasa campuran (code-switching) seperti Bahasa Indonesia (IND) dan Arab (AR).
3. Kontekstual Respons LLM: Memanfaatkan engine Google GenAI SDK (Gemini API) untuk merumuskan jawaban yang cerdas, padat, dan langsung ke inti pertanyaan secara konsisten.
4. Sintesis Suara (TTS): Mengonversi teks jawaban menjadi output vokal alami secara lokal menggunakan model Coqui TTS.

## Tata Letak Direktori Proyek

```text
NLP_Checkpoint2_UAS/
│
├── app/
│   ├── coqui_utils/          # Model dan config Coqui TTS
│   ├── temp/                 # Ruang penyimpanan temporer untuk file backend internal
│   ├── whisper.cpp/          # Repositori lokal Whisper untuk inferensi STT cepat
│   ├── llm.py                # Integrasi Gemini API
│   ├── main.py               # Endpoint utama FastAPI
│   ├── stt.py                # Transkripsi suara (whisper.cpp)
│   ├── tts.py                # TTS dengan Coqui
│   └── utils.py              # Fungsi pembantu penataan teks & normalisasi leksikon
│
├── data/
│   ├── audio/                # Direktori penyimpanan korpus audio input (.wav)
│   ├── output_audio/         # Hasil sintesis suara luaran dari komponen TTS
│   ├── processed/            # Rekam jejak atau berkas yang telah selesai diproses
│   └── ground_truth.json     # Berkas referensi teks asli untuk kalkulasi akurasi
│
├── gradio_app/
│   └── app.py                # Frontend dengan Gradio
│
├── log/
│   ├── analisis_pipeline.csv # Rekapitulasi evaluasi pipa pemrosesan secara real-time
│   └── server_errors.log     # Berkas pencatatan galat operasional server backend
│
├── storage/                  # Direktori penyimpanan state aplikasi (chat_history, rate-state)
│
├── temp/                     # Folder temporer global proyek
│
├── tests/                    # Direktori unit test
│   └── test_language_analysis.py # Berkas pengujian unit test untuk analisis bahasa
│
├── venv/                     # Lingkungan virtual Python (Virtual Environment)
├── .env                      # Menyimpan Gemini API Key
├── .gitignore                # Daftar berkas/folder yang diabaikan oleh Git
├── requirements.txt          # Daftar dependensi Python yang wajib diinstal
├── analisis_pipeline.py      # Skrip otomatisasi pengujian massal korpus gabungan
├── bersihkan_data.py         # Skrip utilitas pra-pemrosesan data korpus
├── hitung_total.py           # Skrip kalkulasi final metrik akurasi WER dan CER murni
├── processing.py             # Modul kustom untuk pemrosesan data audio
```

## Ketentuan Teknis & Konfigurasi Model

- Format Kontainer Audio: Seluruh pemrosesan gelombang sinyal suara wajib menggunakan ekstensi file format .wav.
- Pemetaan Transkripsi Fonetik: Untuk memicu pelafalan kata yang natural oleh generator audio (misalnya menghasilkan transkripsi fonem seperti dəˈnɡan), teks mentah dari Gemini API harus melalui tahap konversi fonetik terlebih dahulu sebelum disintesis.
- Rekomendasi Arsitektur STT: Sangat disarankan untuk memanfaatkan checkpoint model Whisper varian ggml-large-v3-turbo demi akurasi transkripsi yang optimal.
- Karakteristik Voice Speaker: Proses pembuatan output suara digital menggunakan model Coqui v1.2 yang dikombinasikan dengan karakteristik suara kustom pembaca (wibowo).

## Panduan Prosedur Eksekusi (Windows OS)

Ikuti langkah-langkah di bawah ini untuk menjalankan aplikasi menggunakan terminal Windows (PowerShell atau Command Prompt):

1. Aktivasi Environment & Jalankan Backend Server (FastAPI):
   Buka terminal PowerShell pada direktori utama proyek, pastikan virtual environment (venv) sudah aktif, lalu eksekusi Uvicorn:
   
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   
   *Backend server sekarang aktif dan siap melayani request pada port 8000.*

2. Jalankan Antarmuka Dashboard (Gradio Frontend):
   Buka jendela terminal baru (tetap aktifkan virtual environment venv terlebih dahulu), lalu jalankan skrip antarmuka dengan perintah:
   
   python gradio_app/app.py
   
   *Aplikasi web frontend otomatis berjalan. Silakan buka browser Anda dan akses alamat [http://127.0.0.1:7860](http://127.0.0.1:7860) untuk menguji sistem.*

3. Menjalankan Evaluasi Batch (Uji Korpus):
   Jika Anda ingin memproses seluruh dataset audio di folder `data/audio` secara otomatis dan menghasilkan laporannya, jalankan skrip evaluasi:
   
   python analisis_pipeline.py
   
   *Hasil analisis evaluasi korpus akan tersimpan secara otomatis di direktori `log/analisis_pipeline.csv`.*

## Informasi Akademik

Proyek aplikasi asisten suara ini disusun dan dikembangkan sebagai pemenuhan komponen penilaian UAS Praktikum Pemrosesan Bahasa Alami — Program Studi Informatika, Fakultas Matematika dan Ilmu Pengetahuan Alam, Universitas Syiah Kuala (Semester Genap 2025/2026).
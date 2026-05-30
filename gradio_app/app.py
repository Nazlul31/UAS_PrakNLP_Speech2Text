import base64
import os
import tempfile
import uuid

import gradio as gr
import requests
import scipy.io.wavfile


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BACKEND_URL = os.getenv("FASTAPI_URL", "http://localhost:8000/voice-chat")


# ---------------------------------------------------------------------------
# Helper: markup builders
# ---------------------------------------------------------------------------
def status_markup(state="idle", message="Menunggu input suara."):
    state_class = {
        "idle": "idle", "processing": "processing",
        "success": "success", "error": "error"
    }.get(state, "idle")
    return f"""
    <div class="status-wrapper {state_class}">
        <span class="status-indicator-dot"></span>
        <span class="status-message-text">{message}</span>
    </div>
    """


def language_tags_markup(tags):
    if not tags:
        return "<div class='empty-note'>Belum ada ujaran yang diproses.</div>"
    if isinstance(tags, str):
        return f"<div class='tag-cloud'>{tags}</div>"
    chips = []
    items = tags.items() if isinstance(tags, dict) else enumerate(tags)
    for key, value in items:
        chips.append(f"<span class='lang-chip'><b>{key}</b>: {value}</span>")
    return f"<div class='tag-cloud'>{''.join(chips)}</div>"


def ratio_text(ratios):
    if not ratios:
        return ""
    if isinstance(ratios, str):
        return ratios
    color_map = {"IND": "IND", "ID": "IND", "EN": "EN", "AR": "AR", "ID-Slang": "Slang"}
    return "  |  ".join(f"{color_map.get(lang, lang)}: {ratio}" for lang, ratio in ratios.items())


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------
def _write_temp_wav(sample_rate, audio_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        scipy.io.wavfile.write(tmpfile.name, sample_rate, audio_data)
        return tmpfile.name


def _save_response_audio(response):
    path = os.path.join(tempfile.gettempdir(), f"tts_output_{uuid.uuid4()}.wav")
    with open(path, "wb") as f:
        f.write(response.content)
    return path


def _save_base64_audio(audio_base64, session_id):
    path = os.path.join(tempfile.gettempdir(), f"gradio_res_{session_id}.wav")
    with open(path, "wb") as f:
        f.write(base64.b64decode(audio_base64))
    return path


# ---------------------------------------------------------------------------
# Pipeline (logic unchanged)
# ---------------------------------------------------------------------------
def voice_chat_pipeline(audio, mode):
    if audio is None:
        return (
            None, "", "",
            language_tags_markup(None), "", "",
            status_markup("idle", "Silakan rekam suara terlebih dahulu."),
            "Belum ada audio. Rekam suara, lalu tekan Proses Pipeline.",
        )

    sample_rate, audio_data = audio
    input_audio_path = _write_temp_wav(sample_rate, audio_data)

    try:
        with open(input_audio_path, "rb") as audio_file:
            files = {"file": ("voice.wav", audio_file, "audio/wav")}
            data = {"mode": mode}
            response = requests.post(
                f"{BACKEND_URL}?format=json", files=files, data=data, timeout=120
            )

        if response.status_code != 200:
            return (
                None, "", "",
                language_tags_markup("<span class='error-text'>Backend mengembalikan error.</span>"),
                "", "",
                status_markup("error", f"Backend error HTTP {response.status_code}."),
                response.text,
            )

        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            output_audio_path = _save_response_audio(response)
            return (
                output_audio_path,
                "Backend mengirim audio langsung tanpa metadata JSON.",
                "-",
                language_tags_markup("Metadata tagging tidak tersedia pada respons audio langsung."),
                "-",
                "Audio balasan berhasil dibuat. Klik play pada Tahap 5 untuk mendengarkan.",
                status_markup("success", "Pipeline sukses. Audio balasan siap diputar."),
                "Audio siap. Gunakan player pada Tahap 5.",
            )

        result = response.json()
        if result.get("status") not in {None, "success"}:
            return (
                None, "", "",
                language_tags_markup("<span class='error-text'>Pipeline gagal diproses.</span>"),
                "", "",
                status_markup("error", result.get("message", "Pipeline gagal diproses.")),
                result.get("message", "Pipeline gagal diproses."),
            )

        session_id = result.get("session_id", uuid.uuid4().hex)
        output_audio_path = None
        if result.get("audio_base64"):
            output_audio_path = _save_base64_audio(result["audio_base64"], session_id)

        user_text       = result.get("user_text") or result.get("transcription") or ""
        normalized_text = result.get("normalized_text") or ""
        language_tags   = language_tags_markup(result.get("language_tags"))
        ratios          = ratio_text(result.get("language_ratios"))
        llm_response    = result.get("llm_response") or result.get("response_text") or ""

        return (
            output_audio_path,
            user_text, normalized_text, language_tags, ratios, llm_response,
            status_markup("success", f"Pipeline sukses. Session ID: {session_id}"),
            "Audio balasan siap diputar pada Tahap 5." if output_audio_path else "Metadata sukses, tetapi audio tidak ditemukan.",
        )

    except requests.exceptions.Timeout:
        return (
            None, "", "",
            language_tags_markup("<span class='error-text'>Request timeout.</span>"),
            "", "",
            status_markup("error", "Backend terlalu lama merespons."),
            "Coba gunakan rekaman yang lebih pendek.",
        )
    except requests.exceptions.ConnectionError:
        return (
            None, "", "",
            language_tags_markup("<span class='error-text'>Backend tidak tersambung.</span>"),
            "", "",
            status_markup("error", "Tidak bisa terhubung ke backend FastAPI."),
            "Pastikan backend berjalan di localhost:8000.",
        )
    except Exception as exc:
        return (
            None, "", "",
            language_tags_markup("<span class='error-text'>Terjadi kesalahan.</span>"),
            "", "",
            status_markup("error", "Terjadi kesalahan saat menjalankan pipeline."),
            str(exc),
        )
    finally:
        if os.path.exists(input_audio_path):
            os.remove(input_audio_path)


# ---------------------------------------------------------------------------
# Gradio theme — light, clean teal palette
# ---------------------------------------------------------------------------
theme = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.neutral,
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "Inter", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Plus Jakarta Sans"), "monospace"],
).set(
    # body / page
    body_background_fill="#ffffff",
    body_background_fill_dark="#ffffff",
    body_text_color="#0f2926",
    body_text_color_dark="#0f2926",
    body_text_color_subdued="#507a75",
    body_text_color_subdued_dark="#507a75",
    # input / textbox
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_background_fill_focus="#ffffff",
    input_background_fill_focus_dark="#ffffff",
    input_border_color="#e2f0ed",
    input_border_color_dark="#e2f0ed",
    input_border_color_focus="#0d9488",
    input_border_color_focus_dark="#0d9488",
    input_placeholder_color="#99b5b2",
    input_placeholder_color_dark="#99b5b2",
    # block
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color="#e2f0ed",
    block_border_color_dark="#e2f0ed",
    block_label_text_color="#507a75",
    block_label_text_color_dark="#507a75",
    block_title_text_color="#0f2926",
    block_title_text_color_dark="#0f2926",
    block_shadow="0 10px 30px rgba(13,148,136,0.03)",
    block_radius="16px",
    # panel
    panel_background_fill="#f0fdfa",
    panel_background_fill_dark="#f0fdfa",
    # button primary
    button_primary_background_fill="linear-gradient(135deg,#0d9488,#14b8a6)",
    button_primary_background_fill_dark="linear-gradient(135deg,#0d9488,#14b8a6)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_primary_border_color="transparent",
    button_primary_border_color_dark="transparent",
    # button secondary
    button_secondary_background_fill="#f0fdfa",
    button_secondary_background_fill_dark="#f0fdfa",
    button_secondary_text_color="#0d9488",
    button_secondary_text_color_dark="#0d9488",
    button_secondary_border_color="#e2f0ed",
    button_secondary_border_color_dark="#e2f0ed",
    # radio / checkbox
    checkbox_background_color="#ffffff",
    checkbox_background_color_dark="#ffffff",
    checkbox_border_color="#e2f0ed",
    checkbox_border_color_dark="#e2f0ed",
    checkbox_label_background_fill="#f0fdfa",
    checkbox_label_background_fill_dark="#f0fdfa",
    checkbox_label_text_color="#0f2926",
    checkbox_label_text_color_dark="#0f2926",
    # border radius
    input_radius="12px",
    button_large_radius="999px",
    button_small_radius="999px",
    # color accent
    color_accent="#0d9488",
    color_accent_soft="#f0fdfa",
    color_accent_soft_dark="#f0fdfa",
    # error
    error_background_fill="#fff5f5",
    error_border_color="#fecaca",
    error_text_color="#b91c1c",
)


# ---------------------------------------------------------------------------
# CSS overrides
# ---------------------------------------------------------------------------
css = """
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Outfit:wght@300;400;500;600;700;900&display=swap');

/* ── Tokens ── */
:root {
  --bg:            #ffffff;
  --surface:       #ffffff;
  --surface-soft:  #f0fdfa;
  --surface-muted: #e2f8f5;
  --text:          #0f2926;
  --muted:         #507a75;
  --line:          #e2f0ed;
  --brand:         #0d9488;
  --brand-2:       #0f766e;
  --brand-3:       #14b8a6;
  --error:         #ef4444;
  --success:       #10b981;
  --shadow:        0 16px 40px rgba(13,148,136,0.04);
  --shadow-soft:   0 8px 24px rgba(13,148,136,0.03);
}

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; }

/* ── Force light background everywhere in Gradio ── */
html, body {
  background: var(--bg) !important;
  color: var(--text) !important;
}

.gradio-container {
  min-height: 100vh;
  padding: 0 24px 56px !important;
  background:
    radial-gradient(circle at 10% 20%, rgba(20,184,166,0.05) 0%, transparent 40%),
    radial-gradient(circle at 90% 80%, rgba(13,148,136,0.03) 0%, transparent 45%),
    var(--bg) !important;
}

/* Kill default Gradio borders on top-level container */
.block, .form { border: none !important; box-shadow: none !important; background: transparent !important; }

footer, .footer { display: none !important; }
.contain { max-width: none !important; }

/* Make all Gradio textboxes light */
textarea, input[type="text"], input[type="number"] {
  background: #ffffff !important;
  color: var(--text) !important;
  border-color: var(--line) !important;
}

label span, .svelte-1f354aw { color: var(--muted) !important; }

/* ── App shell ── */
.app-shell {
  width: min(1300px, 100%);
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

/* ════════════════════════════════════════════
   NAVBAR (Minimalist Top Bar)
   ════════════════════════════════════════════ */
.app-navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 16px;
  margin-top: 20px;
  box-shadow: var(--shadow-soft);
}

.nav-brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.nav-logo {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  background: linear-gradient(135deg, var(--brand), var(--brand-3));
  display: grid;
  place-items: center;
  color: #ffffff;
  box-shadow: 0 4px 12px rgba(13,148,136,0.2);
}

.nav-logo svg {
  width: 20px;
  height: 20px;
}

.nav-title-group {
  display: flex;
  flex-direction: column;
}

.nav-title {
  font-family: "Outfit", sans-serif;
  font-size: 18px;
  font-weight: 800;
  color: var(--text);
  line-height: 1.2;
}

.nav-subtitle {
  font-size: 11px;
  color: var(--muted);
  font-weight: 600;
  letter-spacing: 0.02em;
}

.nav-badges {
  display: flex;
  align-items: center;
  gap: 8px;
}

.nav-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 8px;
  background: var(--surface-soft);
  border: 1px solid var(--line);
  color: var(--brand);
  font-size: 10.5px;
  font-weight: 800;
}

.nav-badge .badge-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background-color: var(--brand);
  animation: pulse-ring 2s cubic-bezier(0.455, 0.03, 0.515, 0.955) infinite;
}

@keyframes pulse-ring {
  0% { transform: scale(0.8); opacity: 0.5; }
  50% { transform: scale(1.2); opacity: 1; }
  100% { transform: scale(0.8); opacity: 0.5; }
}

/* ════════════════════════════════════════════
   LAYOUT (3-Column Grid)
   ════════════════════════════════════════════ */
.layout-3col {
  display: grid !important;
  grid-template-columns: 28% 36% 36%;
  gap: 20px;
  align-items: start;
}

/* ════════════════════════════════════════════
   CARDS & PANELS
   ════════════════════════════════════════════ */
.card-panel {
  border: 1px solid var(--line) !important;
  border-radius: 20px !important;
  background: #ffffff !important;
  box-shadow: var(--shadow) !important;
  padding: 24px !important;
  min-height: 100%;
}

.panel-header {
  border-bottom: 1px solid var(--line);
  padding-bottom: 16px;
  margin-bottom: 20px;
}

.panel-title-text {
  font-family: "Outfit", sans-serif;
  font-size: 16px;
  font-weight: 800;
  color: var(--text);
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.panel-title-desc {
  font-size: 12px;
  color: var(--muted);
  margin: 4px 0 0;
  font-weight: 500;
  line-height: 1.4;
}

/* ── Voice Visual Circle ── */
.voice-controller-card {
  padding: 16px;
  border-radius: 16px;
  border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(20,184,166,0.05) 0%, rgba(13,148,136,0.02) 100%);
  margin-bottom: 16px;
  display: flex;
  justify-content: center;
  align-items: center;
}

.pulsing-orb-container {
  height: 120px;
  display: grid;
  place-items: center;
  position: relative;
  width: 100%;
}

.pulsing-ring {
  position: absolute;
  width: 90px;
  height: 90px;
  border-radius: 50%;
  background: rgba(13, 148, 136, 0.1);
  animation: pulse-glow 3s infinite ease-in-out;
}

.pulsing-ring-2 {
  position: absolute;
  width: 110px;
  height: 110px;
  border-radius: 50%;
  background: rgba(20, 184, 166, 0.05);
  animation: pulse-glow 3s infinite ease-in-out;
  animation-delay: 1.5s;
}

.mic-orb {
  position: relative;
  width: 70px;
  height: 70px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--brand), var(--brand-3));
  display: grid;
  place-items: center;
  box-shadow: 0 10px 25px rgba(13, 148, 136, 0.3);
  z-index: 2;
}

.mic-orb svg {
  width: 30px;
  height: 30px;
  fill: none;
  stroke: #ffffff;
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}

@keyframes pulse-glow {
  0% { transform: scale(0.9); opacity: 0.3; }
  50% { transform: scale(1.15); opacity: 0.8; }
  100% { transform: scale(0.9); opacity: 0.3; }
}

/* ── Audio Elements ── */
#audio-input, #audio-output {
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  background: #ffffff !important;
  box-shadow: none !important;
  overflow: hidden !important;
}

#audio-input > *, #audio-input div,
#audio-output > *, #audio-output div {
  background: #ffffff !important;
  color: var(--text) !important;
}

#audio-input *, #audio-output * {
  border-color: var(--line) !important;
  outline: none !important;
}

#audio-input .block, #audio-input .label-wrap,
#audio-output .block, #audio-output .label-wrap {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
}

#audio-input .tab-nav, #audio-input .tabs, #audio-input .tabitem,
#audio-input [role="tablist"], #audio-input [role="tab"],
#audio-input .svelte-tab-bar, #audio-input .wrap {
  border: none !important;
  background: #ffffff !important;
}

#audio-input [role="tab"][aria-selected="true"],
#audio-input [role="tab"].selected {
  border-bottom: 2px solid var(--brand) !important;
  color: var(--brand) !important;
  font-weight: 700 !important;
}

#audio-input [role="tab"]:not([aria-selected="true"]) {
  color: var(--muted) !important;
}

#audio-input button, #audio-output button,
#audio-input [role="button"], #audio-output [role="button"] {
  border: 1px solid var(--line) !important;
  border-radius: 8px !important;
  background: var(--surface-soft) !important;
  color: var(--brand) !important;
  font-family: inherit !important;
  font-weight: 600 !important;
  box-shadow: none !important;
}

#audio-input button:hover, #audio-output button:hover {
  background: var(--surface-muted) !important;
}

/* ── Radio buttons (System mode) ── */
.mode-container {
  margin-top: 14px;
}

#mode-select {
  border: 1px solid var(--line) !important;
  border-radius: 12px !important;
  background: var(--surface-soft) !important;
  padding: 10px !important;
  box-shadow: none !important;
}

#mode-select label, #mode-select span {
  font-weight: 700 !important;
  color: var(--text) !important;
}

#mode-select .wrap > label,
#mode-select [data-testid="radio-label"] {
  border: 1px solid var(--line) !important;
  border-radius: 8px !important;
  background: #ffffff !important;
  transition: all 0.2s ease !important;
}

#mode-select .wrap > label:hover,
#mode-select [data-testid="radio-label"]:hover {
  border-color: var(--brand) !important;
  background: var(--surface-soft) !important;
}

#mode-select .wrap > label:has(input:checked),
#mode-select [data-testid="radio-label"]:has(input:checked) {
  background: linear-gradient(135deg, var(--brand), var(--brand-3)) !important;
  border-color: transparent !important;
  color: #ffffff !important;
}

#mode-select .wrap > label:has(input:checked) span {
  color: #ffffff !important;
}

/* ── Submit Action Zone ── */
.action-zone {
  margin: 20px 0 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

#submit-button {
  width: 100% !important;
  height: 44px !important;
  border-radius: 22px !important;
  background: linear-gradient(135deg, var(--brand), var(--brand-3)) !important;
  color: #ffffff !important;
  font-weight: 800 !important;
  letter-spacing: 0.01em !important;
  box-shadow: 0 8px 20px rgba(13, 148, 136, 0.2) !important;
  border: none !important;
  transition: all 0.2s ease !important;
}

#submit-button:hover {
  transform: translateY(-1.5px) !important;
  box-shadow: 0 10px 24px rgba(13, 148, 136, 0.3) !important;
}

.action-hint {
  font-size: 11px;
  color: var(--muted);
  font-weight: 600;
  text-align: center;
}

.tip-card {
  padding: 12px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--surface-soft);
  color: var(--brand-2);
  font-size: 11.5px;
  line-height: 1.5;
  font-weight: 600;
  margin-top: 14px;
}

/* ── Pipeline step card ── */
.step-card {
  border: 1px solid var(--line) !important;
  border-radius: 14px !important;
  background: #ffffff !important;
  padding: 14px !important;
  margin-bottom: 14px !important;
  box-shadow: var(--shadow-soft) !important;
  transition: border-color 0.2s ease !important;
}

.step-card:hover {
  border-color: var(--brand-3) !important;
}

.step-card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}

.step-pill {
  width: 24px;
  height: 24px;
  border-radius: 6px;
  background: linear-gradient(135deg, var(--brand), var(--brand-3));
  color: #ffffff;
  display: grid;
  place-items: center;
  font-size: 11px;
  font-weight: 900;
}

.step-title {
  font-family: "Outfit", sans-serif;
  font-size: 13.5px;
  font-weight: 800;
  color: var(--text);
  margin: 0;
}

/* Textbox overrides inside step-cards */
.step-card textarea, .step-card input[type="text"] {
  background: #ffffff !important;
  color: var(--text) !important;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  font-size: 12.5px !important;
  font-weight: 600 !important;
}

.step-card label span {
  font-size: 11px !important;
  font-weight: 700 !important;
  color: var(--muted) !important;
}

/* ── Language Analytics styling ── */
.tag-cloud {
  min-height: 38px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: #ffffff;
  padding: 6px 10px;
  line-height: 1.6;
}

.lang-chip {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin: 2px;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--surface-soft);
  color: var(--brand-2);
  border: 1px solid var(--line);
  font-size: 11.5px;
  font-weight: 700;
}

.empty-note {
  color: var(--muted);
  font-style: italic;
  font-size: 12px;
  font-weight: 500;
}

.error-text {
  color: var(--error);
  font-weight: 700;
}

/* ── Status display card ── */
.status-wrapper {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 10px;
  border: 1px solid var(--line);
  background: var(--surface-soft);
  margin-bottom: 12px;
}

.status-indicator-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #94a3b8;
  flex-shrink: 0;
}

.status-message-text {
  color: var(--text) !important;
  font-size: 12.5px;
  font-weight: 700;
}

.status-wrapper.processing .status-indicator-dot {
  background: var(--brand-3);
  animation: pulse-ring 1.5s infinite;
}

.status-wrapper.success .status-indicator-dot {
  background: var(--success);
}

.status-wrapper.error .status-indicator-dot {
  background: var(--error);
}

.status-wrapper.error .status-message-text {
  color: var(--error) !important;
}

.status-wrapper.success .status-message-text {
  color: #065f46 !important;
}

#status-detail textarea {
  background: var(--surface-soft) !important;
  border-radius: 10px !important;
  font-size: 11.5px !important;
  color: var(--text) !important;
}

/* ── Footer ── */
.app-footer {
  text-align: center;
  font-size: 11px;
  color: var(--muted);
  font-weight: 700;
  letter-spacing: 0.05em;
  padding: 10px 0 20px;
  border-top: 1px solid var(--line);
  margin-top: 10px;
  text-transform: uppercase;
}

/* ── Responsive styling ── */
@media (max-width: 1024px) {
  .layout-3col {
    grid-template-columns: 35% 65%;
  }
  .layout-3col > div:last-child {
    grid-column: span 2;
  }
}

@media (max-width: 768px) {
  .layout-3col {
    grid-template-columns: 1fr;
  }
  .layout-3col > div:last-child {
    grid-column: span 1;
  }
  .app-navbar {
    flex-direction: column;
    gap: 12px;
    align-items: flex-start;
  }
  .nav-badges {
    width: 100%;
    justify-content: flex-start;
    flex-wrap: wrap;
  }
}
"""


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(theme=theme, css=css, title="S2S Voice Chatbot") as demo:
    with gr.Column(elem_classes="app-shell"):

        # ── Minimalist Navbar ────────────────────────────────────────────────
        gr.HTML("""
        <nav class="app-navbar">
            <div class="nav-brand">
                <div class="nav-logo">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 1 0 6 0V5a3 3 0 0 0-3-3Z"/>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                        <path d="M12 19v3"/>
                        <path d="M8 22h8"/>
                    </svg>
                </div>
                <div class="nav-title-group">
                    <span class="nav-title">AURA S2S</span>
                    <span class="nav-subtitle">Multilingual AI Voice Suite</span>
                </div>
            </div>
            <div class="nav-badges">
                <span class="nav-badge"><span class="badge-dot"></span>STT</span>
                <span class="nav-badge"><span class="badge-dot"></span>LID</span>
                <span class="nav-badge"><span class="badge-dot"></span>LLM</span>
                <span class="nav-badge"><span class="badge-dot"></span>TTS</span>
            </div>
        </nav>
        """)

        # ── Three-column layout ───────────────────────────────────────────────
        with gr.Row(elem_classes="layout-3col"):

            # ── COLUMN 1: Control & Input Hub ────────────────────────────────
            with gr.Column(elem_classes="card-panel"):
                gr.HTML("""
                <div class="panel-header">
                    <h2 class="panel-title-text">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 1 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><path d="M12 19v3"/><path d="M8 22h8"/></svg>
                        Voice Controller
                    </h2>
                    <p class="panel-title-desc">Rekam suara, atur mode respons, dan jalankan pipeline.</p>
                </div>
                """)

                gr.HTML("""
                <div class="voice-controller-card">
                    <div class="pulsing-orb-container" aria-hidden="true">
                        <div class="pulsing-ring"></div>
                        <div class="pulsing-ring-2"></div>
                        <div class="mic-orb">
                            <svg viewBox="0 0 24 24">
                                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 1 0 6 0V5a3 3 0 0 0-3-3Z"/>
                                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                                <path d="M12 19v3"/>
                                <path d="M8 22h8"/>
                            </svg>
                        </div>
                    </div>
                </div>
                """)

                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Pilih sumber audio dan rekam",
                    elem_id="audio-input",
                )

                with gr.Column(elem_classes="mode-container"):
                    mode_select = gr.Radio(
                        choices=["preserve", "normalized"],
                        value="preserve",
                        label="Gaya Respons Asisten",
                        info="preserve: bahasa campuran | normalized: formal",
                        elem_id="mode-select",
                    )

                gr.HTML('<div class="action-zone">')
                submit_btn = gr.Button("Proses Pipeline", variant="primary", elem_id="submit-button")
                gr.HTML('<span class="action-hint">Tekan tombol setelah selesai merekam audio.</span></div>')

                gr.HTML("""
                <div class="tip-card">
                    💡 <b>Tips Penggunaan:</b><br/>
                    Gunakan mikrofon berkualitas baik atau unggah berkas audio (WAV) untuk analisis yang optimal.
                </div>
                """)

            # ── COLUMN 2: Transcription & Diagnostics ──────────────────────────
            with gr.Column(elem_classes="card-panel"):
                gr.HTML("""
                <div class="panel-header">
                    <h2 class="panel-title-text">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21.21 15.89A10 10 0 1 1 8 2.83M22 12A10 10 0 0 0 12 2v10z"/></svg>
                        Real-time Diagnostics
                    </h2>
                    <p class="panel-title-desc">Transkripsi mentah, normalisasi, dan klasifikasi bahasa.</p>
                </div>
                """)

                # Step 1
                with gr.Column(elem_classes="step-card"):
                    gr.HTML("""
                    <div class="step-card-header">
                        <span class="step-pill">1</span>
                        <h3 class="step-title">Speech-to-Text Transcription</h3>
                    </div>
                    """)
                    out_user_text = gr.Textbox(
                        label="Ujaran Terdeteksi",
                        interactive=False,
                        placeholder="Menunggu transkripsi...",
                    )

                # Step 2
                with gr.Column(elem_classes="step-card"):
                    gr.HTML("""
                    <div class="step-card-header">
                        <span class="step-pill">2</span>
                        <h3 class="step-title">Leksikon Normalization</h3>
                    </div>
                    """)
                    out_normalized_text = gr.Textbox(
                        label="Hasil Normalisasi Kata",
                        interactive=False,
                        placeholder="Menunggu normalisasi kata...",
                    )

                # Step 3
                with gr.Column(elem_classes="step-card"):
                    gr.HTML("""
                    <div class="step-card-header">
                        <span class="step-pill">3</span>
                        <h3 class="step-title">Language & Code-Switching Detection</h3>
                    </div>
                    """)
                    out_language_tags = gr.HTML(
                        value=language_tags_markup(None),
                        label="Kata demi Kata Tagging",
                    )
                    out_ratios = gr.Textbox(
                        label="Rasio Distribusi Bahasa",
                        interactive=False,
                        placeholder="Menunggu klasifikasi bahasa...",
                    )

            # ── COLUMN 3: Response Engine & TTS ──────────────────────────────
            with gr.Column(elem_classes="card-panel"):
                gr.HTML("""
                <div class="panel-header">
                    <h2 class="panel-title-text">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                        AI Response Engine
                    </h2>
                    <p class="panel-title-desc">Output teks asisten dan pemutaran audio balasan.</p>
                </div>
                """)

                # Step 4
                with gr.Column(elem_classes="step-card"):
                    gr.HTML("""
                    <div class="step-card-header">
                        <span class="step-pill">4</span>
                        <h3 class="step-title">Asisten Response Text</h3>
                    </div>
                    """)
                    out_llm_response = gr.Textbox(
                        label="Respons Kontekstual LLM",
                        interactive=False,
                        placeholder="Menunggu respons teks...",
                        lines=4,
                    )

                # Step 5
                with gr.Column(elem_classes="step-card"):
                    gr.HTML("""
                    <div class="step-card-header">
                        <span class="step-pill">5</span>
                        <h3 class="step-title">TTS Speech Synthesis</h3>
                    </div>
                    """)
                    audio_output = gr.Audio(
                        type="filepath",
                        label="Balasan Suara Asisten",
                        interactive=False,
                        elem_id="audio-output",
                    )

                # Pipeline logs & status
                gr.HTML("""
                <div class="step-card-header" style="margin-top: 10px; margin-bottom: 5px;">
                    <h3 class="step-title" style="font-size: 12px; text-transform: uppercase; color: var(--muted); letter-spacing: 0.05em;">Log Status & Informasi</h3>
                </div>
                """)
                status_html = gr.HTML(status_markup())
                status_detail = gr.Textbox(
                    label="Detail Status Pipeline",
                    interactive=False,
                    value="System Idle",
                    elem_id="status-detail",
                )

        # ── Footer ────────────────────────────────────────────────────────────
        gr.HTML("<div class='pipeline-footer'>STT  —  Normalisasi  —  Language Tagging  —  LLM  —  TTS</div>")

    # ── Event wiring ─────────────────────────────────────────────────────────
    submit_btn.click(
        fn=voice_chat_pipeline,
        inputs=[audio_input, mode_select],
        outputs=[
            audio_output,
            out_user_text,
            out_normalized_text,
            out_language_tags,
            out_ratios,
            out_llm_response,
            status_html,
            status_detail,
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
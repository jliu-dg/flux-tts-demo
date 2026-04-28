"""Vapi Support Voice Demo - Main Flask app."""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import asyncio
import websockets
import os
import json
import threading
import base64
import time
import re
from urllib.parse import urlsplit, parse_qsl, urlencode, urlunsplit
import janus
import queue
import logging
import requests
from dotenv import load_dotenv
from typing import Optional

from common.agent_functions import (
    HOTWORD_FUNCTION_MAP,
    CHECK_HOTWORD_DEFINITION,
    CLOSE_HOTWORD_SESSION_DEFINITION,
    CLOSE_TRIGGERS,
    CLOSE_IGNORE,
    set_hotword,
    is_conversation_active,
    check_hotword,
)
from saga.functions import SAGA_FUNCTION_MAP, get_random_filler
from saga.definitions import SAGA_FUNCTION_DEFINITIONS
from saga.mock_data import get_city_state, reset_city_state

load_dotenv()

FUNCTION_MAP = {**HOTWORD_FUNCTION_MAP, **SAGA_FUNCTION_MAP}

VOICE_AGENT_URL = "wss://agent.deepgram.com/v1/agent/converse"
AUDIO_SAMPLE_RATE = 16000
TTS_OUTPUT_SAMPLE_RATE = 24000

AUDIO_SETTINGS = {
    "input": {"encoding": "linear16", "sample_rate": AUDIO_SAMPLE_RATE},
    "output": {"encoding": "linear16", "sample_rate": TTS_OUTPUT_SAMPLE_RATE, "container": "none"},
}

HOTWORD_BYPASS = {"check_hotword", "close_hotword_session"}

# Flask setup
app = Flask(__name__, static_folder="./static", static_url_path="/static")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(handler)
logging.getLogger().handlers = []

def emit_latency_event(stage: str, **details):
    payload = {
        "stage": stage,
        "ts_ms": int(time.time() * 1000),
        "details": details,
    }
    socketio.emit("latency_event", payload)

def emit_vapi_flow_event(phase: str, **details):
    socketio.emit("vapi_flow_event", {"phase": phase, "details": details, "ts_ms": int(time.time() * 1000)})


# ---------------------------------------------------------------------------
# Build Deepgram settings from the saga config
# ---------------------------------------------------------------------------

def load_saga_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "configs", "saga.json")
    with open(config_path) as f:
        return json.load(f)


def build_sandbox_tts_base_url(cfg: dict) -> str:
    requested_voice_model = cfg.get("voiceModel")
    if not requested_voice_model:
        raise ValueError("Missing required voiceModel in saga config.")

    endpoint = os.getenv("DEEPGRAM_TTS_ENDPOINT_URL", "").strip()
    if endpoint:
        return endpoint

    backend = cfg.get("voiceBackend", "vapi005")
    return f"https://tts.sandbox.deepgram.com/speak?backend={backend}&model={requested_voice_model}"

def get_pre_recorded_greeting_settings(cfg: dict) -> dict:
    settings = cfg.get("preRecordedGreeting", {})
    variants = settings.get("variants", [])
    by_id = {v.get("id"): v for v in variants if v.get("id") and v.get("text")}
    default_id = settings.get("defaultId")
    if default_id not in by_id and by_id:
        default_id = next(iter(by_id.keys()))
    return {
        "enabled": bool(settings.get("enabled", False) and by_id),
        "default_id": default_id,
        "variants": by_id,
    }


def build_disfluency_prompt(cfg: dict) -> str:
    settings = cfg.get("disfluency", {})
    enabled = settings.get("enabled", False)
    if not enabled:
        return ""

    level = os.getenv("SAGA_DISFLUENCY_LEVEL", settings.get("level", "light")).strip().lower()
    if level in {"off", "none", "disabled"}:
        return ""

    profiles = {
        "light": {
            "usage": "Use subtle disfluencies in about 20-30% of responses, max 1 disfluency per response.",
            "allowed": "Allowed forms: brief hesitations like 'well' or 'so', or a tiny self-correction like 'the grid is - it's stable.'",
        },
        "medium": {
            "usage": "Use natural disfluencies in about 40-50% of responses, max 1-2 disfluencies per response.",
            "allowed": "Allowed forms: brief hesitations, tiny restarts, and light self-corrections.",
        },
        "high": {
            "usage": "Use frequent disfluencies in about 60-70% of responses, max 2 disfluencies per response.",
            "allowed": "Allowed forms: hesitations, short repairs, and occasional repeated lead-in words.",
        },
    }
    profile = profiles.get(level, profiles["light"])

    return (
        "DISFLUENCY MODE:\n"
        "Integrate natural spoken disfluencies in assistant responses only.\n"
        f"{profile['usage']}\n"
        f"{profile['allowed']}\n"
        "Do NOT add disfluencies inside numbers, names, IDs, exact quotes, or critical commands.\n"
        "Keep responses concise and intelligible; clarity always wins over style."
    )


def build_expressive_prompt(cfg: dict, expressive_enabled_override: Optional[bool] = None) -> str:
    settings = cfg.get("expressive", {})
    if expressive_enabled_override is None:
        enabled = settings.get("enabled", False)
    else:
        enabled = expressive_enabled_override
    if not enabled:
        return ""

    mode = os.getenv("SAGA_EXPRESSIVE_MODE", settings.get("mode", "subtle")).strip().lower()
    if mode in {"off", "none", "disabled", "neutral"}:
        return ""

    profiles = {
        "subtle": {
            "style": "Use a slightly more human cadence with gentle emotional contour.",
            "limits": "Keep delivery calm and controlled; no dramatic punctuation bursts.",
        },
        "balanced": {
            "style": "Use clear expressive variation to match context: urgency for alerts, warmth for concierge tasks, confidence for executive updates.",
            "limits": "Stay concise and avoid theatrical phrasing.",
        },
        "vivid": {
            "style": "Use strong expressive variation and dynamic cadence while remaining natural.",
            "limits": "Do not overact and never reduce factual clarity or precision.",
        },
    }
    profile = profiles.get(mode, profiles["subtle"])

    return (
        "EXPRESSIVE MODE:\n"
        f"{profile['style']}\n"
        f"{profile['limits']}\n"
        "Prioritize semantic accuracy over style. Numbers, names, and commands must remain exact."
    )


def build_call_center_style_prompt(cfg: dict) -> str:
    settings = cfg.get("callCenterStyle", {})
    enabled = settings.get("enabled", False)
    if not enabled:
        return ""

    density = os.getenv("SAGA_CALLCENTER_DENSITY", settings.get("density", "medium")).strip().lower()
    if density in {"off", "none", "disabled"}:
        return ""

    density_rules = {
        "light": (
            "Use at least one speech marker in most turns and two in longer turns. "
            "Avoid overloading short confirmations."
        ),
        "medium": (
            "Use at least one speech marker in every turn, two in medium turns, and three in longer turns. "
            "Spread markers naturally across the turn."
        ),
        "high": (
            "Use dense spoken texture: one marker even in short turns, two to three in medium turns, "
            "and three or more in long turns, while staying intelligible."
        ),
    }
    chosen_density = density_rules.get(density, density_rules["medium"])

    return (
        "CALL-CENTER SPEAKING STYLE (PHONE VOICE):\n"
        "Sound like a competent, experienced support agent speaking spontaneously on a live call.\n"
        "No markdown, bullets, headers, or written-formality phrases.\n"
        "Use direct address with 'you' and 'I'.\n"
        "Spell numbers/times/durations the way humans say them aloud.\n"
        "Keep turns around thirty to seventy words; break long explanations into shorter speakable chunks.\n"
        "Do not read URLs, emails, or long alphanumeric strings unless explicitly asked.\n"
        "Speech texture is mandatory: discourse openers, occasional 'um/uh' before specific facts, "
        "trail-off and resume, period-pivot thinking beats, self-corrections, retrieval narration, "
        "reactive markers, and agreement backchannels.\n"
        f"{chosen_density}\n"
        "Avoid stacking fillers back-to-back.\n"
        "Guard against clean scripted prose; if a turn could be pasted into an email unchanged, rewrite it as speech.\n"
        "For apologies and bad news, keep texture but slightly softer."
    )


def build_identity_response_prompt(cfg: dict) -> str:
    return (
        "IDENTITY RESPONSE (HIGH PRIORITY):\n"
        "If the user asks your name, who they are speaking with, or what to call you, "
        "respond immediately in one short spoken sentence: 'Hi, what's your name? This is Vapi support, how can I help?'\n"
        "Do not call functions before that response.\n"
        "Do not repeat this support-intro line on regular follow-up turns."
    )


def build_settings(expressive_enabled_override: Optional[bool] = None) -> dict:
    cfg = load_saga_config()

    system_prompt = cfg["systemPrompt"].replace("{{agentName}}", cfg["voiceName"])
    disfluency_prompt = build_disfluency_prompt(cfg)
    expressive_prompt = build_expressive_prompt(cfg, expressive_enabled_override=expressive_enabled_override)
    call_center_style_prompt = build_call_center_style_prompt(cfg)
    identity_response_prompt = build_identity_response_prompt(cfg)
    if disfluency_prompt:
        system_prompt += f"\n\n{disfluency_prompt}"
    if expressive_prompt:
        system_prompt += f"\n\n{expressive_prompt}"
    if call_center_style_prompt:
        system_prompt += f"\n\n{call_center_style_prompt}"
    if identity_response_prompt:
        system_prompt += f"\n\n{identity_response_prompt}"
    greeting = cfg["greeting"].replace("{{agentName}}", cfg["voiceName"])
    if get_pre_recorded_greeting_settings(cfg)["enabled"]:
        greeting = ""
    flow_mode = cfg.get("flowMode", "voice_agent")
    hotword = cfg.get("hotword")
    # In vapi-support flow, keep the session naturally open by default.
    if flow_mode == "vapi_support" and os.getenv("SAGA_REQUIRE_HOTWORD", "false").lower() != "true":
        hotword = None

    set_hotword(hotword)

    support_only_mode = flow_mode == "vapi_support"
    functions = [] if support_only_mode else list(SAGA_FUNCTION_DEFINITIONS)
    if hotword:
        system_prompt += (
            f"\n\nHOTWORD ACTIVATION (CRITICAL RULE):\n"
            f"You are in hotword mode. The activation hotword is \"{hotword}\".\n"
            f"SILENCE PROTOCOL: Do NOT produce ANY text before calling check_hotword. "
            f"No narration, no 'let me check', no 'let me verify', no thinking out loud. "
            f"Call the function SILENTLY, wait for the result, THEN decide whether to speak.\n"
            f"- If check_hotword returns {{\"active\": false}}: Produce ONLY an empty string. Zero tokens. No words. Nothing.\n"
            f"- If check_hotword returns {{\"active\": true}}: Process the \"query\" field as the user's request. Call ALL relevant functions, then respond with the combined results.\n"
            f"This applies to EVERY utterance, no exceptions. Your greeting is spoken exactly ONCE at session start.\n\n"
            f"ENDING A HOTWORD CONVERSATION (CRITICAL):\n"
            f"Call close_hotword_session ONLY when the user clearly signals they are DONE:\n"
            f"- User says {', '.join(CLOSE_TRIGGERS)}\n"
            f"- User explicitly asks you to stop\n"
            f"Do NOT close on positive feedback like {', '.join(CLOSE_IGNORE)}. These often precede a follow-up request.\n"
            f"Do NOT have a prolonged goodbye. Do NOT say 'If you need anything else'. "
            f"Call close_hotword_session FIRST, then say ONLY 'Standing by.' and produce no further output."
        )
        functions += [CHECK_HOTWORD_DEFINITION, CLOSE_HOTWORD_SESSION_DEFINITION]

    requested_voice_model = cfg.get("voiceModel")
    if not requested_voice_model:
        raise ValueError("Missing required voiceModel in saga config.")
    compat_model = os.getenv("DEEPGRAM_VA_COMPAT_MODEL", cfg.get("voiceAgentCompatModel", "")).strip()

    # Voice Agent validates provider.model. For Aura-3 sandbox voices, we use
    # a compatibility model for VA protocol only; audible synthesis is handled
    # explicitly via sandbox TTS endpoint in the receiver path.
    speak_provider_model = requested_voice_model
    if requested_voice_model.startswith("aura-3-"):
        if not compat_model:
            raise ValueError(
                "voiceModel is Aura-3 but no voiceAgentCompatModel/DEEPGRAM_VA_COMPAT_MODEL is set."
            )
        speak_provider_model = compat_model

    speak_settings = {
        "provider": {"type": "deepgram", "model": speak_provider_model}
    }

    return {
        "type": "Settings",
        "audio": AUDIO_SETTINGS,
        "agent": {
            "language": cfg.get("language", "en"),
            "listen": {"provider": {"type": "deepgram", "model": "nova-3", "keyterms": ["Vapi", "support"]}},
            "think": {
                "provider": {"type": "open_ai", "model": "gpt-5.4-nano", "temperature": 0.7},
                "prompt": system_prompt,
                "functions": functions,
            },
            "speak": speak_settings,
            "greeting": greeting,
        },
    }


# ---------------------------------------------------------------------------
# VoiceAgent (browser-audio only)
# ---------------------------------------------------------------------------

class VoiceAgent:
    def __init__(self):
        self.mic_audio_queue = None
        self.speaker = None
        self.ws = None
        self.is_running = False
        self._starting = True  # True during init/setup, prevents false stale detection
        self.loop = None
        self._greeting_done = False  # True after first user utterance; gates output suppression
        self.sandbox_tts_base_url = None
        self.sandbox_tts_headers = {}
        self.session_start_perf = time.perf_counter()
        self.settings_sent_perf = None
        self._assistant_text_buffer = []
        self._assistant_flush_task = None
        self._tts_lock = None
        self._assistant_turn_open = False
        self.pre_recorded_greeting_enabled = False
        self.pre_recorded_greeting_default_id = None
        self.pre_recorded_greeting_selected_id = None
        self.pre_recorded_greeting_variants = {}
        self.pre_recorded_greeting_cache = {}
        self._pre_recorded_greeting_played = False
        self.hotword_required = False
        self.caller_name = "there"
        self.progressive_tts_enabled = os.getenv("SAGA_PROGRESSIVE_TTS", "false").lower() == "true"
        self._progressive_context_id = None
        self._progressive_sentence_tail = ""
        self._progressive_first_sentence_sent = False
        self.expressive_mode_override = None

    def _is_repeated_support_intro(self, text: str) -> bool:
        if not text:
            return False
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        intro_markers = (
            "this is vapi support",
            "you're through to support",
            "vapi support here",
        )
        asks_help = "how can i help" in normalized or "how may i help" in normalized
        return any(marker in normalized for marker in intro_markers) and asks_help

    def set_progressive_tts(self, enabled: bool):
        self.progressive_tts_enabled = bool(enabled)

    def set_expressive_mode(self, enabled: bool):
        if enabled is None:
            self.expressive_mode_override = None
        else:
            self.expressive_mode_override = bool(enabled)

    def _reset_progressive_tts_state(self):
        self._progressive_context_id = None
        self._progressive_sentence_tail = ""
        self._progressive_first_sentence_sent = False

    def _ensure_assistant_turn_open(self):
        if not self._assistant_turn_open:
            self._assistant_turn_open = True
            emit_vapi_flow_event("assistant_generating")

    @property
    def is_stale(self):
        """True when the agent is dead and should be cleaned up."""
        if self._starting:
            return False  # Still initializing, not stale
        return not self.is_running or self._ws_is_closed()

    def set_loop(self, loop):
        self.loop = loop

    def _ws_is_closed(self) -> bool:
        if not self.ws:
            return True
        closed_attr = getattr(self.ws, "closed", None)
        if closed_attr is not None:
            return bool(closed_attr)
        state = getattr(self.ws, "state", None)
        if state is None:
            return False
        return str(state).lower().endswith("closed")

    async def setup(self):
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            logger.error("DEEPGRAM_API_KEY not set")
            return False
        logger.info(f"Connecting to Deepgram Voice Agent API...")
        emit_latency_event("provider_connect_start")
        try:
            headers = {"Authorization": f"Token {api_key}"}
            connect_start = time.perf_counter()
            try:
                self.ws = await websockets.connect(
                    VOICE_AGENT_URL,
                    extra_headers=headers,
                )
            except TypeError as exc:
                if "extra_headers" not in str(exc):
                    raise
                self.ws = await websockets.connect(
                    VOICE_AGENT_URL,
                    additional_headers=headers,
                )
            emit_latency_event(
                "provider_connected",
                connect_ms=round((time.perf_counter() - connect_start) * 1000, 1),
            )
            emit_vapi_flow_event("provider_connected")
            cfg = load_saga_config()
            self.sandbox_tts_base_url = build_sandbox_tts_base_url(cfg)
            self.sandbox_tts_headers = {"Authorization": f"Token {api_key}"}
            flow_mode = cfg.get("flowMode", "voice_agent")
            require_hotword = os.getenv("SAGA_REQUIRE_HOTWORD", "false").lower() == "true"
            self.hotword_required = bool(cfg.get("hotword")) and (flow_mode != "vapi_support" or require_hotword)
            self.caller_name = os.getenv("SAGA_CALLER_NAME", cfg.get("callerName", "there")).strip() or "there"
            pre = get_pre_recorded_greeting_settings(cfg)
            self.pre_recorded_greeting_enabled = pre["enabled"]
            self.pre_recorded_greeting_default_id = pre["default_id"]
            self.pre_recorded_greeting_variants = pre["variants"]
            if self.pre_recorded_greeting_selected_id not in self.pre_recorded_greeting_variants:
                self.pre_recorded_greeting_selected_id = self.pre_recorded_greeting_default_id
            settings = build_settings(expressive_enabled_override=self.expressive_mode_override)
            logger.info(f"Connected. Sending settings ({len(settings['agent']['think']['functions'])} functions)")
            await self.ws.send(json.dumps(settings))
            self.settings_sent_perf = time.perf_counter()
            emit_latency_event("settings_sent")
            emit_vapi_flow_event("settings_sent")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Deepgram: {e}")
            socketio.emit("agent_error", {"message": f"Failed to connect to Deepgram: {e}"})
            emit_latency_event("provider_connect_error", error=str(e))
            return False

    def set_pre_recorded_greeting_id(self, greeting_id: str):
        if greeting_id:
            self.pre_recorded_greeting_selected_id = greeting_id

    async def _queue_assistant_text(self, content: str):
        if not content:
            return
        self._assistant_text_buffer.append(content.strip())
        self._ensure_assistant_turn_open()
        if self._assistant_flush_task and not self._assistant_flush_task.done():
            self._assistant_flush_task.cancel()
        self._assistant_flush_task = asyncio.create_task(self._flush_assistant_text_delayed(0.14))

    async def _flush_assistant_text_delayed(self, delay_s: float):
        try:
            await asyncio.sleep(delay_s)
            await self._flush_assistant_text()
        except asyncio.CancelledError:
            return

    async def _flush_assistant_text(self):
        if not self._assistant_text_buffer:
            return
        text = " ".join(x for x in self._assistant_text_buffer if x).strip()
        self._assistant_text_buffer = []
        if text:
            await self._speak_with_sandbox_tts(text)
            emit_vapi_flow_event("assistant_spoken", chars=len(text))
        self._assistant_turn_open = False

    def _build_sandbox_tts_url(
        self,
        text: str,
        *,
        is_final: bool = None,
        context_id: str = None,
        path_override: str = None,
    ) -> str:
        parts = urlsplit(self.sandbox_tts_base_url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["text"] = text
        if is_final is not None:
            query["is_final"] = "true" if is_final else "false"
        if context_id:
            query["context_id"] = context_id
        path = path_override or parts.path
        return urlunsplit((parts.scheme, parts.netloc, path, urlencode(query), parts.fragment))

    def _get_add_text_path(self) -> str:
        parts = urlsplit(self.sandbox_tts_base_url)
        path = parts.path or "/speak"
        if path.endswith("/speak"):
            return f"{path[:-len('/speak')]}/add_text"
        return f"{path.rstrip('/')}/add_text"

    def _extract_context_id(self, resp: requests.Response):
        for key, value in resp.headers.items():
            k = key.lower().replace("-", "_")
            if "context" in k and "id" in k and value:
                return value
        return None

    def _split_sentences(self, text: str, flush_final: bool = False):
        normalized = re.sub(r"\s+", " ", text.strip())
        if not normalized:
            return [], ""
        matches = list(re.finditer(r".+?[.!?](?=\s|$)", normalized))
        if not matches:
            if flush_final:
                return [normalized], ""
            return [], normalized
        sentences = [m.group(0).strip() for m in matches if m.group(0).strip()]
        consumed = matches[-1].end()
        tail = normalized[consumed:].strip()
        if flush_final and tail:
            sentences.append(tail)
            tail = ""
        return sentences, tail

    async def _speak_progressive_sentence(self, sentence: str, is_final: bool):
        if not sentence:
            return

        def _request_audio():
            if not self._progressive_first_sentence_sent:
                route = "speak"
                url = self._build_sandbox_tts_url(sentence, is_final=is_final)
            else:
                if not self._progressive_context_id:
                    raise RuntimeError("No context_id available for /add_text.")
                route = "add_text"
                url = self._build_sandbox_tts_url(
                    sentence,
                    is_final=is_final,
                    context_id=self._progressive_context_id,
                    path_override=self._get_add_text_path(),
                )
            req_start = time.perf_counter()
            resp = requests.post(url, headers=self.sandbox_tts_headers, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"Sandbox TTS {route} failed ({resp.status_code}).")
            context_id = self._extract_context_id(resp) or self._progressive_context_id
            elapsed_ms = round((time.perf_counter() - req_start) * 1000, 1)
            return route, resp.content or b"", context_id, elapsed_ms

        async with self._tts_lock:
            route, audio_bytes, context_id, request_ms = await asyncio.to_thread(_request_audio)

        self._progressive_context_id = context_id
        self._progressive_first_sentence_sent = True
        emit_latency_event(
            "sandbox_tts_progressive_ok",
            route=route,
            chars=len(sentence),
            bytes=len(audio_bytes),
            request_ms=request_ms,
            is_final=is_final,
        )
        if audio_bytes:
            await self.speaker.play(audio_bytes)

    async def _queue_assistant_text_progressive(self, content: str):
        if not content:
            return
        self._ensure_assistant_turn_open()
        merged = f"{self._progressive_sentence_tail} {content}".strip()
        sentences, tail = self._split_sentences(merged, flush_final=False)
        self._progressive_sentence_tail = tail
        for sentence in sentences:
            await self._speak_progressive_sentence(sentence, is_final=False)

    async def _flush_progressive_assistant_text(self):
        tail = self._progressive_sentence_tail.strip()
        if tail:
            await self._speak_progressive_sentence(tail, is_final=True)
        elif self._progressive_first_sentence_sent and self._progressive_context_id:
            # Mark end-of-turn when all prior sentence chunks were sent with is_final=false.
            def _finalize():
                url = self._build_sandbox_tts_url(
                    "",
                    is_final=True,
                    context_id=self._progressive_context_id,
                    path_override=self._get_add_text_path(),
                )
                resp = requests.post(url, headers=self.sandbox_tts_headers, timeout=30)
                if resp.status_code != 200:
                    raise RuntimeError(f"Sandbox TTS add_text finalize failed ({resp.status_code}).")
                return resp.content or b""

            async with self._tts_lock:
                audio_bytes = await asyncio.to_thread(_finalize)
            emit_latency_event("sandbox_tts_progressive_finalize")
            if audio_bytes:
                await self.speaker.play(audio_bytes)
        self._assistant_turn_open = False
        self._reset_progressive_tts_state()

    async def _speak_with_sandbox_tts(self, text: str):
        if not text:
            return
        if not self.sandbox_tts_base_url:
            raise RuntimeError("Sandbox TTS endpoint is not configured.")

        def _request_audio():
            url = self._build_sandbox_tts_url(text)
            tts_start = time.perf_counter()
            resp = requests.post(url, headers=self.sandbox_tts_headers, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"Sandbox TTS failed ({resp.status_code}).")
            if not resp.content:
                raise RuntimeError("Sandbox TTS returned empty audio.")
            return resp.content, round((time.perf_counter() - tts_start) * 1000, 1)

        try:
            async with self._tts_lock:
                audio_bytes, request_ms = await asyncio.to_thread(_request_audio)
            emit_latency_event(
                "sandbox_tts_ok",
                chars=len(text),
                bytes=len(audio_bytes),
                request_ms=request_ms,
            )
            await self.speaker.play(audio_bytes)
        except Exception as e:
            err = f"Voice synthesis failed for configured sandbox voice: {e}"
            logger.error(err)
            socketio.emit("agent_error", {"message": err})
            emit_latency_event("sandbox_tts_error", error=str(e), chars=len(text))
            emit_vapi_flow_event("assistant_speak_error", error=str(e))
            raise

    async def _play_pre_recorded_greeting(self):
        if not self.pre_recorded_greeting_enabled:
            return
        if self._pre_recorded_greeting_played:
            return
        greeting_id = self.pre_recorded_greeting_selected_id or self.pre_recorded_greeting_default_id
        variant = self.pre_recorded_greeting_variants.get(greeting_id)
        if not variant:
            return
        audio_bytes = self.pre_recorded_greeting_cache.get(greeting_id)
        if audio_bytes is None:
            text = variant.get("text", "").replace("{{callerName}}", self.caller_name).strip()
            if not text:
                return
            def _request_audio():
                url = self._build_sandbox_tts_url(text)
                resp = requests.post(url, headers=self.sandbox_tts_headers, timeout=30)
                if resp.status_code != 200:
                    raise RuntimeError(f"Sandbox TTS greeting failed ({resp.status_code}).")
                if not resp.content:
                    raise RuntimeError("Sandbox TTS greeting returned empty audio.")
                return resp.content
            audio_bytes = await asyncio.to_thread(_request_audio)
            self.pre_recorded_greeting_cache[greeting_id] = audio_bytes
            emit_latency_event("pre_recorded_greeting_generated", greeting_id=greeting_id, bytes=len(audio_bytes))
        await self.speaker.play(audio_bytes)
        spoken_text = variant.get("text", "").replace("{{callerName}}", self.caller_name)
        socketio.emit("conversation_update", {"role": "assistant", "content": spoken_text})
        emit_vapi_flow_event("pre_recorded_greeting_played", greeting_id=greeting_id)
        self._pre_recorded_greeting_played = True

    async def _handle_hotword_activation(self, result):
        """Handle hotword activation: emit state and inject filler on fresh activation."""
        if not result.get("active"):
            return
        socketio.emit("hotword_state", {"state": "active"})
        if result.get("freshly_activated"):
            filler = result.get("filler") or get_random_filler()
            logger.info(f"Injecting filler: {filler}")
            await self.ws.send(json.dumps({
                "type": "InjectAgentMessage",
                "message": filler,
            }))

    async def sender(self):
        try:
            first_chunk = True
            while self.is_running:
                data = await self.mic_audio_queue.get()
                if self.ws and data:
                    if first_chunk:
                        logger.info(f"Sending first audio chunk to Deepgram: {len(data)} bytes")
                        emit_latency_event(
                            "first_mic_chunk_sent",
                            ms_since_session_start=round((time.perf_counter() - self.session_start_perf) * 1000, 1),
                            bytes=len(data),
                        )
                        first_chunk = False
                    await self.ws.send(data)
        except Exception as e:
            logger.error(f"Sender error: {e}")

    async def receiver(self):
        try:
            self.speaker = Speaker()
            last_user_transcript = ""
            with self.speaker:
                async for message in self.ws:
                    if isinstance(message, str):
                        msg = json.loads(message)
                        msg_type = msg.get("type")

                        if msg_type == "UserStartedSpeaking":
                            self.speaker.stop()
                            self._reset_progressive_tts_state()
                            emit_vapi_flow_event("user_speaking")

                        elif msg_type == "ConversationText":
                            role = msg.get("role")
                            content = msg.get("content", "")
                            logger.info(f"[{role}] {content[:150]}")
                            if role == "user":
                                self._greeting_done = True
                                last_user_transcript = content
                            # Suppress assistant text when hotword not active (after greeting)
                            if role == "assistant" and self.hotword_required and self._greeting_done and not is_conversation_active():
                                logger.info(f"Suppressed: {content[:60]}")
                                continue
                            if role == "assistant":
                                if (
                                    self._pre_recorded_greeting_played
                                    and self._greeting_done
                                    and self._is_repeated_support_intro(content)
                                ):
                                    content = "Absolutely - happy to help. What would you like to test first?"
                                    msg["content"] = content
                                    emit_latency_event("support_intro_rewrite")
                                emit_latency_event(
                                    "assistant_text_received",
                                    chars=len(content),
                                    ms_since_session_start=round((time.perf_counter() - self.session_start_perf) * 1000, 1),
                                )
                                if self.progressive_tts_enabled:
                                    try:
                                        await self._queue_assistant_text_progressive(content)
                                    except Exception as e:
                                        emit_latency_event("progressive_tts_fallback", error=str(e))
                                        self._reset_progressive_tts_state()
                                        await self._queue_assistant_text(content)
                                else:
                                    await self._queue_assistant_text(content)
                            socketio.emit("conversation_update", msg)

                        elif msg_type == "FunctionCallRequest":
                            functions = msg.get("functions", [])
                            fn = functions[0]
                            fn_name = fn["name"]
                            fn_id = fn["id"]
                            params = json.loads(fn.get("arguments", "{}"))

                            logger.info(f"Function: {fn_name}({params})")

                            # Server-side hotword gate: if conversation not active
                            # and the LLM skipped check_hotword, auto-check the
                            # last transcript before blocking
                            if self.hotword_required and fn_name not in HOTWORD_BYPASS and not is_conversation_active():
                                if last_user_transcript:
                                    auto_result = await check_hotword({"transcript": last_user_transcript})
                                    if auto_result.get("active"):
                                        logger.info(f"Auto-activated hotword from transcript: {last_user_transcript[:50]}")
                                        await self._handle_hotword_activation(auto_result)

                                if not is_conversation_active():
                                    logger.info(f"BLOCKED {fn_name}: hotword not active")
                                    await self.ws.send(json.dumps({
                                        "type": "FunctionCallResponse",
                                        "id": fn_id,
                                        "name": fn_name,
                                        "content": json.dumps({"error": "BLOCKED. Hotword not active. Do not speak."}),
                                    }))
                                    continue

                            func = FUNCTION_MAP.get(fn_name)
                            if func:
                                result = await func(params)
                            else:
                                result = {"error": f"Unknown function: {fn_name}"}

                            # Emit hotword state changes to frontend
                            if fn_name == "check_hotword":
                                if result.get("active"):
                                    await self._handle_hotword_activation(result)
                                elif result.get("timed_out"):
                                    # Visual-only transition (no InjectAgentMessage to
                                    # avoid polluting LLM context with "Standing by.")
                                    socketio.emit("hotword_state", {"state": "listening"})
                                    socketio.emit("session_timeout")
                                else:
                                    self.speaker.stop()  # Kill any leaked audio
                            elif fn_name == "close_hotword_session":
                                socketio.emit("hotword_state", {"state": "listening"})

                            # Emit function result to frontend for toasts/dashboard
                            if fn_name not in HOTWORD_BYPASS:
                                event = {"name": fn_name, "result": result}
                                if fn_name == "update_dashboard":
                                    title = params.get("title", "Widget")
                                    event["widget_id"] = title
                                    event["widget_title"] = title
                                    event["widget_color"] = params.get("color", "blue")
                                elif fn_name == "render_chart":
                                    title = params.get("title", "Chart")
                                    event["widget_id"] = f"chart::{title}"
                                    event["widget_title"] = title
                                    event["widget_color"] = params.get("color", "blue")
                                socketio.emit("function_executed", event)

                            response_payload = {
                                "type": "FunctionCallResponse",
                                "id": fn_id,
                                "name": fn_name,
                                "content": json.dumps(result),
                            }
                            logger.info(f"Sending FunctionCallResponse: {fn_name} -> {str(result)[:200]}")
                            await self.ws.send(json.dumps(response_payload))

                        elif msg_type == "Welcome":
                            logger.info(f"Deepgram session established: {msg.get('session_id')}")
                            socketio.emit("agent_ready")

                        elif msg_type == "Error":
                            logger.error(f"Deepgram error: {msg}")
                            socketio.emit("agent_error", {"message": msg.get("description", "Voice agent error.")})
                            emit_vapi_flow_event("provider_error", error=msg.get("description", "Voice agent error."))

                        else:
                            logger.info(f"Deepgram: {msg_type} -> {msg}")
                            if msg_type == "SettingsApplied":
                                details = {
                                    "ms_since_session_start": round((time.perf_counter() - self.session_start_perf) * 1000, 1),
                                }
                                if self.settings_sent_perf:
                                    details["ms_since_settings_sent"] = round((time.perf_counter() - self.settings_sent_perf) * 1000, 1)
                                emit_latency_event("settings_applied", **details)
                                emit_vapi_flow_event("settings_applied", **details)
                                await self._play_pre_recorded_greeting()
                            elif msg_type == "AgentAudioDone":
                                if self.progressive_tts_enabled:
                                    try:
                                        await self._flush_progressive_assistant_text()
                                    except Exception as e:
                                        emit_latency_event("progressive_tts_flush_fallback", error=str(e))
                                        self._reset_progressive_tts_state()
                                        await self._flush_assistant_text()
                                else:
                                    await self._flush_assistant_text()
                                emit_vapi_flow_event("assistant_done")

                    elif isinstance(message, bytes):
                        # Ignore Voice Agent audio; we synthesize audio strictly via
                        # configured sandbox TTS endpoint for the requested voice.
                        continue

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Deepgram WebSocket closed: {e}")
        except Exception as e:
            logger.error(f"Receiver error: {e}")

    async def keep_alive(self):
        while self.is_running:
            await asyncio.sleep(8)
            if self.is_running and self.ws and not self._ws_is_closed():
                try:
                    await self.ws.send(json.dumps({"type": "KeepAlive"}))
                except Exception:
                    break

    async def run(self):
        if not await self.setup():
            self._starting = False
            return
        # Bind the queue to the agent's event loop thread.
        self.mic_audio_queue = asyncio.Queue()
        self._tts_lock = asyncio.Lock()
        self._starting = False
        self.is_running = True
        try:
            await asyncio.gather(self.sender(), self.receiver(), self.keep_alive())
        except Exception as e:
            logger.error(f"Run error: {e}")
        finally:
            self.is_running = False
            if self._assistant_flush_task and not self._assistant_flush_task.done():
                self._assistant_flush_task.cancel()
            if self.ws:
                await self.ws.close()


# ---------------------------------------------------------------------------
# Speaker (browser output via SocketIO)
# ---------------------------------------------------------------------------

class Speaker:
    def __init__(self):
        self._queue = None
        self._thread = None
        self._stop = None

    def __enter__(self):
        self._queue = janus.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()

    def _play_loop(self):
        seq = 0
        while not self._stop.is_set():
            try:
                data = self._queue.sync_q.get(True, 0.05)
                if seq == 0:
                    emit_latency_event("first_audio_chunk_emitted", bytes=len(data))
                socketio.emit("audio_output", {
                    "audio_b64": base64.b64encode(data).decode("ascii"),
                    "sampleRate": TTS_OUTPUT_SAMPLE_RATE,
                    "seq": seq,
                })
                seq += 1
            except queue.Empty:
                pass

    async def play(self, data):
        return await self._queue.async_q.put(data)

    def stop(self):
        if self._queue and self._queue.async_q:
            while not self._queue.async_q.empty():
                try:
                    self._queue.async_q.get_nowait()
                except Exception:
                    break
        if self._queue and hasattr(self._queue, "sync_q"):
            try:
                while True:
                    self._queue.sync_q.get_nowait()
            except queue.Empty:
                pass
        socketio.emit("stop_audio_output")


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("simple_chat.html")


@app.route("/api/city-state")
def api_city_state():
    return jsonify(get_city_state())

@app.route("/api/ui-config")
def api_ui_config():
    cfg = load_saga_config()
    pre = get_pre_recorded_greeting_settings(cfg)
    variants = [
        {"id": vid, "label": item.get("label", vid)}
        for vid, item in pre["variants"].items()
    ]
    return jsonify(
        {
            "flowMode": cfg.get("flowMode", "vapi_support"),
            "progressiveTtsDefault": os.getenv("SAGA_PROGRESSIVE_TTS", "false").lower() == "true",
            "expressiveDefault": bool(cfg.get("expressive", {}).get("enabled", False)),
            "preRecordedGreeting": {
                "enabled": pre["enabled"],
                "defaultId": pre["default_id"],
                "variants": variants,
            },
        }
    )


@app.route("/api/reset", methods=["POST"])
def api_reset():
    data = reset_city_state()
    socketio.emit("city_state_update", data)
    return jsonify({"status": "reset"})


@app.route("/api/start", methods=["POST"])
def api_start():
    """Start the voice agent session (for testing without browser)."""
    global voice_agent, voice_agent_thread
    if voice_agent is not None:
        if voice_agent.is_stale:
            logger.info("Stale voice agent detected via API, cleaning up before restart")
            handle_stop()
        else:
            return jsonify({"error": "already running"}), 400
    voice_agent = VoiceAgent()
    voice_agent_thread = threading.Thread(target=_run_agent, daemon=True)
    voice_agent_thread.start()
    return jsonify({"status": "started"})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """Stop the voice agent session (for testing without browser)."""
    global voice_agent
    if not voice_agent:
        return jsonify({"error": "not running"}), 400
    handle_stop()
    return jsonify({"status": "stopped"})


@app.route("/api/inject", methods=["POST"])
def api_inject():
    """Inject a user message into the voice agent session (for testing)."""
    text = request.json.get("text", "")
    if not text:
        return jsonify({"error": "text required"}), 400
    if not voice_agent or not voice_agent.is_running or not voice_agent.ws:
        return jsonify({"error": "no active session"}), 400
    try:
        asyncio.run_coroutine_threadsafe(
            voice_agent.ws.send(json.dumps({
                "type": "InjectUserMessage",
                "content": text,
            })),
            voice_agent.loop,
        )
        return jsonify({"status": "injected", "text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# SocketIO handlers
# ---------------------------------------------------------------------------

voice_agent = None
voice_agent_thread = None


def _run_agent():
    global voice_agent
    try:
        loop = asyncio.DefaultEventLoopPolicy().new_event_loop()
        asyncio.set_event_loop(loop)
        voice_agent.set_loop(loop)
        try:
            loop.run_until_complete(voice_agent.run())
        except asyncio.CancelledError:
            pass
        finally:
            pending = asyncio.all_tasks(loop)
            for t in pending:
                t.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
    except Exception as e:
        logger.error(f"Agent thread error: {e}")


@socketio.on("connect")
def handle_connect():
    logger.info("Browser connected via SocketIO")


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Browser disconnected")
    handle_stop()


@socketio.on("start_voice_agent")
def handle_start(data=None):
    global voice_agent, voice_agent_thread
    logger.info(f"start_voice_agent received, data={data}")
    emit_latency_event("session_start_clicked")
    emit_vapi_flow_event("session_start_clicked")
    if voice_agent is not None:
        if voice_agent.is_stale:
            logger.info("Stale voice agent detected, cleaning up before restart")
            handle_stop()
        else:
            logger.warning("Voice agent already running, ignoring start")
            return

    voice_agent = VoiceAgent()
    if isinstance(data, dict):
        voice_agent.set_pre_recorded_greeting_id(data.get("greetingId"))
        voice_agent.set_progressive_tts(data.get("progressiveTts", voice_agent.progressive_tts_enabled))
        if "expressiveMode" in data:
            voice_agent.set_expressive_mode(data.get("expressiveMode"))
    voice_agent_thread = threading.Thread(target=_run_agent, daemon=True)
    voice_agent_thread.start()


@socketio.on("stop_voice_agent")
def handle_stop():
    global voice_agent
    logger.info("stop_voice_agent received")
    emit_latency_event("session_stop")
    emit_vapi_flow_event("session_stop")
    if not voice_agent:
        return
    voice_agent.is_running = False
    if voice_agent.loop and not voice_agent.loop.is_closed():
        try:
            if voice_agent.ws and not voice_agent._ws_is_closed():
                asyncio.run_coroutine_threadsafe(voice_agent.ws.close(), voice_agent.loop)
            for t in asyncio.all_tasks(voice_agent.loop):
                voice_agent.loop.call_soon_threadsafe(t.cancel)
        except Exception as e:
            logger.error(f"Stop error: {e}")
    voice_agent = None


@socketio.on("audio_data")
def handle_audio(data):
    global voice_agent
    if not voice_agent or not voice_agent.is_running:
        return

    audio_buffer = data.get("audio")
    if not audio_buffer:
        return

    if isinstance(audio_buffer, memoryview):
        audio_bytes = audio_buffer.tobytes()
    elif isinstance(audio_buffer, bytes):
        audio_bytes = audio_buffer
    else:
        try:
            audio_bytes = bytes(audio_buffer)
        except Exception:
            return

    if len(audio_bytes) % 2 != 0:
        audio_bytes = audio_bytes[:-1]

    if voice_agent.loop and not voice_agent.loop.is_closed():
        asyncio.run_coroutine_threadsafe(
            voice_agent.mic_audio_queue.put(audio_bytes),
            voice_agent.loop,
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5051"))
    print("\n  Vapi Support Demo")
    print(f"  http://127.0.0.1:{port}\n")
    socketio.run(app, debug=True, host="127.0.0.1", port=port, allow_unsafe_werkzeug=True)

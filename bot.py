import os
import io
import json
import time
import base64
import random
import sqlite3
import asyncio
import logging
import threading
import httpx
import openai
from datetime import time as dtime, datetime
from zoneinfo import ZoneInfo
from collections import defaultdict
from openai import AsyncOpenAI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Required environment variables ────────────────────────────────────────────
def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise SystemExit(
            f"Missing required environment variable: {name}. "
            f"Set it (see .env.example) before starting the bot."
        )
    return val


# ── OpenRouter client ─────────────────────────────────────────────────────────
OPENROUTER_API_KEY = _require_env("OPENROUTER_API_KEY")

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

OR_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "X-Title": "Telegram Claude Bot",
}

# ── Available chat models (добавляй сюда любые модели с OpenRouter) ───────────
AVAILABLE_MODELS = [
    {"id": "anthropic/claude-sonnet-4-6",           "label": "🟠 Claude Sonnet 4.6"},
    {"id": "z-ai/glm-5.2",                          "label": "🔵 GLM 5.2"},
    {"id": "google/gemini-2.5-flash",                "label": "🟢 Gemini 2.5 Flash"},
    {"id": "openai/gpt-4o",                          "label": "⚫ GPT-4o"},
    {"id": "deepseek/deepseek-v4-flash",             "label": "🔷 DeepSeek V4 Flash"},
    {"id": "deepseek/deepseek-r1-0528",             "label": "🧠 DeepSeek R1"},
    {"id": "meta-llama/llama-3.3-70b-instruct",     "label": "🟣 Llama 3.3 70B (free)"},
]

# Use the GA "Nano Banana" model. The older "...-image-preview" slug advertises
# NO supported parameters on OpenRouter's image endpoint, so sending aspect_ratio
# (or n) to it gets rejected — which previously surfaced as a misleading
# "check your balance" message. The GA model accepts aspect_ratio.
IMAGE_MODEL = os.environ.get("OPENROUTER_IMAGE_MODEL", "google/gemini-2.5-flash-image")

# Aspect ratios accepted by gemini-2.5-flash-image. Anything else falls back to 1:1.
ALLOWED_RATIOS = {"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"}

# Default chat model can be overridden via OPENROUTER_MODEL.
DEFAULT_CHAT_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-6")

# Multimodal model used for image understanding and voice transcription
# (must accept image + audio input). Gemini Flash handles both cheaply.
AUX_MODEL = os.environ.get("OPENROUTER_AUX_MODEL", "google/gemini-2.5-flash")

# Text-to-speech (audio-output) model + voice for /say.
TTS_MODEL = os.environ.get("OPENROUTER_TTS_MODEL", "openai/gpt-audio-mini")
TTS_VOICE = os.environ.get("OPENROUTER_TTS_VOICE", "ash")

# Telegram hard limit for a single text message.
TG_MAX_LEN = 4096

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant inside a Telegram bot. "
    "Format replies for plain text (no Markdown unless the user asks). "
    "Keep answers focused and practical."
)

# Used when an image is sent without an explicit request: the vision model looks
# at the picture and returns a short edit instruction to make it absurd/funny.
CURSED_PROMPT = (
    "Посмотри на это изображение. Придумай, как превратить его в максимально "
    "всратую, нелепую и абсурдно смешную версию с щепоткой чёрного юмора: "
    "гротеск, кринж, утрированные детали, неожиданные абсурдные элементы. "
    "Это безобидный шуточный юмор, без жести и оскорблений конкретных людей. "
    "Верни ТОЛЬКО короткую инструкцию (1–2 предложения) для модели редактирования "
    "изображений — что конкретно изменить/добавить. Без вступлений и пояснений."
)

# Random captions for cursed-mode results (instead of always the same one).
CURSED_CAPTIONS = [
    "😈 держи",
    "ну как тебе такое",
    "я старался 💀",
    "это шедевр, не спорь",
    "осторожно, кринж",
    "произведение искусства 🗿",
    "галерея заплачет",
    "лучше оригинала, очевидно",
    "не показывай это детям",
    "вот это я понимаю арт ✨",
    "получилось всрато, как ты любишь",
    "не благодари",
]

# Personality for spontaneous chat interjections (set from the character poll:
# sarcastic provocateur, dark humour + mild profanity OK, medium length).
PERSONALITY_PROMPT = (
    "Ты — участник группового чата с характером саркастичного остряка и провокатора. "
    "У тебя есть чёткое мнение, ты можешь спорить и подкалывать, не боишься непопулярных позиций. "
    "Чёрный юмор и сарказм приветствуются; мат допустим в меру, БЕЗ жести, без оскорблений конкретных "
    "людей, без дискриминации и без травли. Пиши живым разговорным языком, 2–4 предложения. "
    "Тебе показывают последние сообщения чата — вбрось своё мнение по обсуждаемой теме: можно поддеть, "
    "поспорить или добавить неожиданный угол. Не представляйся и не здоровайся, пиши как обычный участник, "
    "реагируй на тему в целом, а не на конкретного человека по имени."
)

# Daily morning greeting, generated fresh each time for variety.
GREETING_PROMPT = (
    "Напиши короткое (2–4 предложения) утреннее пожелание хорошего дня для "
    "группового чата. Стиль — всратый, абсурдный, с щепоткой чёрного юмора, "
    "дружелюбно-токсичный, но без оскорблений конкретных людей. Обращайся ко всем "
    "как 'кожаные мешки'. Обязательно вставь пожелание, чтобы те, кто сегодня идёт "
    "на работу, лишний раз улыбнулись. Каждый раз придумывай заново и по-новому. "
    "Верни только текст пожелания, без кавычек и пояснений."
)

# Timezone for the daily greeting.
MOSCOW_TZ = ZoneInfo("Europe/Moscow")

# ── Per-user state: model + conversation history ──────────────────────────────
# History is keyed per (chat_id, user_id) so users in a group get separate threads.
conversations: dict[tuple[int, int], list[dict]] = defaultdict(list)
user_model:    dict[int, str]        = {}   # chat_id → model id
image_model:   dict[int, str]        = {}   # chat_id → image model id
cursed_mode:   dict[int, bool]       = {}   # chat_id → auto "cursed" remix (default ON)
last_active:   dict[int, float]      = {}   # chat_id → epoch seconds of last activity
morning_mode:  dict[int, bool]       = {}   # chat_id → daily greeting on/off (default ON)
web_mode:      dict[int, bool]       = {}   # chat_id → web search on/off (default OFF)
memories:      dict[int, list[str]]  = defaultdict(list)  # user_id → remembered facts
automem_mode:  dict[int, bool]       = {}   # user_id → auto-memory on/off (default ON)
spontan_mode:  dict[int, bool]       = {}   # chat_id → spontaneous chime-in on/off (default ON)
last_gen:      dict[int, dict]       = {}   # chat_id → last image request (for "more")
_img_cooldown: dict[int, float]      = {}   # chat_id → last image-op timestamp
_grp_buffer:   dict[int, list[tuple[float, str]]] = defaultdict(list)  # recent group msgs
_spont_last:   dict[int, float]      = {}   # chat_id → last spontaneous interjection ts
chime_context: dict[int, list[str]]  = defaultdict(list)  # chat_id → bot's recent chime-ins
recent_msgs:   dict[int, list[str]]  = defaultdict(list)  # chat_id → recent "name: text" log

MAX_HISTORY    = 300
MAX_TOKENS     = 1024
IMAGE_COOLDOWN = 20      # seconds between image operations per chat
MAX_MEMORIES   = 30      # max remembered facts per user
DOC_CHUNK      = 12000   # chars per chunk when summarizing long documents
DOC_MAX_CHUNKS = 12      # cap chunks to bound cost on huge documents
MAX_CHIME_NOTES = 20     # how many recent chime-ins to remember per chat
RECENT_LOG_MAX  = 60     # recent chat messages kept for /scene
SCENE_DEFAULT   = 15     # default number of messages for /scene
SCENE_MAX       = 50     # max messages /scene will use

# Spontaneous "chime-in": if a group sees >= SPONT_THRESHOLD messages within
# SPONT_WINDOW seconds without the bot, it may drop an opinion — at most once
# per SPONT_COOLDOWN seconds.
SPONT_WINDOW    = 180
SPONT_THRESHOLD = 10
SPONT_COOLDOWN  = 300
SPONT_BUFFER    = 30    # max recent messages kept per chat for context

# Image models offered via /imgmodel. "res" = endpoint accepts a resolution tier.
IMAGE_MODELS = [
    {"id": "google/gemini-2.5-flash-image", "label": "🍌 Gemini Flash (Nano Banana)", "res": False},
    {"id": "bytedance-seed/seedream-4.5",   "label": "🌊 Seedream 4.5",                "res": True},
    {"id": "black-forest-labs/flux.2-pro",  "label": "⚡ FLUX.2 Pro",                  "res": False},
]
ALLOWED_RESOLUTIONS = {"512", "1K", "2K", "4K"}

# ── Persistence (SQLite) ──────────────────────────────────────────────────────
# NOTE: on Railway the filesystem is ephemeral. Attach a Volume and point
# DB_PATH at it (e.g. /data/bot.db) so the database survives redeploys.
DB_PATH = os.environ.get("DB_PATH", "bot.db")
_db = sqlite3.connect(DB_PATH, check_same_thread=False)
_db_lock = threading.Lock()


def _add_column(table: str, column: str, decl: str) -> None:
    """Add a column to an existing table if it isn't there yet (simple migration)."""
    cols = [r[1] for r in _db.execute(f"PRAGMA table_info({table})")]
    if column not in cols:
        _db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def db_init() -> None:
    """Create tables and load persisted state into the in-memory caches."""
    with _db_lock:
        _db.executescript(
            """
            CREATE TABLE IF NOT EXISTS chat_state (
                chat_id     INTEGER PRIMARY KEY,
                model       TEXT,
                cursed      INTEGER,
                last_active REAL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                role    TEXT,
                content TEXT,
                ts      REAL
            );
            CREATE TABLE IF NOT EXISTS usage_log (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                kind    TEXT,
                cost    REAL,
                ts      REAL
            );
            CREATE TABLE IF NOT EXISTS memories (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                fact    TEXT,
                ts      REAL
            );
            CREATE TABLE IF NOT EXISTS user_prefs (
                user_id INTEGER PRIMARY KEY,
                automem INTEGER
            );
            CREATE TABLE IF NOT EXISTS group_notes (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER,
                text    TEXT,
                ts      REAL
            );
            CREATE INDEX IF NOT EXISTS idx_notes_chat ON group_notes(chat_id, id);
            CREATE INDEX IF NOT EXISTS idx_usage_chat ON usage_log(chat_id, ts);
            """
        )
        # Migrations for columns added over time.
        _add_column("chat_state", "morning", "INTEGER")
        _add_column("chat_state", "web", "INTEGER")
        _add_column("chat_state", "img_model", "TEXT")
        _add_column("chat_state", "spontan", "INTEGER")
        _add_column("messages", "user_id", "INTEGER")
        _add_column("memories", "user_id", "INTEGER")
        _db.execute("CREATE INDEX IF NOT EXISTS idx_messages_key ON messages(chat_id, user_id, id)")
        _db.execute("CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id, id)")
        _db.commit()

        for chat_id, model, cursed, la, morning, web, imgm, spontan in _db.execute(
            "SELECT chat_id, model, cursed, last_active, morning, web, img_model, spontan "
            "FROM chat_state"
        ):
            if model:
                user_model[chat_id] = model
            if cursed is not None:
                cursed_mode[chat_id] = bool(cursed)
            if la is not None:
                last_active[chat_id] = la
            if morning is not None:
                morning_mode[chat_id] = bool(morning)
            if web is not None:
                web_mode[chat_id] = bool(web)
            if imgm:
                image_model[chat_id] = imgm
            if spontan is not None:
                spontan_mode[chat_id] = bool(spontan)
        # Old rows have NULL user_id; private chats have chat_id == user_id, so
        # coalescing NULL→chat_id keeps legacy history reachable.
        for chat_id, user_id, role, content in _db.execute(
            "SELECT chat_id, COALESCE(user_id, chat_id), role, content FROM messages ORDER BY id"
        ):
            conversations[(chat_id, user_id)].append({"role": role, "content": content})
        for user_id, fact in _db.execute(
            "SELECT COALESCE(user_id, chat_id), fact FROM memories ORDER BY id"
        ):
            memories[user_id].append(fact)
        for user_id, automem in _db.execute("SELECT user_id, automem FROM user_prefs"):
            if automem is not None:
                automem_mode[user_id] = bool(automem)
        for chat_id, text in _db.execute(
            "SELECT chat_id, text FROM group_notes ORDER BY id"
        ):
            chime_context[chat_id].append(text)
    stored_msgs = sum(len(v) for v in conversations.values())
    logger.info("DB loaded: %d chat settings, %d threads, %d messages",
                len(set(user_model) | set(cursed_mode) | set(last_active)),
                len(conversations), stored_msgs)


def save_chat_state(chat_id: int) -> None:
    with _db_lock:
        _db.execute(
            "INSERT INTO chat_state"
            "(chat_id, model, cursed, last_active, morning, web, img_model, spontan) "
            "VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(chat_id) DO UPDATE SET "
            "model=excluded.model, cursed=excluded.cursed, last_active=excluded.last_active, "
            "morning=excluded.morning, web=excluded.web, img_model=excluded.img_model, "
            "spontan=excluded.spontan",
            (chat_id, user_model.get(chat_id),
             1 if get_cursed(chat_id) else 0, last_active.get(chat_id),
             1 if get_morning(chat_id) else 0, 1 if get_web(chat_id) else 0,
             image_model.get(chat_id), 1 if get_chime(chat_id) else 0),
        )
        _db.commit()


def save_user_pref(user_id: int) -> None:
    with _db_lock:
        _db.execute(
            "INSERT INTO user_prefs(user_id, automem) VALUES(?,?) "
            "ON CONFLICT(user_id) DO UPDATE SET automem=excluded.automem",
            (user_id, 1 if get_automem(user_id) else 0),
        )
        _db.commit()


def log_cost(chat_id: int, kind: str, cost: float) -> None:
    if not cost:
        return
    with _db_lock:
        _db.execute(
            "INSERT INTO usage_log(chat_id, kind, cost, ts) VALUES(?,?,?,?)",
            (chat_id, kind, float(cost), time.time()),
        )
        _db.commit()


def get_spend(chat_id: int) -> tuple[float, float, float]:
    """Return (today_this_chat, total_this_chat, total_all_chats) in USD."""
    midnight = datetime.now(MOSCOW_TZ).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    with _db_lock:
        today = _db.execute(
            "SELECT COALESCE(SUM(cost),0) FROM usage_log WHERE chat_id=? AND ts>=?",
            (chat_id, midnight)).fetchone()[0]
        total = _db.execute(
            "SELECT COALESCE(SUM(cost),0) FROM usage_log WHERE chat_id=?", (chat_id,)).fetchone()[0]
        grand = _db.execute("SELECT COALESCE(SUM(cost),0) FROM usage_log").fetchone()[0]
    return today, total, grand


def add_memory(user_id: int, fact: str) -> None:
    memories[user_id].append(fact)
    with _db_lock:
        _db.execute("INSERT INTO memories(user_id, fact, ts) VALUES(?,?,?)",
                    (user_id, fact, time.time()))
        _db.commit()


def clear_memories(user_id: int) -> None:
    memories[user_id].clear()
    with _db_lock:
        _db.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
        _db.commit()


def add_chime_note(chat_id: int, text: str) -> None:
    """Remember (per chat) what the bot spontaneously said, capped to the last few."""
    chime_context[chat_id].append(text)
    chime_context[chat_id][:] = chime_context[chat_id][-MAX_CHIME_NOTES:]
    with _db_lock:
        _db.execute("INSERT INTO group_notes(chat_id, text, ts) VALUES(?,?,?)",
                    (chat_id, text, time.time()))
        _db.execute(
            "DELETE FROM group_notes WHERE chat_id=? AND id NOT IN "
            "(SELECT id FROM group_notes WHERE chat_id=? ORDER BY id DESC LIMIT ?)",
            (chat_id, chat_id, MAX_CHIME_NOTES),
        )
        _db.commit()


def save_message(chat_id: int, user_id: int, role: str, content: str) -> None:
    with _db_lock:
        _db.execute(
            "INSERT INTO messages(chat_id, user_id, role, content, ts) VALUES(?,?,?,?,?)",
            (chat_id, user_id, role, content, time.time()),
        )
        # Keep only the most recent MAX_HISTORY rows per (chat, user) thread.
        _db.execute(
            "DELETE FROM messages WHERE chat_id=? AND user_id=? AND id NOT IN "
            "(SELECT id FROM messages WHERE chat_id=? AND user_id=? ORDER BY id DESC LIMIT ?)",
            (chat_id, user_id, chat_id, user_id, MAX_HISTORY),
        )
        _db.commit()


def clear_messages(chat_id: int, user_id: int) -> None:
    with _db_lock:
        _db.execute("DELETE FROM messages WHERE chat_id=? AND user_id=?", (chat_id, user_id))
        _db.commit()


def get_model(chat_id: int) -> str:
    return user_model.get(chat_id, DEFAULT_CHAT_MODEL)


def get_cursed(chat_id: int) -> bool:
    return cursed_mode.get(chat_id, True)


def get_morning(chat_id: int) -> bool:
    return morning_mode.get(chat_id, True)


def get_web(chat_id: int) -> bool:
    return web_mode.get(chat_id, False)


def get_automem(user_id: int) -> bool:
    return automem_mode.get(user_id, True)


def get_chime(chat_id: int) -> bool:
    return spontan_mode.get(chat_id, True)


def get_image_model(chat_id: int) -> str:
    return image_model.get(chat_id, IMAGE_MODEL)


def _image_model_supports_res(chat_id: int) -> bool:
    mid = get_image_model(chat_id)
    return any(m["id"] == mid and m["res"] for m in IMAGE_MODELS)


def build_system(user_id: int, chat_id: int | None = None) -> str:
    """System prompt with this user's facts and the chat's recent chime-ins."""
    parts = [SYSTEM_PROMPT]
    facts = memories.get(user_id)
    if facts:
        parts.append("Запомненные факты о пользователе:\n" +
                     "\n".join(f"- {f}" for f in facts))
    notes = chime_context.get(chat_id) if chat_id is not None else None
    if notes:
        parts.append("Ранее в этом чате ты сам высказывал такие мнения:\n" +
                     "\n".join(f"- {n}" for n in notes))
    return "\n\n".join(parts)


def chat_model_id(chat_id: int) -> str:
    """Selected model, with :online suffix when web search is enabled."""
    mid = get_model(chat_id)
    return f"{mid}:online" if get_web(chat_id) else mid


# ── Retry on transient OpenRouter errors (429 / 5xx / timeouts) ───────────────
_RETRY_STATUSES = {429, 500, 502, 503, 504}


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _RETRY_STATUSES
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return getattr(exc, "status_code", None) in _RETRY_STATUSES
    return False


async def with_retry(factory, attempts: int = 3, base: float = 1.5):
    """Call async factory(), retrying transient errors with exponential backoff."""
    for i in range(attempts):
        try:
            return await factory()
        except Exception as e:
            if i == attempts - 1 or not _is_retryable(e):
                raise
            await asyncio.sleep(base * (2 ** i))


def _cost_from_response(response) -> float:
    """Best-effort extraction of OpenRouter cost from a chat completion."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0.0
    cost = getattr(usage, "cost", None)
    if cost is None:
        extra = getattr(usage, "model_extra", None) or {}
        cost = extra.get("cost")
    try:
        return float(cost) if cost else 0.0
    except (TypeError, ValueError):
        return 0.0


# ── Image cooldown (anti-flood) ───────────────────────────────────────────────
def _cooldown_left(chat_id: int) -> float:
    return max(0.0, IMAGE_COOLDOWN - (time.time() - _img_cooldown.get(chat_id, 0)))


def _mark_image(chat_id: int) -> None:
    _img_cooldown[chat_id] = time.time()


def cursed_keyboard(chat_id: int) -> InlineKeyboardMarkup:
    on = get_cursed(chat_id)
    text = "🔴 Выключить всратый режим" if on else "🟢 Включить всратый режим"
    action = "off" if on else "on"
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(text, callback_data=f"cursed:{action}")]]
    )


def more_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("🔄 Ещё вариант", callback_data="regen")]]
    )


def automem_keyboard(user_id: int) -> InlineKeyboardMarkup:
    on = get_automem(user_id)
    text = "🔴 Выключить авто-память" if on else "🟢 Включить авто-память"
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(text, callback_data=f"automem:{'off' if on else 'on'}")]]
    )


def image_model_keyboard(chat_id: int) -> InlineKeyboardMarkup:
    current = get_image_model(chat_id)
    rows = []
    for m in IMAGE_MODELS:
        tick = "✅ " if m["id"] == current else ""
        rows.append([InlineKeyboardButton(f"{tick}{m['label']}", callback_data=f"setimg:{m['id']}")])
    return InlineKeyboardMarkup(rows)


def get_model_label(chat_id: int) -> str:
    mid = get_model(chat_id)
    for m in AVAILABLE_MODELS:
        if m["id"] == mid:
            return m["label"]
    return mid


# ── Model picker keyboard ─────────────────────────────────────────────────────
def model_keyboard(chat_id: int) -> InlineKeyboardMarkup:
    current = get_model(chat_id)
    buttons = []
    for m in AVAILABLE_MODELS:
        tick = "✅ " if m["id"] == current else ""
        buttons.append([InlineKeyboardButton(
            f"{tick}{m['label']}",
            callback_data=f"setmodel:{m['id']}"
        )])
    return InlineKeyboardMarkup(buttons)


# ── Chat helpers ──────────────────────────────────────────────────────────────
def _ids(update: Update) -> tuple[int, int]:
    """Return (chat_id, user_id). Falls back to chat_id when no user (rare)."""
    chat_id = update.effective_chat.id
    user = update.effective_user
    return chat_id, (user.id if user else chat_id)


def trim_history(key: tuple[int, int]) -> None:
    if len(conversations[key]) > MAX_HISTORY:
        conversations[key] = conversations[key][-MAX_HISTORY:]


async def ask_model(chat_id: int, user_id: int, user_text: str) -> str:
    key = (chat_id, user_id)
    conversations[key].append({"role": "user", "content": user_text})
    trim_history(key)
    save_message(chat_id, user_id, "user", user_text)

    messages = [{"role": "system", "content": build_system(user_id, chat_id)}] + conversations[key]

    response = await with_retry(lambda: client.chat.completions.create(
        model=chat_model_id(chat_id),
        max_tokens=MAX_TOKENS,
        messages=messages,
        extra_headers={"X-Title": "Telegram Claude Bot"},
        extra_body={"usage": {"include": True}},
    ))

    reply = response.choices[0].message.content
    if not reply:
        reply = "⚠️ Модель вернула пустой ответ. Попробуй переформулировать."
    conversations[key].append({"role": "assistant", "content": reply})
    save_message(chat_id, user_id, "assistant", reply)
    log_cost(chat_id, "chat", _cost_from_response(response))
    return reply


async def stream_reply(update: Update, chat_id: int, user_id: int, user_text: str) -> None:
    """Like ask_model, but edits the Telegram message as tokens stream in."""
    key = (chat_id, user_id)
    conversations[key].append({"role": "user", "content": user_text})
    trim_history(key)
    save_message(chat_id, user_id, "user", user_text)

    messages = [{"role": "system", "content": build_system(user_id, chat_id)}] + conversations[key]

    stream = await with_retry(lambda: client.chat.completions.create(
        model=chat_model_id(chat_id),
        max_tokens=MAX_TOKENS,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        extra_headers={"X-Title": "Telegram Claude Bot"},
        extra_body={"usage": {"include": True}},
    ))

    placeholder = await update.message.reply_text("✍️ …")
    acc, last_edit, cost = "", 0.0, 0.0
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            acc += chunk.choices[0].delta.content
        if getattr(chunk, "usage", None):
            cost = _cost_from_response(chunk) or cost
        now = time.time()
        if acc and now - last_edit > 1.2:
            try:
                await placeholder.edit_text(acc[-TG_MAX_LEN:])
                last_edit = now
            except Exception:
                pass

    if not acc:
        acc = "⚠️ Модель вернула пустой ответ. Попробуй переформулировать."

    # Final render (split if longer than one Telegram message).
    try:
        await placeholder.edit_text(acc[:TG_MAX_LEN])
    except Exception:
        pass
    for i in range(TG_MAX_LEN, len(acc), TG_MAX_LEN):
        await update.message.reply_text(acc[i:i + TG_MAX_LEN])

    conversations[key].append({"role": "assistant", "content": acc})
    save_message(chat_id, user_id, "assistant", acc)
    log_cost(chat_id, "chat", cost)


# ── Auto-memory: extract durable facts about the user in the background ────────
AUTOMEM_PROMPT = (
    "Из сообщения пользователя выдели устойчивые факты о нём, которые стоит запомнить "
    "надолго (имя, город, профессия, увлечения, явные предпочтения по общению). "
    "Игнорируй сиюминутное и вопросы. Верни каждый факт с новой строки, кратко "
    "и от третьего лица. Если запоминать нечего — верни ровно NONE.\n\nСообщение: "
)


async def auto_extract_memory(chat_id: int, user_id: int, text: str) -> None:
    if not get_automem(user_id) or len(text) < 12:
        return
    if len(memories.get(user_id, [])) >= MAX_MEMORIES:
        return
    try:
        out = await gen_text(AUTOMEM_PROMPT + text, max_tokens=160,
                             temperature=0.2, chat_id=chat_id, kind="automem")
    except Exception:
        logger.exception("Auto-memory extraction failed")
        return
    existing = [m.lower() for m in memories.get(user_id, [])]
    for line in out.splitlines():
        fact = line.strip().lstrip("-•* ").strip()
        if not fact or fact.upper() == "NONE" or len(fact) > 200:
            continue
        if any(fact.lower() in e or e in fact.lower() for e in existing):
            continue
        if len(memories.get(user_id, [])) >= MAX_MEMORIES:
            break
        add_memory(user_id, fact)
        existing.append(fact.lower())


# ── Media helpers ─────────────────────────────────────────────────────────────
def _data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


async def download_tg_file(context: ContextTypes.DEFAULT_TYPE, file_id: str) -> bytes:
    tg_file = await context.bot.get_file(file_id)
    return bytes(await tg_file.download_as_bytearray())


async def ask_multimodal(parts: list[dict], max_tokens: int = 1024,
                         chat_id: int | None = None, kind: str = "aux") -> str:
    """One-off multimodal call (image/audio/file understanding). Not stored in history."""
    response = await with_retry(lambda: client.chat.completions.create(
        model=AUX_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": parts}],
        extra_headers={"X-Title": "Telegram Claude Bot"},
        extra_body={"usage": {"include": True}},
    ))
    if chat_id is not None:
        log_cost(chat_id, kind, _cost_from_response(response))
    return response.choices[0].message.content or ""


async def gen_text(prompt: str, max_tokens: int = 300, temperature: float = 1.1,
                   chat_id: int | None = None, kind: str = "text") -> str:
    """One-off text generation (no history). Higher temperature for variety."""
    response = await with_retry(lambda: client.chat.completions.create(
        model=AUX_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
        extra_headers={"X-Title": "Telegram Claude Bot"},
        extra_body={"usage": {"include": True}},
    ))
    if chat_id is not None:
        log_cost(chat_id, kind, _cost_from_response(response))
    return response.choices[0].message.content or ""


async def gen_chat(system: str, user_content: str, chat_id: int | None = None,
                   kind: str = "gen", max_tokens: int = 300, temperature: float = 0.9) -> str:
    """One-off system+user generation (no history)."""
    response = await with_retry(lambda: client.chat.completions.create(
        model=AUX_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user_content}],
        extra_headers={"X-Title": "Telegram Claude Bot"},
        extra_body={"usage": {"include": True}},
    ))
    if chat_id is not None:
        log_cost(chat_id, kind, _cost_from_response(response))
    return response.choices[0].message.content or ""


async def tts_bytes(chat_id: int, text: str) -> bytes:
    """Synthesize speech via an OpenRouter audio model. Returns OGG/Opus bytes."""
    response = await with_retry(lambda: client.chat.completions.create(
        model=TTS_MODEL,
        messages=[{"role": "user", "content": text}],
        extra_headers={"X-Title": "Telegram Claude Bot"},
        extra_body={
            "modalities": ["text", "audio"],
            "audio": {"voice": TTS_VOICE, "format": "opus"},
            "usage": {"include": True},
        },
    ))
    log_cost(chat_id, "tts", _cost_from_response(response))
    msg = response.choices[0].message
    audio = getattr(msg, "audio", None)
    if audio is None and getattr(msg, "model_extra", None):
        audio = msg.model_extra.get("audio")
    data = audio.get("data") if isinstance(audio, dict) else getattr(audio, "data", None)
    if not data:
        raise ValueError("No audio in TTS response")
    return base64.b64decode(data)


async def gen_opinion(chat_id: int, transcript: str) -> str:
    """Generate a spontaneous in-character opinion about the recent chat."""
    response = await with_retry(lambda: client.chat.completions.create(
        model=AUX_MODEL,
        max_tokens=300,
        temperature=1.0,
        messages=[
            {"role": "system", "content": PERSONALITY_PROMPT},
            {"role": "user", "content":
                "Последние сообщения чата:\n" + transcript +
                "\n\nВбрось короткую реплику по обсуждаемой теме."},
        ],
        extra_headers={"X-Title": "Telegram Claude Bot"},
        extra_body={"usage": {"include": True}},
    ))
    log_cost(chat_id, "chime", _cost_from_response(response))
    return response.choices[0].message.content or ""


def _touch(chat_id: int) -> None:
    """Mark a chat as active right now (used to pick recipients for daily greeting)."""
    last_active[chat_id] = time.time()
    save_chat_state(chat_id)


def _pdf_to_text(raw: bytes) -> str:
    """Extract text from a PDF (sync; run via asyncio.to_thread)."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(raw))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


async def answer_over_text(question: str, text: str, name: str, chat_id: int) -> str:
    """Answer a question over (possibly long) document text via map-reduce."""
    text = text.strip()
    if not text:
        return ""
    if len(text) <= DOC_CHUNK * 2:
        return await ask_multimodal(
            [{"type": "text", "text": f"{question}\n\nСодержимое файла {name}:\n{text}"}],
            max_tokens=1500, chat_id=chat_id, kind="doc")

    chunks = [text[i:i + DOC_CHUNK] for i in range(0, len(text), DOC_CHUNK)]
    truncated = len(chunks) > DOC_MAX_CHUNKS
    chunks = chunks[:DOC_MAX_CHUNKS]
    partials = []
    for idx, chunk in enumerate(chunks):
        part = await ask_multimodal([{"type": "text", "text": (
            f"Это часть {idx + 1} из {len(chunks)} документа «{name}». "
            f"Выпиши кратко всё, что относится к запросу: {question}\n\n{chunk}")}],
            max_tokens=600, chat_id=chat_id, kind="doc")
        partials.append(part)
    answer = await ask_multimodal([{"type": "text", "text": (
        f"На основе выжимок из документа «{name}» ответь на запрос: {question}\n\n"
        + "\n\n".join(partials))}], max_tokens=1500, chat_id=chat_id, kind="doc")
    if truncated:
        answer += f"\n\n⚠️ Документ очень большой — обработаны первые {DOC_MAX_CHUNKS} фрагментов."
    return answer


# ── Image generation ──────────────────────────────────────────────────────────
async def generate_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    references: list[str] | None = None,
    model: str | None = None,
    resolution: str | None = None,
) -> tuple[bytes, float]:
    """Returns (image_bytes, cost_usd)."""
    payload = {
        "model": model or IMAGE_MODEL,
        "prompt": prompt,
    }
    if resolution in ALLOWED_RESOLUTIONS:
        payload["resolution"] = resolution
    if references:
        # Image-to-image: keep the source image's geometry, don't force a ratio.
        payload["input_references"] = [
            {"type": "image_url", "image_url": {"url": url}} for url in references
        ]
    elif aspect_ratio in ALLOWED_RATIOS:
        # Only send aspect_ratio when the value is one the model accepts.
        payload["aspect_ratio"] = aspect_ratio

    async def _post():
        async with httpx.AsyncClient(timeout=120) as http:
            resp = await http.post(
                "https://openrouter.ai/api/v1/images",
                headers=OR_HEADERS,
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    data = await with_retry(_post)
    cost = float((data.get("usage") or {}).get("cost") or 0.0)

    items = data.get("data") or []
    if not items:
        raise ValueError(f"No image returned by OpenRouter: {data}")

    item = items[0]
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"]), cost
    if item.get("url"):
        async with httpx.AsyncClient(timeout=60) as http:
            img_resp = await http.get(item["url"])
            img_resp.raise_for_status()
            return img_resp.content, cost
    raise ValueError(f"Unexpected image response: {list(item.keys())}")


def _openrouter_error_text(resp: httpx.Response) -> str:
    """Extract a human-readable error message from an OpenRouter error response."""
    try:
        body = resp.json()
        err = body.get("error", body)
        if isinstance(err, dict):
            return err.get("message") or str(err)
        return str(err)
    except Exception:
        return resp.text[:300]


# ── Group-chat helpers ────────────────────────────────────────────────────────
def _is_group(update: Update) -> bool:
    return update.effective_chat.type in ("group", "supergroup")


def _addressed_to_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """True if the message @mentions the bot or replies to one of its messages."""
    msg = update.message
    # Reply to one of the bot's own messages
    reply = msg.reply_to_message
    if reply and reply.from_user and reply.from_user.id == context.bot.id:
        return True
    # @username or inline text-mention of the bot
    text = msg.text or msg.caption or ""
    bot_username = (context.bot.username or "").lower()
    entities = list(msg.entities or ()) + list(msg.caption_entities or ())
    for ent in entities:
        if ent.type == "mention":
            if text[ent.offset:ent.offset + ent.length].lower() == f"@{bot_username}":
                return True
        elif ent.type == "text_mention" and ent.user and ent.user.id == context.bot.id:
            return True
    return False


def _strip_bot_mention(text: str, context: ContextTypes.DEFAULT_TYPE) -> str:
    if not text:
        return text
    return text.replace(f"@{context.bot.username}", "").strip()


# ── Handlers ──────────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        f"👋 Привет! Сейчас работаю на {get_model_label(chat_id)}.\n\n"
        "Просто напиши мне — отвечаю вживую и помню беседу.\n"
        "🎙 Голосовое — распознаю и отвечу.\n"
        "🖼 Фото без запроса — сделаю всратую версию; с запросом — отредактирую.\n"
        "📄 PDF/текстовый файл — отвечу по содержимому.\n\n"
        "Команды:\n"
        "/model — выбрать модель\n"
        "/imagine <prompt> [--ratio 16:9] — сгенерировать картинку\n"
        "/cursed — всратый режим 😈\n"
        "/chime — спонтанные вбросы в беседу 🗣\n"
        "/web — веб-поиск 🌐\n"
        "/morning — утренние пожелания 🌅\n"
        "/remember, /memory, /forget — личная память 🧠\n"
        "/stats — расходы 💸\n"
        "/reset — очистить историю\n"
        "/info — модели и статус\n"
        "/help — справка"
    )


async def cmd_model_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        f"Текущая модель: {get_model_label(chat_id)}\n\nВыбери модель:",
        reply_markup=model_keyboard(chat_id),
    )


async def callback_set_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    chat_id = query.message.chat_id
    model_id = query.data.split(":", 1)[1]

    user_model[chat_id] = model_id
    conversations[chat_id].clear()   # сбрасываем историю при смене модели
    save_chat_state(chat_id)
    clear_messages(chat_id)

    # найдём label
    label = model_id
    for m in AVAILABLE_MODELS:
        if m["id"] == model_id:
            label = m["label"]
            break

    # обновляем сообщение с галочкой на выбранной модели
    await query.edit_message_text(
        f"✅ Модель переключена на {label}\nИстория очищена.",
        reply_markup=model_keyboard(chat_id),
    )


async def cmd_imgmodel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    label = next((m["label"] for m in IMAGE_MODELS if m["id"] == get_image_model(chat_id)),
                 get_image_model(chat_id))
    await update.message.reply_text(
        f"Текущая модель картинок: {label}\n\nВыбери модель для /imagine и редактирования:",
        reply_markup=image_model_keyboard(chat_id),
    )


async def callback_set_img(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    image_model[chat_id] = query.data.split(":", 1)[1]
    save_chat_state(chat_id)
    label = next((m["label"] for m in IMAGE_MODELS if m["id"] == image_model[chat_id]),
                 image_model[chat_id])
    await query.edit_message_text(
        f"✅ Модель картинок: {label}", reply_markup=image_model_keyboard(chat_id))


async def cmd_cursed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    state = "включён 😈" if get_cursed(chat_id) else "выключен"
    await update.message.reply_text(
        f"Всратый режим сейчас {state}.\n\n"
        "Когда включён, бот переделывает любое присланное фото (если его прямо ни о чём "
        "не просят) в смешную абсурдную версию. Если попросить что-то конкретное — "
        "отредактирует по запросу в любом случае.",
        reply_markup=cursed_keyboard(chat_id),
    )


async def callback_cursed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    cursed_mode[chat_id] = (query.data.split(":", 1)[1] == "on")
    save_chat_state(chat_id)
    state = "включён 😈" if cursed_mode[chat_id] else "выключен"
    await query.edit_message_text(
        f"Всратый режим {state}.",
        reply_markup=cursed_keyboard(chat_id),
    )


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id, user_id = _ids(update)
    conversations[(chat_id, user_id)].clear()
    clear_messages(chat_id, user_id)
    await update.message.reply_text("🗑️ История очищена. Начинаем заново!")


async def cmd_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id, user_id = _ids(update)
    count = len(conversations.get((chat_id, user_id), []))
    img_label = next((m["label"] for m in IMAGE_MODELS if m["id"] == get_image_model(chat_id)),
                     get_image_model(chat_id))
    await update.message.reply_text(
        f"🤖 Модель чата:   {get_model_label(chat_id)}\n"
        f"🎨 Модель картинок: {img_label}\n"
        f"🌐 Веб-поиск: {'вкл' if get_web(chat_id) else 'выкл'}\n"
        f"😈 Всратый режим: {'вкл' if get_cursed(chat_id) else 'выкл'}\n"
        f"🧠 Память: {len(memories.get(user_id, []))} фактов "
        f"(авто: {'вкл' if get_automem(user_id) else 'выкл'})\n"
        f"📊 История: {count} сообщений (макс. {MAX_HISTORY})."
    )


async def cmd_imagine(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Использование: /imagine <промпт> [--ratio 16:9] [--res 2K]\n\n"
            "Форматы: 1:1 | 16:9 | 9:16 | 4:3 | 3:4\n"
            "Разрешение (если модель поддерживает): 1K | 2K | 4K — выбрать модель: /imgmodel\n\n"
            "Пример:\n"
            "  /imagine dragon flying over mountains\n"
            "  /imagine minimalist crypto logo --ratio 1:1"
        )
        return

    args = list(context.args)
    aspect_ratio = "1:1"
    if "--ratio" in args:
        idx = args.index("--ratio")
        if idx + 1 < len(args):
            aspect_ratio = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
    resolution = None
    if "--res" in args:
        idx = args.index("--res")
        if idx + 1 < len(args):
            resolution = args[idx + 1].upper()
            args = args[:idx] + args[idx + 2:]

    prompt = " ".join(args)
    chat_id = update.effective_chat.id
    model = get_image_model(chat_id)
    res = resolution if (resolution in ALLOWED_RESOLUTIONS and _image_model_supports_res(chat_id)) else None

    rem = _cooldown_left(chat_id)
    if rem > 0:
        await update.message.reply_text(f"⏳ Подожди ещё {int(rem) + 1}с перед следующей картинкой.")
        return
    _mark_image(chat_id)

    await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
    status_msg = await update.message.reply_text(
        f"🎨 Генерирую...\nПромпт: {prompt}\nФормат: {aspect_ratio}"
    )

    try:
        image_bytes, cost = await generate_image(prompt, aspect_ratio, model=model, resolution=res)
        log_cost(chat_id, "image", cost)
        last_gen[chat_id] = {"kind": "imagine", "prompt": prompt, "aspect": aspect_ratio,
                             "src": None, "model": model, "res": res}
        caption = f"🎨 {prompt}"
        if len(caption) > 1024:            # Telegram caption limit
            caption = caption[:1021] + "..."
        await update.message.reply_photo(
            photo=io.BytesIO(image_bytes), caption=caption, reply_markup=more_keyboard())
        await status_msg.delete()
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        detail = _openrouter_error_text(e.response)
        logger.error("Image API error: %s — %s", code, detail)
        if code in (401, 403):
            msg = "⚠️ Проблема с ключом OpenRouter (доступ запрещён). Проверь OPENROUTER_API_KEY."
        elif code == 402:
            msg = "⚠️ Недостаточно средств на балансе OpenRouter."
        elif code == 429:
            msg = "⚠️ Слишком много запросов (rate limit). Подожди немного и попробуй снова."
        elif code == 400:
            msg = f"⚠️ Запрос отклонён моделью: {detail}"
        else:
            msg = f"⚠️ Ошибка генерации (HTTP {code}): {detail}"
        await status_msg.edit_text(msg)
    except Exception as e:
        logger.exception("Image error")
        await status_msg.edit_text(f"⚠️ Что-то пошло не так: {e}")


async def gen_scene_prompt(chat_id: int, transcript: str) -> str:
    """Turn a chat excerpt into a vivid image-generation prompt."""
    instruction = (
        "На основе этого фрагмента переписки придумай одно яркое визуальное описание "
        "сцены/иллюстрации, передающей суть и настроение обсуждения (можно с юмором). "
        "Верни ТОЛЬКО промпт для генератора изображений на английском языке, одной строкой, "
        "без пояснений и кавычек.\n\nПереписка:\n" + transcript)
    return (await gen_text(instruction, max_tokens=200, temperature=0.9,
                           chat_id=chat_id, kind="scene")).strip()


async def cmd_scene(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate an image based on the last N chat messages."""
    chat_id = update.effective_chat.id
    n = SCENE_DEFAULT
    if context.args:
        try:
            n = int(context.args[0])
        except ValueError:
            await update.message.reply_text("Использование: /scene [сколько последних сообщений]\nНапример: /scene 20")
            return
    n = max(3, min(n, SCENE_MAX))

    log = recent_msgs.get(chat_id, [])
    if len(log) < 3:
        await update.message.reply_text("Пока мало сообщений для сцены — пообщайтесь немного и попробуй снова.")
        return
    excerpt = log[-n:]

    rem = _cooldown_left(chat_id)
    if rem > 0:
        await update.message.reply_text(f"⏳ Подожди ещё {int(rem) + 1}с перед следующей картинкой.")
        return
    _mark_image(chat_id)

    await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
    status = await update.message.reply_text(f"🖼 Рисую сцену по последним {len(excerpt)} сообщениям…")
    try:
        prompt = await gen_scene_prompt(chat_id, "\n".join(excerpt))
        if not prompt:
            await status.edit_text("⚠️ Не удалось придумать сцену. Попробуй ещё раз.")
            return
        model = get_image_model(chat_id)
        out, cost = await generate_image(prompt, "16:9", model=model)
        log_cost(chat_id, "image", cost)
        last_gen[chat_id] = {"kind": "imagine", "prompt": prompt, "aspect": "16:9",
                             "src": None, "model": model, "res": None}
        await update.message.reply_photo(
            photo=io.BytesIO(out), caption=f"🖼 Сцена по последним {len(excerpt)} сообщениям",
            reply_markup=more_keyboard())
        await status.delete()
    except httpx.HTTPStatusError as e:
        detail = _openrouter_error_text(e.response)
        logger.error("Scene API error: %s — %s", e.response.status_code, detail)
        await status.edit_text(f"⚠️ Ошибка (HTTP {e.response.status_code}): {detail}")
    except Exception as e:
        logger.exception("Scene error")
        await status.edit_text(f"⚠️ Что-то пошло не так: {e}")


async def send_long(message, text: str) -> None:
    """Send text in <=TG_MAX_LEN chunks (Telegram rejects longer messages)."""
    for i in range(0, len(text), TG_MAX_LEN):
        await message.reply_text(text[i:i + TG_MAX_LEN])


async def _maybe_chime_in(update: Update, context: ContextTypes.DEFAULT_TYPE,
                          chat_id: int, text: str) -> None:
    """Track group chatter and occasionally drop a spontaneous opinion."""
    if not get_chime(chat_id) or not text:
        return
    now = time.time()
    user = update.effective_user
    name = (user.first_name or user.username or "кто-то") if user else "кто-то"
    buf = _grp_buffer[chat_id]
    buf.append((now, f"{name}: {text}"))
    # Keep only messages within the time window (and cap the buffer size).
    cutoff = now - SPONT_WINDOW
    buf[:] = [(ts, t) for ts, t in buf if ts >= cutoff][-SPONT_BUFFER:]

    if len(buf) < SPONT_THRESHOLD:
        return
    if now - _spont_last.get(chat_id, 0) < SPONT_COOLDOWN:
        return

    transcript = "\n".join(t for _, t in buf)
    _spont_last[chat_id] = now
    buf.clear()                       # reset: these count as "with the bot's input"
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        opinion = (await gen_opinion(chat_id, transcript)).strip()
        if opinion:
            await context.bot.send_message(chat_id, opinion)
            add_chime_note(chat_id, opinion)
    except Exception:
        logger.exception("Chime-in failed for chat %s", chat_id)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id, user_id = _ids(update)
    _touch(chat_id)
    text = update.message.text
    # Keep a rolling log of recent chat messages for /scene.
    user = update.effective_user
    name = (user.first_name or user.username or "кто-то") if user else "кто-то"
    recent_msgs[chat_id].append(f"{name}: {text}")
    recent_msgs[chat_id][:] = recent_msgs[chat_id][-RECENT_LOG_MAX:]
    # In groups, only respond when the bot is addressed (mention or reply).
    if _is_group(update):
        if not _addressed_to_bot(update, context):
            await _maybe_chime_in(update, context, chat_id, text)
            return
        text = _strip_bot_mention(text, context)
        if not text:
            await update.message.reply_text("Да? Напиши вопрос вместе с упоминанием.")
            return
        _grp_buffer[chat_id].clear()   # bot is participating → reset chatter counter
        _spont_last[chat_id] = time.time()
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    try:
        await stream_reply(update, chat_id, user_id, text)
        asyncio.create_task(auto_extract_memory(chat_id, user_id, text))
    except httpx.HTTPStatusError as e:
        detail = _openrouter_error_text(e.response)
        logger.error("Chat API error: %s — %s", e.response.status_code, detail)
        await update.message.reply_text(f"⚠️ Ошибка модели (HTTP {e.response.status_code}): {detail}")
    except Exception as e:
        logger.exception("Chat error")
        await update.message.reply_text("⚠️ Что-то пошло не так. Попробуй ещё раз.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """If the bot is asked something → edit the image as requested.
    If a photo arrives with no request → auto 'cursed' remix: analyse it, then
    transform it into an absurd/funny version with a touch of dark humour."""
    chat_id = update.effective_chat.id
    _touch(chat_id)
    msg = update.message

    # Accept both compressed photos and images sent as a file/document.
    if msg.photo:
        file_id = msg.photo[-1].file_id          # largest available size
        mime = "image/jpeg"
    elif msg.document and (msg.document.mime_type or "").startswith("image/"):
        file_id = msg.document.file_id
        mime = msg.document.mime_type
    else:
        return

    # Is there an explicit instruction for the bot?
    caption = (msg.caption or "").strip()
    if _is_group(update):
        # In groups, an instruction only counts if the bot is addressed.
        prompt = _strip_bot_mention(caption, context) if _addressed_to_bot(update, context) else ""
    else:
        prompt = caption

    # No explicit request + cursed mode off → stay quiet.
    if not prompt and not get_cursed(chat_id):
        return

    # Anti-flood: at most one image op per chat per IMAGE_COOLDOWN seconds.
    rem = _cooldown_left(chat_id)
    if rem > 0:
        if prompt:   # only nag on explicit requests; stay quiet for auto-cursed
            await msg.reply_text(f"⏳ Подожди ещё {int(rem) + 1}с перед следующей картинкой.")
        return
    _mark_image(chat_id)

    try:
        raw = await download_tg_file(context, file_id)
        src_url = _data_url(raw, mime)
        await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")

        model = get_image_model(chat_id)
        if prompt:
            # Explicit image-to-image edit
            status = await msg.reply_text(f"🎨 Редактирую по запросу: {prompt}")
            out, cost = await generate_image(prompt, references=[src_url], model=model)
            log_cost(chat_id, "image", cost)
            last_gen[chat_id] = {"kind": "edit", "prompt": prompt, "aspect": "1:1",
                                 "src": src_url, "model": model, "res": None}
            await msg.reply_photo(photo=io.BytesIO(out), caption=f"🎨 {prompt}"[:1024],
                                  reply_markup=more_keyboard())
            await status.delete()
        else:
            # No request → make it cursed & funny (analyse, then transform)
            status = await msg.reply_text("😈 Сейчас будет всрато...")
            instruction = (await ask_multimodal([
                {"type": "text", "text": CURSED_PROMPT},
                {"type": "image_url", "image_url": {"url": src_url}},
            ], chat_id=chat_id, kind="vision")).strip() \
                or "Make this image absurd, cursed and darkly funny, exaggerate everything."
            out, cost = await generate_image(instruction, references=[src_url], model=model)
            log_cost(chat_id, "image", cost)
            last_gen[chat_id] = {"kind": "cursed", "prompt": None, "aspect": "1:1",
                                 "src": src_url, "model": model, "res": None}
            await msg.reply_photo(photo=io.BytesIO(out), caption=random.choice(CURSED_CAPTIONS),
                                  reply_markup=more_keyboard())
            await status.delete()
    except httpx.HTTPStatusError as e:
        detail = _openrouter_error_text(e.response)
        logger.error("Photo API error: %s — %s", e.response.status_code, detail)
        await msg.reply_text(f"⚠️ Ошибка (HTTP {e.response.status_code}): {detail}")
    except Exception as e:
        logger.exception("Photo error")
        await msg.reply_text(f"⚠️ Не получилось обработать фото: {e}")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Transcribe a voice note / audio, then answer it like a normal message."""
    chat_id, user_id = _ids(update)
    _touch(chat_id)
    msg = update.message
    media = msg.voice or msg.audio
    if media is None:
        return
    # In groups, only transcribe voice that replies to / mentions the bot.
    if _is_group(update) and not _addressed_to_bot(update, context):
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    try:
        raw = await download_tg_file(context, media.file_id)
        mime = media.mime_type or "audio/ogg"
        fmt = "ogg" if "ogg" in mime or "opus" in mime else mime.split("/")[-1]

        transcript = (await ask_multimodal([
            {"type": "text", "text": "Транскрибируй это аудио. Верни только текст, без комментариев."},
            {"type": "input_audio", "input_audio": {
                "data": base64.b64encode(raw).decode(), "format": fmt}},
        ], max_tokens=2048, chat_id=chat_id, kind="voice")).strip()

        if not transcript:
            await msg.reply_text("⚠️ Не удалось распознать речь. Попробуй записать ещё раз.")
            return

        await msg.reply_text(f"🎤 «{transcript}»")
        reply = await ask_model(chat_id, user_id, transcript)
        await send_long(msg, reply)
        asyncio.create_task(auto_extract_memory(chat_id, user_id, transcript))
    except httpx.HTTPStatusError as e:
        detail = _openrouter_error_text(e.response)
        logger.error("Voice API error: %s — %s", e.response.status_code, detail)
        await msg.reply_text(f"⚠️ Ошибка распознавания (HTTP {e.response.status_code}): {detail}")
    except Exception:
        logger.exception("Voice error")
        await msg.reply_text("⚠️ Не получилось обработать голосовое. Попробуй ещё раз.")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Что я умею:\n\n"
        "💬 Текст — отвечу (ответ печатается на лету) и запомню беседу.\n"
        "🎙 Голосовое — распознаю и отвечу.\n"
        "🖼 Фото без запроса — переделаю во всратую смешную версию 😈\n"
        "✏️ Фото с подписью/упоминанием — отредактирую как просишь.\n"
        "📄 PDF или текстовый файл — отвечу по содержимому.\n"
        "🔄 Под картинками есть кнопка «Ещё вариант».\n\n"
        "Команды:\n"
        "/model — выбрать модель чата\n"
        "/imagine <промпт> [--ratio 16:9] [--res 2K] — сгенерировать картинку\n"
        "/imgmodel — выбрать модель картинок\n"
        "/scene [N] — картинка по последним N сообщениям 🖼\n"
        "/roast [@кто|reply] — поджарить участника 🔥\n"
        "/summary [N] — саркастичные итоги чата\n"
        "/poll [тема] — шуточный опрос\n"
        "/predict, /tarot, /8ball — предсказания 🔮\n"
        "/say <текст> — отвечу голосом 🎙\n"
        "/cursed — всратый режим 😈\n"
        "/chime — спонтанные вбросы в беседу 🗣\n"
        "/web — веб-поиск 🌐\n"
        "/morning — утренние пожелания 🌅\n"
        "/remember <факт>, /memory, /forget — личная память 🧠\n"
        "/automem — авто-запоминание фактов\n"
        "/stats — расходы 💸\n"
        "/reset — очистить историю\n"
        "/info — текущие модели и статистика\n"
        "/help — эта справка\n\n"
        "🌅 Каждое утро в 8:00 (МСК) желаю хорошего дня в активных чатах."
    )


async def callback_regen(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Regenerate another variant of the last image for this chat."""
    query = update.callback_query
    chat_id = query.message.chat_id
    data = last_gen.get(chat_id)
    if not data:
        await query.answer("Нет данных для повтора 🤷", show_alert=True)
        return
    rem = _cooldown_left(chat_id)
    if rem > 0:
        await query.answer(f"Подожди ещё {int(rem) + 1}с", show_alert=True)
        return
    await query.answer("Генерирую ещё вариант…")
    _mark_image(chat_id)
    model = data.get("model")
    res = data.get("res")
    try:
        kind = data["kind"]
        if kind == "imagine":
            out, cost = await generate_image(data["prompt"], data.get("aspect", "1:1"),
                                             model=model, resolution=res)
            cap = f"🎨 {data['prompt']}"[:1024]
        elif kind == "edit":
            out, cost = await generate_image(data["prompt"], references=[data["src"]], model=model)
            cap = f"🎨 {data['prompt']}"[:1024]
        else:  # cursed
            instruction = (await ask_multimodal([
                {"type": "text", "text": CURSED_PROMPT},
                {"type": "image_url", "image_url": {"url": data["src"]}},
            ], chat_id=chat_id, kind="vision")).strip() or "Make this image absurd and funny."
            out, cost = await generate_image(instruction, references=[data["src"]], model=model)
            cap = random.choice(CURSED_CAPTIONS)
        log_cost(chat_id, "image", cost)
        await context.bot.send_photo(chat_id, photo=io.BytesIO(out),
                                     caption=cap, reply_markup=more_keyboard())
    except Exception as e:
        logger.exception("Regen error")
        await context.bot.send_message(chat_id, f"⚠️ Не вышло сгенерировать ещё вариант: {e}")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    today, total, grand = get_spend(chat_id)
    await update.message.reply_text(
        "💸 Расходы OpenRouter:\n"
        f"• этот чат сегодня: ${today:.4f}\n"
        f"• этот чат всего:   ${total:.4f}\n"
        f"• все чаты всего:   ${grand:.4f}"
    )


def morning_keyboard(chat_id: int) -> InlineKeyboardMarkup:
    on = get_morning(chat_id)
    text = "🔴 Выключить утренние пожелания" if on else "🟢 Включить утренние пожелания"
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(text, callback_data=f"morning:{'off' if on else 'on'}")]]
    )


async def cmd_morning(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    state = "включены 🌅" if get_morning(chat_id) else "выключены"
    await update.message.reply_text(
        f"Утренние пожелания в 8:00 (МСК) сейчас {state}.",
        reply_markup=morning_keyboard(chat_id),
    )


async def callback_morning(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    morning_mode[chat_id] = (query.data.split(":", 1)[1] == "on")
    save_chat_state(chat_id)
    state = "включены 🌅" if morning_mode[chat_id] else "выключены"
    await query.edit_message_text(f"Утренние пожелания {state}.", reply_markup=morning_keyboard(chat_id))


def web_keyboard(chat_id: int) -> InlineKeyboardMarkup:
    on = get_web(chat_id)
    text = "🔴 Выключить веб-поиск" if on else "🟢 Включить веб-поиск"
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(text, callback_data=f"web:{'off' if on else 'on'}")]]
    )


async def cmd_web(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    state = "включён 🌐" if get_web(chat_id) else "выключен"
    await update.message.reply_text(
        f"Веб-поиск сейчас {state}.\n\n"
        "Когда включён, модель может искать в интернете свежую информацию "
        "(это чуть дороже и медленнее).",
        reply_markup=web_keyboard(chat_id),
    )


async def callback_web(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    web_mode[chat_id] = (query.data.split(":", 1)[1] == "on")
    save_chat_state(chat_id)
    state = "включён 🌐" if web_mode[chat_id] else "выключен"
    await query.edit_message_text(f"Веб-поиск {state}.", reply_markup=web_keyboard(chat_id))


def chime_keyboard(chat_id: int) -> InlineKeyboardMarkup:
    on = get_chime(chat_id)
    text = "🔴 Выключить вбросы" if on else "🟢 Включить вбросы"
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(text, callback_data=f"chime:{'off' if on else 'on'}")]]
    )


async def cmd_chime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    state = "включены 🗣" if get_chime(chat_id) else "выключены"
    await update.message.reply_text(
        f"Спонтанные вбросы сейчас {state}.\n\n"
        f"Когда включены, если в чате за ~{SPONT_WINDOW // 60} мин набирается "
        f"{SPONT_THRESHOLD}+ сообщений без меня — я могу вставить своё мнение по теме "
        f"(не чаще раза в {SPONT_COOLDOWN // 60} мин).",
        reply_markup=chime_keyboard(chat_id),
    )


async def callback_chime(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    spontan_mode[chat_id] = (query.data.split(":", 1)[1] == "on")
    save_chat_state(chat_id)
    state = "включены 🗣" if spontan_mode[chat_id] else "выключены"
    await query.edit_message_text(f"Спонтанные вбросы {state}.", reply_markup=chime_keyboard(chat_id))


async def cmd_remember(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _, user_id = _ids(update)
    fact = " ".join(context.args).strip()
    if not fact:
        await update.message.reply_text(
            "Использование: /remember <факт>\n"
            "Например: /remember меня зовут Глеб, люблю краткие ответы"
        )
        return
    if len(memories[user_id]) >= MAX_MEMORIES:
        await update.message.reply_text(
            f"Достигнут лимит {MAX_MEMORIES} фактов. Очисти через /forget."
        )
        return
    add_memory(user_id, fact)
    await update.message.reply_text(f"🧠 Запомнил: {fact}")


async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _, user_id = _ids(update)
    facts = memories.get(user_id)
    if not facts:
        await update.message.reply_text("Пока ничего не запомнено. Добавь через /remember <факт>.")
        return
    lines = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
    await update.message.reply_text("🧠 Я помню:\n" + lines)


async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _, user_id = _ids(update)
    clear_memories(user_id)
    await update.message.reply_text("🧠 Память очищена.")


async def cmd_automem(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _, user_id = _ids(update)
    state = "включена 🧠" if get_automem(user_id) else "выключена"
    await update.message.reply_text(
        f"Авто-память сейчас {state}.\n\n"
        "Когда включена, я сам подмечаю и запоминаю устойчивые факты о тебе из переписки. "
        "Посмотреть — /memory, очистить — /forget.",
        reply_markup=automem_keyboard(user_id),
    )


async def callback_automem(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    _, user_id = _ids(update)
    automem_mode[user_id] = (query.data.split(":", 1)[1] == "on")
    save_user_pref(user_id)
    state = "включена 🧠" if automem_mode[user_id] else "выключена"
    await query.edit_message_text(f"Авто-память {state}.", reply_markup=automem_keyboard(user_id))


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Answer questions about a sent PDF or text file."""
    chat_id = update.effective_chat.id
    _touch(chat_id)
    msg = update.message
    doc = msg.document
    if doc is None:
        return
    if _is_group(update) and not _addressed_to_bot(update, context):
        return

    mime = doc.mime_type or ""
    name = doc.file_name or "file"
    question = _strip_bot_mention((msg.caption or "").strip(), context) \
        or "Кратко перескажи содержимое этого документа и выдели главное."

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    try:
        raw = await download_tg_file(context, doc.file_id)
        is_pdf = mime == "application/pdf" or name.lower().endswith(".pdf")
        is_text = mime.startswith("text/") or name.lower().endswith(
            (".txt", ".md", ".csv", ".json", ".log", ".py", ".js", ".html", ".xml", ".yaml", ".yml"))

        if is_pdf:
            try:
                text = await asyncio.to_thread(_pdf_to_text, raw)
            except Exception:
                logger.exception("PDF extract failed")
                text = ""
            if text.strip():
                answer = await answer_over_text(question, text, name, chat_id)
            else:
                # Scanned/image PDF — fall back to sending the file to a multimodal model.
                answer = await ask_multimodal([
                    {"type": "text", "text": question},
                    {"type": "file", "file": {
                        "filename": name, "file_data": _data_url(raw, "application/pdf")}},
                ], max_tokens=1500, chat_id=chat_id, kind="doc")
        elif is_text:
            text = raw.decode("utf-8", "replace")
            answer = await answer_over_text(question, text, name, chat_id)
        else:
            await msg.reply_text(
                f"⚠️ Формат «{mime or name}» пока не поддерживаю. Пришли PDF или текстовый файл.")
            return

        await send_long(msg, answer or "⚠️ Не удалось разобрать документ.")
    except httpx.HTTPStatusError as e:
        detail = _openrouter_error_text(e.response)
        logger.error("Doc API error: %s — %s", e.response.status_code, detail)
        await msg.reply_text(f"⚠️ Ошибка (HTTP {e.response.status_code}): {detail}")
    except Exception as e:
        logger.exception("Doc error")
        await msg.reply_text(f"⚠️ Не получилось обработать документ: {e}")


def _recent_by_name(chat_id: int, name: str, limit: int = 15) -> list[str]:
    pref = f"{name}:".lower()
    lines = [l for l in recent_msgs.get(chat_id, []) if l.lower().startswith(pref)]
    return lines[-limit:]


ROAST_PROMPT = (
    "Ты — мастер дружеского роаста в групповом чате. Поджарь участника остроумно, едко, "
    "с чёрным юмором и сарказмом; лёгкий мат допустим. Это ДРУЖЕСКАЯ шутка: без реальной "
    "жестокости и угроз, без тем расы, религии, пола, ориентации, инвалидности, здоровья и "
    "внешности как травли, без сексуализации. Опирайся на сообщения человека, если они есть. "
    "2–4 предложения, только текст роаста."
)


async def cmd_roast(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    msg = update.message
    sample: list[str] = []
    if msg.reply_to_message and msg.reply_to_message.from_user:
        u = msg.reply_to_message.from_user
        target = u.first_name or u.username or "этот тип"
        if msg.reply_to_message.text:
            sample.append(f"{target}: {msg.reply_to_message.text}")
        sample += _recent_by_name(chat_id, target)
    elif context.args:
        target = " ".join(context.args).lstrip("@")
        sample = _recent_by_name(chat_id, target)
    else:
        u = update.effective_user
        target = (u.first_name or u.username or "ты") if u else "ты"
        sample = _recent_by_name(chat_id, target)

    ctx = ("Сообщения цели:\n" + "\n".join(sample)) if sample else \
        "Сообщений нет — импровизируй по имени."
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    try:
        roast = (await gen_chat(ROAST_PROMPT, f"Цель роаста: {target}.\n{ctx}",
                                chat_id, "roast", max_tokens=300, temperature=1.0)).strip()
        await msg.reply_text(roast or "Даже шутить не о чем 🙃")
    except Exception as e:
        logger.exception("Roast error")
        await msg.reply_text(f"⚠️ Не вышло: {e}")


async def cmd_summary(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    n = 30
    if context.args:
        try:
            n = int(context.args[0])
        except ValueError:
            pass
    n = max(5, min(n, RECENT_LOG_MAX))
    log = recent_msgs.get(chat_id, [])
    if len(log) < 5:
        await update.message.reply_text("Мало сообщений для итогов. Пообщайтесь ещё.")
        return
    system = ("Ты — саркастичный комментатор чата. Сделай смешное едкое саммари обсуждения, "
              "обращайся к участникам как 'кожаные мешки', чёрный юмор и лёгкий мат ок, "
              "без травли конкретных людей. 3–6 предложений.")
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    try:
        s = await gen_chat(system, "Сообщения:\n" + "\n".join(log[-n:]),
                           chat_id, "summary", max_tokens=500, temperature=0.9)
        await send_long(update.message, s or "Нечего подытожить.")
    except Exception as e:
        logger.exception("Summary error")
        await update.message.reply_text(f"⚠️ Не вышло: {e}")


async def cmd_8ball(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    q = " ".join(context.args).strip()
    if not q:
        await update.message.reply_text("Использование: /8ball <вопрос>\nНапример: /8ball повезёт ли мне сегодня?")
        return
    system = ("Ты — абсурдный магический шар-предсказатель с чёрным юмором. Дай короткий "
              "(1–2 фразы) саркастичный, но смешной ответ на вопрос. Лёгкий мат ок.")
    try:
        ans = await gen_chat(system, q, chat_id, "fortune", max_tokens=120, temperature=1.1)
        await update.message.reply_text("🎱 " + ans.strip())
    except Exception as e:
        logger.exception("8ball error")
        await update.message.reply_text(f"⚠️ Шар треснул: {e}")


async def cmd_predict(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    msg = update.message
    if msg.reply_to_message and msg.reply_to_message.from_user:
        u = msg.reply_to_message.from_user
        target = u.first_name or u.username or "тебя"
    elif context.args:
        target = " ".join(context.args).lstrip("@")
    else:
        u = update.effective_user
        target = (u.first_name or u.username or "тебя") if u else "тебя"
    system = ("Ты выдаёшь абсурдные шуточные предсказания на день с чёрным юмором и сарказмом. "
              "1–3 предложения, конкретно и нелепо.")
    try:
        ans = await gen_chat(system, f"Предскажи сегодняшний день для: {target}",
                             chat_id, "fortune", max_tokens=160, temperature=1.1)
        await msg.reply_text("🔮 " + ans.strip())
    except Exception as e:
        logger.exception("Predict error")
        await msg.reply_text(f"⚠️ Будущее размыто: {e}")


async def cmd_tarot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    system = ("Ты — таролог-шарлатан. Вытяни 3 ВЫДУМАННЫЕ абсурдные карты и дай ехидное "
              "шуточное толкование на сегодня. Назови карты и дай расклад, 3–5 предложений, "
              "чёрный юмор приветствуется.")
    try:
        ans = await gen_chat(system, "Сделай расклад на сегодня.",
                             chat_id, "fortune", max_tokens=300, temperature=1.1)
        await update.message.reply_text("🃏 " + ans.strip())
    except Exception as e:
        logger.exception("Tarot error")
        await update.message.reply_text(f"⚠️ Карты рассыпались: {e}")


async def cmd_poll(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    topic = " ".join(context.args).strip()
    basis = topic or "\n".join(recent_msgs.get(chat_id, [])[-20:])
    if not basis:
        await update.message.reply_text("Использование: /poll <тема> — или просто пообщайтесь, и я возьму тему из чата.")
        return
    system = ('Придумай ОДИН шуточный опрос по теме. Верни СТРОГО JSON без обрамления: '
              '{"question": "...", "options": ["...", "..."]}. 2–4 варианта, вопрос до 250 '
              'символов, вариант до 90 символов, с юмором и сарказмом.')
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    try:
        raw = await gen_chat(system, basis, chat_id, "poll", max_tokens=300, temperature=1.0)
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1])
        question = str(data["question"])[:250]
        options = [str(o)[:90] for o in data["options"] if str(o).strip()][:10]
        if len(options) < 2:
            raise ValueError("not enough options")
        await context.bot.send_poll(chat_id, question, options, is_anonymous=False)
    except Exception as e:
        logger.exception("Poll error")
        await update.message.reply_text(f"⚠️ Опрос не сложился: {e}")


async def cmd_say(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    text = " ".join(context.args).strip()
    if not text:
        await update.message.reply_text("Использование: /say <текст или вопрос> — отвечу голосом 🎙")
        return
    await context.bot.send_chat_action(chat_id=chat_id, action="record_voice")
    try:
        answer = (await gen_chat(
            "Ответь кратко и разговорно, как для зачитывания вслух: 1–3 предложения, "
            "без разметки и эмодзи.", text, chat_id, "chat", max_tokens=200, temperature=0.7)
        ).strip() or text
        audio = await tts_bytes(chat_id, answer)
        bio = io.BytesIO(audio)
        bio.name = "voice.ogg"
        try:
            await update.message.reply_voice(voice=bio, caption=answer[:1024])
        except Exception:
            bio.seek(0)
            await update.message.reply_audio(audio=bio, caption=answer[:1024])
    except httpx.HTTPStatusError as e:
        detail = _openrouter_error_text(e.response)
        logger.error("TTS API error: %s — %s", e.response.status_code, detail)
        await update.message.reply_text(f"⚠️ Голос не вышел (HTTP {e.response.status_code}): {detail}")
    except Exception as e:
        logger.exception("Say error")
        await update.message.reply_text(f"⚠️ Голос не вышел: {e}")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log any uncaught handler exception instead of failing silently."""
    logger.error("Unhandled exception", exc_info=context.error)


async def morning_greeting(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Post a fresh 'cursed' good-morning to every chat active in the last 24h."""
    cutoff = time.time() - 24 * 3600
    recipients = [cid for cid, ts in last_active.items() if ts >= cutoff and get_morning(cid)]
    logger.info("Morning greeting → %d active chats", len(recipients))
    for chat_id in recipients:
        try:
            text = (await gen_text(GREETING_PROMPT, chat_id=chat_id, kind="greeting")).strip()
            if not text:
                text = ("Доброе утро, кожаные мешки 🦴 Хорошего вам дня, а кто сегодня "
                        "тащится на работу — улыбнитесь лишний раз, вам идёт.")
            await context.bot.send_message(chat_id, text)
        except Exception:
            logger.exception("Greeting failed for chat %s", chat_id)


# ── Bot command menu (shown in Telegram UI) ───────────────────────────────────
async def _post_init(app) -> None:
    await app.bot.set_my_commands([
        BotCommand("start",   "Запустить бота"),
        BotCommand("model",   "Выбрать модель"),
        BotCommand("imagine",  "Сгенерировать картинку"),
        BotCommand("imgmodel", "Модель картинок"),
        BotCommand("scene",    "Картинка по последним сообщениям"),
        BotCommand("roast",    "Поджарить участника 🔥"),
        BotCommand("summary",  "Саркастичные итоги чата"),
        BotCommand("poll",     "Шуточный опрос по теме"),
        BotCommand("predict",  "Абсурдное предсказание"),
        BotCommand("tarot",    "Расклад таро-шарлатана"),
        BotCommand("8ball",    "Магический шар"),
        BotCommand("say",      "Ответить голосом 🎙"),
        BotCommand("cursed",   "Всратый режим вкл/выкл"),
        BotCommand("chime",    "Спонтанные вбросы вкл/выкл"),
        BotCommand("web",      "Веб-поиск вкл/выкл"),
        BotCommand("morning",  "Утренние пожелания вкл/выкл"),
        BotCommand("remember", "Запомнить факт"),
        BotCommand("memory",   "Показать память"),
        BotCommand("forget",   "Очистить память"),
        BotCommand("automem",  "Авто-память вкл/выкл"),
        BotCommand("stats",    "Расходы"),
        BotCommand("reset",    "Очистить историю"),
        BotCommand("info",     "Модели и статистика"),
        BotCommand("help",     "Справка"),
    ])
    logger.info("Bot commands registered")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    token = _require_env("TELEGRAM_BOT_TOKEN")
    db_init()
    app = ApplicationBuilder().token(token).post_init(_post_init).build()

    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("help",     cmd_help))
    app.add_handler(CommandHandler("model",    cmd_model_menu))
    app.add_handler(CommandHandler("reset",    cmd_reset))
    app.add_handler(CommandHandler("info",     cmd_info))
    app.add_handler(CommandHandler("imagine",  cmd_imagine))
    app.add_handler(CommandHandler("imgmodel", cmd_imgmodel))
    app.add_handler(CommandHandler("scene",    cmd_scene))
    app.add_handler(CommandHandler("roast",    cmd_roast))
    app.add_handler(CommandHandler("summary",  cmd_summary))
    app.add_handler(CommandHandler("8ball",    cmd_8ball))
    app.add_handler(CommandHandler("predict",  cmd_predict))
    app.add_handler(CommandHandler("tarot",    cmd_tarot))
    app.add_handler(CommandHandler("poll",     cmd_poll))
    app.add_handler(CommandHandler("say",      cmd_say))
    app.add_handler(CommandHandler("cursed",   cmd_cursed))
    app.add_handler(CommandHandler("chime",    cmd_chime))
    app.add_handler(CommandHandler("stats",    cmd_stats))
    app.add_handler(CommandHandler("morning",  cmd_morning))
    app.add_handler(CommandHandler("web",      cmd_web))
    app.add_handler(CommandHandler("remember", cmd_remember))
    app.add_handler(CommandHandler("memory",   cmd_memory))
    app.add_handler(CommandHandler("forget",   cmd_forget))
    app.add_handler(CommandHandler("automem",  cmd_automem))
    app.add_handler(CallbackQueryHandler(callback_set_model, pattern=r"^setmodel:"))
    app.add_handler(CallbackQueryHandler(callback_set_img, pattern=r"^setimg:"))
    app.add_handler(CallbackQueryHandler(callback_cursed, pattern=r"^cursed:"))
    app.add_handler(CallbackQueryHandler(callback_morning, pattern=r"^morning:"))
    app.add_handler(CallbackQueryHandler(callback_web, pattern=r"^web:"))
    app.add_handler(CallbackQueryHandler(callback_chime, pattern=r"^chime:"))
    app.add_handler(CallbackQueryHandler(callback_automem, pattern=r"^automem:"))
    app.add_handler(CallbackQueryHandler(callback_regen, pattern=r"^regen$"))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(on_error)

    # Daily 08:00 Moscow-time greeting to chats active in the last 24h.
    if app.job_queue is not None:
        app.job_queue.run_daily(
            morning_greeting,
            time=dtime(hour=8, minute=0, tzinfo=MOSCOW_TZ),
            name="morning_greeting",
        )
        logger.info("Scheduled daily morning greeting at 08:00 Europe/Moscow")
    else:
        logger.warning("JobQueue unavailable — install python-telegram-bot[job-queue] to enable the daily greeting")

    logger.info("Bot started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

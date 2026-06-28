import os
import io
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
conversations: dict[int, list[dict]] = defaultdict(list)
user_model:    dict[int, str]        = {}   # chat_id → model id
cursed_mode:   dict[int, bool]       = {}   # chat_id → auto "cursed" remix (default ON)
last_active:   dict[int, float]      = {}   # chat_id → epoch seconds of last activity
morning_mode:  dict[int, bool]       = {}   # chat_id → daily greeting on/off (default ON)
web_mode:      dict[int, bool]       = {}   # chat_id → web search on/off (default OFF)
memories:      dict[int, list[str]]  = defaultdict(list)  # chat_id → remembered facts
last_gen:      dict[int, dict]       = {}   # chat_id → last image request (for "more")
_img_cooldown: dict[int, float]      = {}   # chat_id → last image-op timestamp

MAX_HISTORY    = 40
MAX_TOKENS     = 1024
IMAGE_COOLDOWN = 20    # seconds between image operations per chat
MAX_MEMORIES   = 30    # max remembered facts per chat

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
            CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
            CREATE INDEX IF NOT EXISTS idx_usage_chat ON usage_log(chat_id, ts);
            CREATE INDEX IF NOT EXISTS idx_mem_chat ON memories(chat_id, id);
            """
        )
        # Migrations for chat_state columns added over time.
        _add_column("chat_state", "morning", "INTEGER")
        _add_column("chat_state", "web", "INTEGER")
        _db.commit()

        for chat_id, model, cursed, la, morning, web in _db.execute(
            "SELECT chat_id, model, cursed, last_active, morning, web FROM chat_state"
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
        for chat_id, role, content in _db.execute(
            "SELECT chat_id, role, content FROM messages ORDER BY id"
        ):
            conversations[chat_id].append({"role": role, "content": content})
        for chat_id, fact in _db.execute(
            "SELECT chat_id, fact FROM memories ORDER BY id"
        ):
            memories[chat_id].append(fact)
    known_chats = set(user_model) | set(cursed_mode) | set(last_active)
    stored_msgs = sum(len(v) for v in conversations.values())
    logger.info("DB loaded: %d chats, %d stored messages", len(known_chats), stored_msgs)


def save_chat_state(chat_id: int) -> None:
    with _db_lock:
        _db.execute(
            "INSERT INTO chat_state(chat_id, model, cursed, last_active, morning, web) "
            "VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(chat_id) DO UPDATE SET "
            "model=excluded.model, cursed=excluded.cursed, last_active=excluded.last_active, "
            "morning=excluded.morning, web=excluded.web",
            (chat_id, user_model.get(chat_id),
             1 if get_cursed(chat_id) else 0, last_active.get(chat_id),
             1 if get_morning(chat_id) else 0, 1 if get_web(chat_id) else 0),
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


def add_memory(chat_id: int, fact: str) -> None:
    memories[chat_id].append(fact)
    with _db_lock:
        _db.execute("INSERT INTO memories(chat_id, fact, ts) VALUES(?,?,?)",
                    (chat_id, fact, time.time()))
        _db.commit()


def clear_memories(chat_id: int) -> None:
    memories[chat_id].clear()
    with _db_lock:
        _db.execute("DELETE FROM memories WHERE chat_id=?", (chat_id,))
        _db.commit()


def save_message(chat_id: int, role: str, content: str) -> None:
    with _db_lock:
        _db.execute(
            "INSERT INTO messages(chat_id, role, content, ts) VALUES(?,?,?,?)",
            (chat_id, role, content, time.time()),
        )
        # Keep only the most recent MAX_HISTORY rows per chat.
        _db.execute(
            "DELETE FROM messages WHERE chat_id=? AND id NOT IN "
            "(SELECT id FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT ?)",
            (chat_id, chat_id, MAX_HISTORY),
        )
        _db.commit()


def clear_messages(chat_id: int) -> None:
    with _db_lock:
        _db.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        _db.commit()


def get_model(chat_id: int) -> str:
    return user_model.get(chat_id, DEFAULT_CHAT_MODEL)


def get_cursed(chat_id: int) -> bool:
    return cursed_mode.get(chat_id, True)


def get_morning(chat_id: int) -> bool:
    return morning_mode.get(chat_id, True)


def get_web(chat_id: int) -> bool:
    return web_mode.get(chat_id, False)


def build_system(chat_id: int) -> str:
    """System prompt with any remembered facts appended."""
    facts = memories.get(chat_id)
    if not facts:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + "\n\nЗапомненные факты о пользователе:\n" + \
        "\n".join(f"- {f}" for f in facts)


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
def trim_history(chat_id: int) -> None:
    if len(conversations[chat_id]) > MAX_HISTORY:
        conversations[chat_id] = conversations[chat_id][-MAX_HISTORY:]


async def ask_model(chat_id: int, user_text: str) -> str:
    conversations[chat_id].append({"role": "user", "content": user_text})
    trim_history(chat_id)
    save_message(chat_id, "user", user_text)

    messages = [{"role": "system", "content": build_system(chat_id)}] + conversations[chat_id]

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
    conversations[chat_id].append({"role": "assistant", "content": reply})
    save_message(chat_id, "assistant", reply)
    log_cost(chat_id, "chat", _cost_from_response(response))
    return reply


async def stream_reply(update: Update, chat_id: int, user_text: str) -> None:
    """Like ask_model, but edits the Telegram message as tokens stream in."""
    conversations[chat_id].append({"role": "user", "content": user_text})
    trim_history(chat_id)
    save_message(chat_id, "user", user_text)

    messages = [{"role": "system", "content": build_system(chat_id)}] + conversations[chat_id]

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

    conversations[chat_id].append({"role": "assistant", "content": acc})
    save_message(chat_id, "assistant", acc)
    log_cost(chat_id, "chat", cost)


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


def _touch(chat_id: int) -> None:
    """Mark a chat as active right now (used to pick recipients for daily greeting)."""
    last_active[chat_id] = time.time()
    save_chat_state(chat_id)


# ── Image generation ──────────────────────────────────────────────────────────
async def generate_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    references: list[str] | None = None,
) -> tuple[bytes, float]:
    """Returns (image_bytes, cost_usd)."""
    payload = {
        "model": IMAGE_MODEL,
        "prompt": prompt,
    }
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
        "Просто напиши мне — я помню нашу беседу.\n"
        "🎙 Голосовое — распознаю и отвечу.\n"
        "🖼 Фото без запроса — сделаю всратую смешную версию; с запросом — отредактирую.\n\n"
        "Команды:\n"
        "/model   – выбрать модель\n"
        "/reset   – очистить историю\n"
        "/info    – текущая модель и статистика\n"
        "/imagine <prompt> [--ratio 16:9] – сгенерировать картинку\n"
        "/help    – справка"
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
    chat_id = update.effective_chat.id
    conversations[chat_id].clear()
    clear_messages(chat_id)
    await update.message.reply_text("🗑️ История очищена. Начинаем заново!")


async def cmd_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    count = len(conversations[chat_id])
    await update.message.reply_text(
        f"🤖 Модель чата:   {get_model_label(chat_id)}\n"
        f"🎨 Модель картинок: {IMAGE_MODEL}\n"
        f"🌐 Веб-поиск: {'вкл' if get_web(chat_id) else 'выкл'}\n"
        f"😈 Всратый режим: {'вкл' if get_cursed(chat_id) else 'выкл'}\n"
        f"🧠 Запомнено фактов: {len(memories.get(chat_id, []))}\n"
        f"📊 История: {count} сообщений (макс. {MAX_HISTORY})."
    )


async def cmd_imagine(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Использование: /imagine <промпт> [--ratio 16:9]\n\n"
            "Форматы: 1:1 | 16:9 | 9:16 | 4:3 | 3:4\n\n"
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

    prompt = " ".join(args)
    chat_id = update.effective_chat.id

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
        image_bytes, cost = await generate_image(prompt, aspect_ratio)
        log_cost(chat_id, "image", cost)
        last_gen[chat_id] = {"kind": "imagine", "prompt": prompt, "aspect": aspect_ratio, "src": None}
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


async def send_long(message, text: str) -> None:
    """Send text in <=TG_MAX_LEN chunks (Telegram rejects longer messages)."""
    for i in range(0, len(text), TG_MAX_LEN):
        await message.reply_text(text[i:i + TG_MAX_LEN])


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    _touch(chat_id)
    text = update.message.text
    # In groups, only respond when the bot is addressed (mention or reply).
    if _is_group(update):
        if not _addressed_to_bot(update, context):
            return
        text = _strip_bot_mention(text, context)
        if not text:
            await update.message.reply_text("Да? Напиши вопрос вместе с упоминанием.")
            return
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    try:
        await stream_reply(update, chat_id, text)
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

        if prompt:
            # Explicit image-to-image edit
            status = await msg.reply_text(f"🎨 Редактирую по запросу: {prompt}")
            out, cost = await generate_image(prompt, references=[src_url])
            log_cost(chat_id, "image", cost)
            last_gen[chat_id] = {"kind": "edit", "prompt": prompt, "aspect": "1:1", "src": src_url}
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
            out, cost = await generate_image(instruction, references=[src_url])
            log_cost(chat_id, "image", cost)
            last_gen[chat_id] = {"kind": "cursed", "prompt": None, "aspect": "1:1", "src": src_url}
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
    chat_id = update.effective_chat.id
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
        reply = await ask_model(chat_id, transcript)
        await send_long(msg, reply)
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
        "/model — выбрать модель\n"
        "/imagine <промпт> [--ratio 16:9] — сгенерировать картинку\n"
        "/cursed — всратый режим 😈\n"
        "/web — веб-поиск 🌐\n"
        "/morning — утренние пожелания 🌅\n"
        "/remember <факт>, /memory, /forget — личная память 🧠\n"
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
    try:
        kind = data["kind"]
        if kind == "imagine":
            out, cost = await generate_image(data["prompt"], data.get("aspect", "1:1"))
            cap = f"🎨 {data['prompt']}"[:1024]
        elif kind == "edit":
            out, cost = await generate_image(data["prompt"], references=[data["src"]])
            cap = f"🎨 {data['prompt']}"[:1024]
        else:  # cursed
            instruction = (await ask_multimodal([
                {"type": "text", "text": CURSED_PROMPT},
                {"type": "image_url", "image_url": {"url": data["src"]}},
            ], chat_id=chat_id, kind="vision")).strip() or "Make this image absurd and funny."
            out, cost = await generate_image(instruction, references=[data["src"]])
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


async def cmd_remember(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    fact = " ".join(context.args).strip()
    if not fact:
        await update.message.reply_text(
            "Использование: /remember <факт>\n"
            "Например: /remember меня зовут Глеб, люблю краткие ответы"
        )
        return
    if len(memories[chat_id]) >= MAX_MEMORIES:
        await update.message.reply_text(
            f"Достигнут лимит {MAX_MEMORIES} фактов. Очисти через /forget."
        )
        return
    add_memory(chat_id, fact)
    await update.message.reply_text(f"🧠 Запомнил: {fact}")


async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    facts = memories.get(chat_id)
    if not facts:
        await update.message.reply_text("Пока ничего не запомнено. Добавь через /remember <факт>.")
        return
    lines = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
    await update.message.reply_text("🧠 Я помню:\n" + lines)


async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    clear_memories(chat_id)
    await update.message.reply_text("🧠 Память очищена.")


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
            parts = [
                {"type": "text", "text": question},
                {"type": "file", "file": {
                    "filename": name, "file_data": _data_url(raw, "application/pdf")}},
            ]
            answer = await ask_multimodal(parts, max_tokens=1500, chat_id=chat_id, kind="doc")
        elif is_text:
            text = raw.decode("utf-8", "replace")[:20000]
            answer = await ask_multimodal(
                [{"type": "text", "text": f"{question}\n\nСодержимое файла {name}:\n{text}"}],
                max_tokens=1500, chat_id=chat_id, kind="doc")
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
        BotCommand("cursed",   "Всратый режим вкл/выкл"),
        BotCommand("web",      "Веб-поиск вкл/выкл"),
        BotCommand("morning",  "Утренние пожелания вкл/выкл"),
        BotCommand("remember", "Запомнить факт"),
        BotCommand("memory",   "Показать память"),
        BotCommand("forget",   "Очистить память"),
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
    app.add_handler(CommandHandler("cursed",   cmd_cursed))
    app.add_handler(CommandHandler("stats",    cmd_stats))
    app.add_handler(CommandHandler("morning",  cmd_morning))
    app.add_handler(CommandHandler("web",      cmd_web))
    app.add_handler(CommandHandler("remember", cmd_remember))
    app.add_handler(CommandHandler("memory",   cmd_memory))
    app.add_handler(CommandHandler("forget",   cmd_forget))
    app.add_handler(CallbackQueryHandler(callback_set_model, pattern=r"^setmodel:"))
    app.add_handler(CallbackQueryHandler(callback_cursed, pattern=r"^cursed:"))
    app.add_handler(CallbackQueryHandler(callback_morning, pattern=r"^morning:"))
    app.add_handler(CallbackQueryHandler(callback_web, pattern=r"^web:"))
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

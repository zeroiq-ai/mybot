import os
import io
import base64
import logging
import httpx
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
    {"id": "google/gemini-2.0-flash-001",            "label": "🟢 Gemini Flash 2.0"},
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
AUX_MODEL = os.environ.get("OPENROUTER_AUX_MODEL", "google/gemini-2.0-flash-001")

# Telegram hard limit for a single text message.
TG_MAX_LEN = 4096

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant inside a Telegram bot. "
    "Format replies for plain text (no Markdown unless the user asks). "
    "Keep answers focused and practical."
)

# ── Per-user state: model + conversation history ──────────────────────────────
conversations: dict[int, list[dict]] = defaultdict(list)
user_model:    dict[int, str]        = {}   # chat_id → model id

MAX_HISTORY = 40
MAX_TOKENS  = 1024


def get_model(chat_id: int) -> str:
    return user_model.get(chat_id, DEFAULT_CHAT_MODEL)


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

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversations[chat_id]

    response = await client.chat.completions.create(
        model=get_model(chat_id),
        max_tokens=MAX_TOKENS,
        messages=messages,
        extra_headers={"X-Title": "Telegram Claude Bot"},
    )

    reply = response.choices[0].message.content
    if not reply:
        reply = "⚠️ Модель вернула пустой ответ. Попробуй переформулировать."
    conversations[chat_id].append({"role": "assistant", "content": reply})
    return reply


# ── Media helpers ─────────────────────────────────────────────────────────────
def _data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


async def download_tg_file(context: ContextTypes.DEFAULT_TYPE, file_id: str) -> bytes:
    tg_file = await context.bot.get_file(file_id)
    return bytes(await tg_file.download_as_bytearray())


async def ask_multimodal(parts: list[dict], max_tokens: int = 1024) -> str:
    """One-off multimodal call (image/audio understanding). Not stored in history."""
    response = await client.chat.completions.create(
        model=AUX_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": parts}],
        extra_headers={"X-Title": "Telegram Claude Bot"},
    )
    return response.choices[0].message.content or ""


# ── Image generation ──────────────────────────────────────────────────────────
async def generate_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    references: list[str] | None = None,
) -> bytes:
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

    async with httpx.AsyncClient(timeout=120) as http:
        resp = await http.post(
            "https://openrouter.ai/api/v1/images",
            headers=OR_HEADERS,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    items = data.get("data") or []
    if not items:
        raise ValueError(f"No image returned by OpenRouter: {data}")

    item = items[0]
    if item.get("b64_json"):
        import base64
        return base64.b64decode(item["b64_json"])
    if item.get("url"):
        async with httpx.AsyncClient(timeout=60) as http:
            img_resp = await http.get(item["url"])
            img_resp.raise_for_status()
            return img_resp.content
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
        "🖼 Фото без подписи — опишу; фото с подписью — отредактирую.\n\n"
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


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    conversations[update.effective_chat.id].clear()
    await update.message.reply_text("🗑️ История очищена. Начинаем заново!")


async def cmd_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    count = len(conversations[chat_id])
    await update.message.reply_text(
        f"🤖 Модель чата:   {get_model_label(chat_id)}\n"
        f"🎨 Модель картинок: {IMAGE_MODEL}\n"
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

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="upload_photo")
    status_msg = await update.message.reply_text(
        f"🎨 Генерирую...\nПромпт: {prompt}\nФормат: {aspect_ratio}"
    )

    try:
        image_bytes = await generate_image(prompt, aspect_ratio)
        caption = f"🎨 {prompt}"
        if len(caption) > 1024:            # Telegram caption limit
            caption = caption[:1021] + "..."
        await update.message.reply_photo(photo=io.BytesIO(image_bytes), caption=caption)
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
        reply = await ask_model(chat_id, text)
        await send_long(update.message, reply)
    except httpx.HTTPStatusError as e:
        detail = _openrouter_error_text(e.response)
        logger.error("Chat API error: %s — %s", e.response.status_code, detail)
        await update.message.reply_text(f"⚠️ Ошибка модели (HTTP {e.response.status_code}): {detail}")
    except Exception as e:
        logger.exception("Chat error")
        await update.message.reply_text("⚠️ Что-то пошло не так. Попробуй ещё раз.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Photo with a caption → edit it. Photo without a caption → describe/answer."""
    chat_id = update.effective_chat.id
    msg = update.message
    # In groups, only act when the bot is addressed.
    if _is_group(update) and not _addressed_to_bot(update, context):
        return
    photo = msg.photo[-1]              # largest available size
    caption = _strip_bot_mention((msg.caption or "").strip(), context)

    try:
        raw = await download_tg_file(context, photo.file_id)
        src_url = _data_url(raw, "image/jpeg")

        if caption:
            # Image-to-image edit
            await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
            status = await msg.reply_text(f"🎨 Редактирую по запросу: {caption}")
            out = await generate_image(caption, references=[src_url])
            cap = f"🎨 {caption}"[:1024]
            await msg.reply_photo(photo=io.BytesIO(out), caption=cap)
            await status.delete()
        else:
            # Image understanding
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            answer = await ask_multimodal([
                {"type": "text", "text": "Опиши это изображение кратко и по делу."},
                {"type": "image_url", "image_url": {"url": src_url}},
            ])
            # Keep history portable across text-only models: store a text summary.
            conversations[chat_id].append({"role": "user", "content": "[прислал изображение]"})
            conversations[chat_id].append({"role": "assistant", "content": answer})
            trim_history(chat_id)
            await send_long(msg, answer or "⚠️ Не удалось разобрать изображение.")
    except httpx.HTTPStatusError as e:
        detail = _openrouter_error_text(e.response)
        logger.error("Photo API error: %s — %s", e.response.status_code, detail)
        await msg.reply_text(f"⚠️ Ошибка (HTTP {e.response.status_code}): {detail}")
    except Exception:
        logger.exception("Photo error")
        await msg.reply_text("⚠️ Не получилось обработать фото. Попробуй ещё раз.")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Transcribe a voice note / audio, then answer it like a normal message."""
    chat_id = update.effective_chat.id
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
        ], max_tokens=2048)).strip()

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
        "💬 Просто напиши текст — отвечу и запомню беседу.\n"
        "🎙 Пришли голосовое — распознаю и отвечу.\n"
        "🖼 Пришли фото без подписи — опишу его.\n"
        "✏️ Пришли фото с подписью — отредактирую по ней.\n\n"
        "Команды:\n"
        "/model — выбрать модель\n"
        "/reset — очистить историю\n"
        "/info — текущие модели и статистика\n"
        "/imagine <промпт> [--ratio 16:9] — сгенерировать картинку\n"
        "/help — эта справка"
    )


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log any uncaught handler exception instead of failing silently."""
    logger.error("Unhandled exception", exc_info=context.error)


# ── Bot command menu (shown in Telegram UI) ───────────────────────────────────
async def _post_init(app) -> None:
    await app.bot.set_my_commands([
        BotCommand("start",   "Запустить бота"),
        BotCommand("model",   "Выбрать модель"),
        BotCommand("imagine", "Сгенерировать картинку"),
        BotCommand("reset",   "Очистить историю"),
        BotCommand("info",    "Модели и статистика"),
        BotCommand("help",    "Справка"),
    ])
    logger.info("Bot commands registered")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    token = _require_env("TELEGRAM_BOT_TOKEN")
    app = ApplicationBuilder().token(token).post_init(_post_init).build()

    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("help",    cmd_help))
    app.add_handler(CommandHandler("model",   cmd_model_menu))
    app.add_handler(CommandHandler("reset",   cmd_reset))
    app.add_handler(CommandHandler("info",    cmd_info))
    app.add_handler(CommandHandler("imagine", cmd_imagine))
    app.add_handler(CallbackQueryHandler(callback_set_model, pattern=r"^setmodel:"))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(on_error)

    logger.info("Bot started")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

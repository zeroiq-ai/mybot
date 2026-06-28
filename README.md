# OpenRouter Telegram Bot

A Telegram bot backed by [OpenRouter](https://openrouter.ai) with per-chat
conversation memory, a model picker, and image generation.

## Features
- Chat against any OpenRouter model, switchable live with `/model`
- Per-chat conversation history (last 40 messages kept automatically)
- Image generation via `/imagine` (Gemini 2.5 Flash Image / "Nano Banana")
- **Voice notes** — transcribed and answered automatically
- **Photos** — send with a caption to edit the image, or without one to have it described
- Command menu registered in the Telegram UI + `/help`
- `/reset` to wipe history, `/info` to see current models + stats
- "Typing…"/"uploading…" indicators and graceful, specific error messages
- Long replies are split automatically across Telegram's 4096-char limit

## Commands
| Command | Description |
| --- | --- |
| `/start` | Welcome message + current model |
| `/model` | Pick the chat model (inline buttons) |
| `/reset` | Clear conversation history |
| `/info` | Show current chat model, image model, history size |
| `/imagine <prompt> [--ratio 16:9]` | Generate an image |
| `/help` | Show what the bot can do |

You can also just send a **voice note** (transcribed + answered), a **photo
with a caption** (edited to your instruction), or a **photo with no caption**
(described back to you).

Supported `--ratio` values: `1:1 16:9 9:16 4:3 3:4 2:3 3:2 4:5 5:4 21:9`
(anything else falls back to `1:1`).

---

## Step 1 — Get your keys

### Telegram token
1. Open Telegram and search for **@BotFather**
2. Send `/newbot`, follow the prompts
3. BotFather replies with a token like `7123456789:AAFxxxxxx`

### OpenRouter API key
1. Go to https://openrouter.ai/keys
2. Create a key — looks like `sk-or-...`
3. Add some credit (image models are pay-per-image; chat is pay-per-token)

---

## Step 2 — Configure

Copy the example env file and fill it in:

```bash
cp .env.example .env
```

```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=anthropic/claude-sonnet-4-6        # default chat model (optional)
OPENROUTER_IMAGE_MODEL=google/gemini-2.5-flash-image # image model (optional)
OPENROUTER_AUX_MODEL=google/gemini-2.0-flash-001     # vision + voice model (optional)
```

> **Note on the aux model:** voice transcription and photo understanding go
> through `OPENROUTER_AUX_MODEL`, which must accept **image and audio input**.
> Gemini Flash handles both cheaply. Voice notes are sent as OGG/Opus.

> **Note on image models:** use `google/gemini-2.5-flash-image` (the GA
> "Nano Banana"). The older `...-image-preview` slug advertises no supported
> parameters on OpenRouter's image endpoint, so passing `aspect_ratio` to it
> returns an HTTP error — which is the usual cause of "generation failed /
> check your balance" even when the balance is fine.

---

## Step 3 — Run

### Local
```bash
pip install -r requirements.txt
export $(grep -v '^#' .env | xargs)
python bot.py
```

### Deploy to Railway / Render (free)
1. Push this folder to GitHub
2. **Railway:** New Project → Deploy from GitHub repo → add the env vars
   under the Variables tab. The `Procfile` is auto-detected.
3. **Render:** New → Background Worker → Build `pip install -r requirements.txt`,
   Start `python bot.py`, add env vars in the Environment tab.

---

## Notes
- Conversation history is **in-memory only** — it resets when the process
  restarts. For persistence, add SQLite or Redis.
- Switching the chat model with `/model` clears that chat's history.
- No rate limiting/queuing — fine for personal use; add queuing for many users.

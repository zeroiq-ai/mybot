"""Unit tests for bot logic that doesn't require network access."""
import os
import importlib.util
import types
import tempfile

import pytest

# Required env + isolated DB must be set before importing the module.
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "dummy")
os.environ.setdefault("OPENROUTER_API_KEY", "dummy")
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

_spec = importlib.util.spec_from_file_location(
    "bot", os.path.join(os.path.dirname(__file__), "..", "bot.py"))
bot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bot)
bot.db_init()


# ── pure helpers ──────────────────────────────────────────────────────────────
def test_chat_model_id_web_suffix():
    cid = 111
    bot.web_mode.pop(cid, None)
    assert not bot.chat_model_id(cid).endswith(":online")
    bot.web_mode[cid] = True
    assert bot.chat_model_id(cid).endswith(":online")
    bot.web_mode.pop(cid, None)


def test_build_system_includes_memory():
    uid = 222
    bot.memories[uid] = ["любит краткие ответы"]
    assert "любит краткие ответы" in bot.build_system(uid)
    bot.memories.pop(uid, None)
    assert bot.build_system(uid) == bot.SYSTEM_PROMPT


def test_defaults():
    assert bot.get_cursed(999) is True       # cursed default ON
    assert bot.get_web(999) is False         # web default OFF
    assert bot.get_morning(999) is True      # greeting default ON
    assert bot.get_automem(999) is True      # auto-memory default ON
    assert bot.get_image_model(999) == bot.IMAGE_MODEL


def test_cost_from_response():
    resp = types.SimpleNamespace(usage=types.SimpleNamespace(cost=0.0123))
    assert bot._cost_from_response(resp) == pytest.approx(0.0123)
    assert bot._cost_from_response(types.SimpleNamespace(usage=None)) == 0.0


def test_is_retryable():
    import httpx
    req = httpx.Request("POST", "https://x")
    resp500 = httpx.Response(503, request=req)
    resp400 = httpx.Response(400, request=req)
    assert bot._is_retryable(httpx.HTTPStatusError("x", request=req, response=resp500))
    assert not bot._is_retryable(httpx.HTTPStatusError("x", request=req, response=resp400))
    assert bot._is_retryable(httpx.TimeoutException("t"))
    assert not bot._is_retryable(ValueError("nope"))


def test_image_model_supports_res():
    cid = 333
    bot.image_model[cid] = "bytedance-seed/seedream-4.5"   # res=True
    assert bot._image_model_supports_res(cid)
    bot.image_model[cid] = "google/gemini-2.5-flash-image"  # res=False
    assert not bot._image_model_supports_res(cid)
    bot.image_model.pop(cid, None)


def test_cooldown():
    cid = 444
    bot._img_cooldown.pop(cid, None)
    assert bot._cooldown_left(cid) == 0
    bot._mark_image(cid)
    assert bot._cooldown_left(cid) > 0


# ── persistence round-trips ───────────────────────────────────────────────────
def test_message_persistence_and_trim():
    chat, user = 1001, 2002
    bot.clear_messages(chat, user)
    for i in range(bot.MAX_HISTORY + 5):
        bot.save_message(chat, user, "user", f"m{i}")
    rows = bot._db.execute(
        "SELECT content FROM messages WHERE chat_id=? AND user_id=? ORDER BY id",
        (chat, user)).fetchall()
    assert len(rows) == bot.MAX_HISTORY          # trimmed
    assert rows[-1][0] == f"m{bot.MAX_HISTORY + 4}"  # newest kept


def test_memory_persistence_per_user():
    uid = 3003
    bot.clear_memories(uid)
    bot.add_memory(uid, "город Москва")
    facts = [r[0] for r in bot._db.execute(
        "SELECT fact FROM memories WHERE user_id=?", (uid,))]
    assert "город Москва" in facts


def test_cost_logging_and_spend():
    chat = 4004
    bot._db.execute("DELETE FROM usage_log WHERE chat_id=?", (chat,))
    bot._db.commit()
    bot.log_cost(chat, "chat", 0.01)
    bot.log_cost(chat, "image", 0.04)
    bot.log_cost(chat, "chat", 0.0)   # zero cost is ignored
    today, total, grand = bot.get_spend(chat)
    assert total == pytest.approx(0.05)
    assert grand >= total

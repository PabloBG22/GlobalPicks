"""
Script CLI para listar mensajes recientes con su topic (message_thread_id).
"""
from pathlib import Path
import sys
import argparse
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.telegram_client import get_default_client  # noqa: E402


def format_ts(ts: int | None) -> str:
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(ts).isoformat(sep=" ")
    except Exception:
        return str(ts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lista mensajes de Telegram con topic.")
    parser.add_argument("--chat-id", type=int, default=None, help="Filtra por chat_id")
    args = parser.parse_args()

    client = get_default_client()
    msgs = client.list_messages(chat_id=args.chat_id)
    if not msgs:
        print("No hay mensajes en los updates (envía algo al bot y reintenta).")
        return

    print("Mensajes detectados:")
    for msg in msgs:
        chat_id = msg.get("chat_id")
        thread = msg.get("message_thread_id")
        text = msg.get("text") or ""
        date = format_ts(msg.get("date"))
        print(f"- chat_id={chat_id} thread={thread} date={date} text={text!r}")


if __name__ == "__main__":
    main()

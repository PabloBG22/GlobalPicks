"""
Script CLI para enviar un mensaje a un chat (y topic opcional).
"""
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.telegram_client import get_default_client  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Envía un mensaje por Telegram.")
    parser.add_argument("--chat-id", type=int, required=True, help="chat_id destino (ej. -1002774105668)")
    parser.add_argument("--thread-id", type=int, default=None, help="message_thread_id si usas foros/temas")
    parser.add_argument("--text", required=True, help="Texto a enviar")
    parser.add_argument("--parse-mode", default=None, help="Opcional: HTML o MarkdownV2")
    args = parser.parse_args()

    client = get_default_client()
    resp = client.send_message(
        chat_id=args.chat_id,
        text=args.text,
        parse_mode=args.parse_mode,
        thread_id=args.thread_id,
    )
    print("Mensaje enviado:", resp)


if __name__ == "__main__":
    main()

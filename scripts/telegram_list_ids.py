"""
Script CLI para listar los chat_id disponibles desde los últimos updates del bot.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.telegram_client import get_default_client


def main() -> None:
    client = get_default_client()
    chats = client.list_chat_ids()
    if not chats:
        print("No hay chats registrados todavía. Envía un mensaje al bot y vuelve a ejecutar.")
        return

    print("Chats detectados:")
    for chat in chats:
        name = chat.get("title") or chat.get("username") or chat.get("first_name") or ""
        if chat.get("last_name"):
            name = f"{name} {chat.get('last_name')}".strip()
        print(f"- id: {chat.get('id')} | type: {chat.get('type')} | name: {name}")


if __name__ == "__main__":
    main()

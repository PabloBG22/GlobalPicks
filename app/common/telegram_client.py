import requests
from typing import Any, Dict, List, Optional, Union

from app.common.config import settings

# Diccionario de IDs de prueba por mercado y topic
Prueba: Dict[str, Dict[str, Union[str, int]]] = {
    "Corners": {"chat_id": -1002774105668, "thread": 46, "Mercado": "Corners"},
    "Goles": {"chat_id": -1002774105668, "thread": 45, "Mercado": "Goles"},
    "Btts": {"chat_id": -1002774105668, "thread": 35, "Mercado": "Btts"},
    "Gol ht": {"chat_id": -1002774105668, "thread": 36, "Mercado": "Gol ht"},
}

# Diccionario de destinos para el canal GlobalPicks
# Usa las mismas claves que `_mapear_topic_telegram` para que coincidan los envíos
GlobalPicks: Dict[str, Dict[str, Union[str, int, str]]] = {
    "Gol ht": {"chat_id": -1002596320341, "thread": 650, "date": "2025-12-16 10:21:30", "text": "GOL HT"},
    "Btts": {"chat_id": -1002596320341, "thread": 645, "date": "2025-12-16 10:21:45", "text": "BTTS"},
    "Goles": {"chat_id": -1002596320341, "thread": 654, "date": "2025-12-16 10:22:18", "text": "OVER GOLES"},
    "Corners": {"chat_id": -1002596320341, "thread": 1244, "date": "2025-12-16 10:22:32", "text": "CORNERS"},
}


class TelegramClient:
    """
    Cliente mínimo para interactuar con la Bot API de Telegram usando el token configurado.
    """

    def __init__(self, token: Optional[str] = None) -> None:
        self.token = token or settings.telegram_token
        if not self.token:
            raise ValueError("TELEGRAM_TOKEN no está configurado")
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    def get_me(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/getMe", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_updates(self, offset: Optional[int] = None, timeout: int = 0) -> List[Dict[str, Any]]:
        params = {"offset": offset, "timeout": timeout}
        resp = requests.get(f"{self.base_url}/getUpdates", params=params, timeout=max(timeout + 5, 10))
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("result", [])

    def list_chat_ids(self) -> List[Dict[str, Any]]:
        """
        Devuelve chats únicos vistos en los updates recientes con info básica.
        """
        chats: dict[str, Dict[str, Any]] = {}
        for upd in self.get_updates():
            msg = upd.get("message") or upd.get("edited_message") or {}
            chat = msg.get("chat") or {}
            chat_id = chat.get("id")
            if chat_id is None:
                continue
            chats[str(chat_id)] = {
                "id": chat_id,
                "type": chat.get("type"),
                "title": chat.get("title"),
                "username": chat.get("username"),
                "first_name": chat.get("first_name"),
                "last_name": chat.get("last_name"),
            }
        return list(chats.values())

    def list_messages(self, chat_id: Optional[Union[str, int]] = None) -> List[Dict[str, Any]]:
        """
        Devuelve los mensajes recientes. Si se indica chat_id, filtra por ese chat.
        Incluye message_thread_id cuando existe (topics en grupos con foros).
        """
        messages: List[Dict[str, Any]] = []
        for upd in self.get_updates():
            msg = upd.get("message") or upd.get("edited_message")
            if not msg:
                continue
            chat = msg.get("chat") or {}
            if chat_id is not None and chat.get("id") != chat_id:
                continue
            messages.append({
                "update_id": upd.get("update_id"),
                "chat_id": chat.get("id"),
                "chat_type": chat.get("type"),
                "chat_title": chat.get("title"),
                "from": msg.get("from"),
                "date": msg.get("date"),
                "text": msg.get("text"),
                "message_id": msg.get("message_id"),
                "message_thread_id": msg.get("message_thread_id"),  # topic id en foros
            })
        return messages

    def send_message(
        self,
        chat_id: Union[str, int],
        text: str,
        parse_mode: Optional[str] = None,
        thread_id: Optional[int] = None,
        disable_web_page_preview: bool = False,
    ) -> Dict[str, Any]:
        """
        Envía un mensaje al chat indicado. Usa thread_id (message_thread_id) para foros/temas.
        """
        data = {"chat_id": chat_id, "text": text, "disable_web_page_preview": disable_web_page_preview}
        if parse_mode:
            data["parse_mode"] = parse_mode
        if thread_id is not None:
            data["message_thread_id"] = thread_id
        resp = requests.post(f"{self.base_url}/sendMessage", data=data, timeout=10)
        resp.raise_for_status()
        return resp.json()


def get_default_client() -> TelegramClient:
    """
    Devuelve un cliente usando el token de configuración global.
    """
    return TelegramClient()

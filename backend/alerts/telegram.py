"""Client Telegram via API Bot (httpx).

Envoie des messages via l'API Telegram Bot.
Pas de dépendance supplémentaire — httpx est déjà en pyproject.toml.
"""

from __future__ import annotations

import httpx
from loguru import logger

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramClient:
    """Client Telegram pour envoyer des messages via l'API Bot."""

    def __init__(self, bot_token: str, chat_id: str) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._url = TELEGRAM_API_URL.format(token=bot_token)

    async def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Envoie un message Telegram. Retry 1x si timeout."""
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        self._url,
                        json={
                            "chat_id": self._chat_id,
                            "text": text,
                            "parse_mode": parse_mode,
                        },
                    )
                    if response.status_code == 200:
                        return True
                    logger.warning(
                        "Telegram: erreur HTTP {} (tentative {})",
                        response.status_code,
                        attempt + 1,
                    )
            except httpx.TimeoutException:
                logger.warning(
                    "Telegram: timeout (tentative {})", attempt + 1
                )
            except httpx.HTTPError as e:
                logger.warning(
                    "Telegram: erreur réseau {} (tentative {})", e, attempt + 1
                )
                break  # Pas de retry sur erreur réseau non-timeout

        logger.error("Telegram: échec envoi message après 2 tentatives")
        return False

    async def send_trade_alert(self, trade: dict, strategy: str) -> bool:
        """Envoie une alerte de trade clôturé."""
        pnl = trade.get("net_pnl", 0)
        emoji = "+" if pnl >= 0 else ""
        text = (
            f"<b>Trade {strategy}</b>\n"
            f"{trade.get('direction', '?')} "
            f"{trade.get('entry_price', 0):.2f} → {trade.get('exit_price', 0):.2f}\n"
            f"Net: <b>{emoji}{pnl:.2f}$</b>\n"
            f"Raison: {trade.get('exit_reason', '?')}"
        )
        return await self.send_message(text)

    async def send_kill_switch_alert(self, strategy: str, loss_pct: float) -> bool:
        """Envoie une alerte kill switch."""
        text = (
            f"<b>KILL SWITCH</b>\n"
            f"Stratégie: {strategy}\n"
            f"Perte session: {loss_pct:.1f}%\n"
            f"Stratégie arrêtée automatiquement."
        )
        return await self.send_message(text)

    async def send_startup_message(self, strategies: list[str]) -> bool:
        """Envoie un message au démarrage."""
        strats = ", ".join(strategies) if strategies else "aucune"
        text = (
            f"<b>Scalp Radar démarré</b>\n"
            f"Stratégies actives: {strats}"
        )
        return await self.send_message(text)

    async def send_shutdown_message(self) -> bool:
        """Envoie un message à l'arrêt."""
        return await self.send_message("<b>Scalp Radar arrêté</b>")

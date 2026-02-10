"""Rate limiter par catégorie d'endpoint, basé sur token bucket.

Chaque catégorie (market_data, trade, account, position) a son propre bucket
avec un débit configuré dans exchanges.yaml.
"""

from __future__ import annotations

import asyncio
import time

from loguru import logger

from backend.core.config import ExchangeConfig, RateLimitsConfig


class TokenBucket:
    """Token bucket simple pour le rate limiting."""

    def __init__(self, rate: float, category: str) -> None:
        """
        Args:
            rate: Nombre de requêtes autorisées par seconde.
            category: Nom de la catégorie (pour le logging).
        """
        self.rate = rate
        self.category = category
        self.tokens = rate
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
        self.last_refill = now

    async def acquire(self, count: int = 1) -> None:
        """Attend qu'un token soit disponible. Ne drop jamais."""
        async with self._lock:
            while True:
                self._refill()
                if self.tokens >= count:
                    self.tokens -= count
                    return
                # Calcul du temps d'attente
                wait = (count - self.tokens) / self.rate
                logger.debug(
                    "Rate limit {}: attente {:.2f}s ({} tokens demandés)",
                    self.category,
                    wait,
                    count,
                )
                # Release le lock pendant l'attente
                self._lock.release()
                await asyncio.sleep(wait)
                await self._lock.acquire()


class RateLimiter:
    """Rate limiter multi-catégorie pour les API exchange."""

    def __init__(self, config: RateLimitsConfig) -> None:
        self.buckets: dict[str, TokenBucket] = {
            "market_data": TokenBucket(
                config.market_data.requests_per_second, "market_data"
            ),
            "trade": TokenBucket(
                config.trade.requests_per_second, "trade"
            ),
            "account": TokenBucket(
                config.account.requests_per_second, "account"
            ),
            "position": TokenBucket(
                config.position.requests_per_second, "position"
            ),
        }
        logger.debug(
            "RateLimiter initialisé : {}",
            {k: b.rate for k, b in self.buckets.items()},
        )

    async def acquire(self, category: str) -> None:
        """Acquiert un token pour la catégorie donnée."""
        bucket = self.buckets.get(category)
        if not bucket:
            logger.warning("Catégorie rate limit inconnue : {}", category)
            return
        await bucket.acquire()

    async def acquire_multiple(self, category: str, count: int) -> None:
        """Acquiert plusieurs tokens pour un batch de requêtes."""
        bucket = self.buckets.get(category)
        if not bucket:
            logger.warning("Catégorie rate limit inconnue : {}", category)
            return
        await bucket.acquire(count)

    @classmethod
    def from_exchange_config(cls, config: ExchangeConfig) -> RateLimiter:
        """Crée un RateLimiter depuis une ExchangeConfig."""
        return cls(config.rate_limits)

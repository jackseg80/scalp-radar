"""Stratégie Envelope DCA SHORT (Mean Reversion Multi-Niveaux).

Miroir de envelope_dca (LONG). SMA sur le close + N enveloppes hautes.
Entrée SHORT à chaque niveau touché (DCA). TP = retour à la SMA. SL = % au-dessus du prix moyen.
Timeframe : 1h.
"""

from __future__ import annotations

from backend.core.config import EnvelopeDCAShortConfig
from backend.strategies.envelope_dca import EnvelopeDCAStrategy


class EnvelopeDCAShortStrategy(EnvelopeDCAStrategy):
    """Envelope DCA SHORT — Mean Reversion multi-niveaux (direction short).

    Réutilise toute la logique de EnvelopeDCAStrategy.
    Seuls le nom et la direction par défaut (sides=["short"]) changent.
    """

    name = "envelope_dca_short"

    def __init__(self, config: EnvelopeDCAShortConfig) -> None:
        self._config = config

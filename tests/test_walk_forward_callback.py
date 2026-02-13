"""Tests Sprint 14 Bloc B — Progress callback WFO."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from backend.optimization.walk_forward import WalkForwardOptimizer


@pytest.mark.asyncio
async def test_progress_callback_called():
    """Le progress callback est appelé après chaque fenêtre WFO."""
    callback = MagicMock()
    optimizer = WalkForwardOptimizer("config")

    # Ce test nécessite des candles en DB — on skip si pas de données
    # (le test d'intégration complet sera dans le Bloc C avec JobManager)
    try:
        wfo = await optimizer.optimize(
            "envelope_dca", "BTC/USDT",
            is_window_days=60, oos_window_days=20, step_days=20,
            progress_callback=callback,
        )
    except ValueError as exc:
        if "Pas de candles" in str(exc) or "Pas assez de données" in str(exc):
            pytest.skip("Pas de candles en DB pour ce test")
        raise

    # Le callback doit avoir été appelé au moins une fois (une fenêtre min)
    assert callback.call_count >= 1
    # Le 1er appel = ~80% / nb_fenêtres (WFO termine à 80%)
    first_call = callback.call_args_list[0]
    pct, phase = first_call[0]
    assert 0 < pct <= 80.0
    assert "WFO Fenêtre" in phase


@pytest.mark.asyncio
async def test_cancel_event_interrupts():
    """Un cancel_event.set() interrompt l'optimisation."""
    import asyncio

    cancel_event = threading.Event()
    optimizer = WalkForwardOptimizer("config")

    # Setter immédiatement pour tester l'interruption dès la 1ère fenêtre
    cancel_event.set()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await optimizer.optimize(
            "envelope_dca", "BTC/USDT",
            is_window_days=60, oos_window_days=20, step_days=20,
            cancel_event=cancel_event,
        )

    # Vérifier que l'exception contient "annulé"
    assert "annul" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_params_override_merged():
    """params_override fusionne dans la grille default."""
    callback = MagicMock()
    optimizer = WalkForwardOptimizer("config")

    # Override 2 params (grille très réduite pour un test rapide)
    override = {"ma_period": [7], "num_levels": [2]}

    try:
        wfo = await optimizer.optimize(
            "envelope_dca", "BTC/USDT",
            is_window_days=60, oos_window_days=20, step_days=20,
            progress_callback=callback,
            params_override=override,
        )
    except ValueError as exc:
        if "Pas de candles" in str(exc) or "Pas assez de données" in str(exc):
            pytest.skip("Pas de candles en DB pour ce test")
        raise

    # Les params recommandés doivent contenir les valeurs overrides
    assert wfo.recommended_params.get("ma_period") == 7
    assert wfo.recommended_params.get("num_levels") == 2

"""API endpoints pour les résultats d'optimisation WFO."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.core.config import get_config
from backend.optimization.optimization_db import (
    get_comparison_async,
    get_result_by_id_async,
    get_results_async,
)

router = APIRouter(prefix="/api/optimization", tags=["optimization"])


@router.get("/results")
async def get_optimization_results(
    strategy: str | None = Query(default=None, description="Filtrer par stratégie"),
    asset: str | None = Query(default=None, description="Filtrer par asset"),
    min_grade: str | None = Query(default=None, description="Grade minimum (A/B/C/D/F)"),
    latest_only: bool = Query(default=True, description="Seulement les résultats is_latest=1"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    limit: int = Query(default=50, ge=1, le=500, description="Nombre de résultats max"),
) -> dict:
    """Retourne les résultats WFO avec filtres et pagination.

    Response: { results: [...], total: int }
    Chaque résultat contient : id, strategy_name, asset, grade, total_score,
    oos_sharpe, consistency, oos_is_ratio, dsr, param_stability, n_windows,
    created_at, is_latest
    """
    config = get_config()
    # Extraire le path de la DB depuis database_url (format: "sqlite:///data/scalp_radar.db")
    db_url = config.secrets.database_url
    if db_url.startswith("sqlite:///"):
        db_path = db_url[10:]  # Retirer "sqlite:///"
    else:
        db_path = "data/scalp_radar.db"  # Fallback par défaut

    results = await get_results_async(
        db_path=db_path,
        strategy=strategy,
        asset=asset,
        min_grade=min_grade,
        latest_only=latest_only,
        offset=offset,
        limit=limit,
    )
    return results


@router.get("/comparison")
async def get_optimization_comparison() -> dict:
    """Retourne un tableau croisé strategies × assets (is_latest=1 seulement).

    Response: {
        "strategies": ["vwap_rsi", "envelope_dca", ...],
        "assets": ["BTC/USDT", "ETH/USDT", ...],
        "matrix": {
            "vwap_rsi": {
                "BTC/USDT": {"grade": "F", "total_score": 15, "oos_sharpe": -0.5, ...},
                ...
            }
        }
    }

    Cases vides (stratégie non testée sur un asset) : clé absente du dict.
    """
    config = get_config()
    db_url = config.secrets.database_url
    if db_url.startswith("sqlite:///"):
        db_path = db_url[10:]
    else:
        db_path = "data/scalp_radar.db"

    comparison = await get_comparison_async(db_path=db_path)
    return comparison


@router.get("/{result_id}")
async def get_optimization_detail(result_id: int) -> dict:
    """Retourne le détail complet d'un résultat WFO.

    Response inclut best_params, wfo_windows, monte_carlo_summary,
    validation_summary, warnings (JSON parsés).

    404 si result_id inexistant.
    """
    config = get_config()
    db_url = config.secrets.database_url
    if db_url.startswith("sqlite:///"):
        db_path = db_url[10:]
    else:
        db_path = "data/scalp_radar.db"

    result = await get_result_by_id_async(db_path=db_path, result_id=result_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Résultat {result_id} non trouvé")

    return result

"""API endpoints pour les résultats d'optimisation WFO."""

from __future__ import annotations

from fastapi import APIRouter, Body, Header, HTTPException, Query, Response

from backend.core.config import get_config
from backend.optimization.optimization_db import (
    get_comparison_async,
    get_result_by_id_async,
    get_results_async,
    save_result_from_payload_sync,
)

router = APIRouter(prefix="/api/optimization", tags=["optimization"])


def _get_db_path() -> str:
    """Extrait le path DB depuis la config."""
    config = get_config()
    db_url = config.secrets.database_url
    if db_url.startswith("sqlite:///"):
        return db_url[10:]
    return "data/scalp_radar.db"


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
    db_path = _get_db_path()

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
    db_path = _get_db_path()

    comparison = await get_comparison_async(db_path=db_path)
    return comparison


@router.get("/{result_id}")
async def get_optimization_detail(result_id: int) -> dict:
    """Retourne le détail complet d'un résultat WFO.

    Response inclut best_params, wfo_windows, monte_carlo_summary,
    validation_summary, warnings (JSON parsés).

    404 si result_id inexistant.
    """
    db_path = _get_db_path()

    result = await get_result_by_id_async(db_path=db_path, result_id=result_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Résultat {result_id} non trouvé")

    return result


# ─── POST (sync local → serveur) ─────────────────────────────────────────

# Champs NOT NULL du schéma optimization_results (hors id/is_latest/source auto-gérés)
_REQUIRED_FIELDS = [
    "strategy_name", "asset", "timeframe", "created_at",
    "grade", "total_score", "n_windows", "best_params",
]


@router.post("/results", status_code=201)
async def post_optimization_result(
    payload: dict = Body(...),
    x_api_key: str = Header(None, alias="X-API-Key"),
    response: Response = None,
) -> dict:
    """Reçoit un résultat WFO depuis le local et l'insère en DB serveur.

    - 201 si inséré
    - 200 si doublon (déjà existant, UNIQUE constraint)
    - 401 si clé API manquante/invalide
    - 422 si champs obligatoires manquants
    """
    # Auth
    config = get_config()
    server_key = config.secrets.sync_api_key
    if not server_key:
        raise HTTPException(status_code=401, detail="Sync non configuré sur ce serveur")
    if not x_api_key or x_api_key != server_key:
        raise HTTPException(status_code=401, detail="Clé API invalide")

    # Validation payload
    missing = [f for f in _REQUIRED_FIELDS if f not in payload or payload[f] is None]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Champs obligatoires manquants : {', '.join(missing)}",
        )

    # Insert
    db_path = _get_db_path()
    status = save_result_from_payload_sync(db_path, payload)

    if status == "exists":
        response.status_code = 200
        return {"status": "already_exists"}

    return {"status": "created"}

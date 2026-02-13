"""API endpoints pour les résultats d'optimisation WFO."""

from __future__ import annotations

from fastapi import APIRouter, Body, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel

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


# ─── Sprint 14 — Explorateur de Paramètres ────────────────────────────────


class SubmitJobRequest(BaseModel):
    """Payload pour POST /api/optimization/run."""
    strategy_name: str
    asset: str
    params_override: dict | None = None


@router.post("/run")
async def submit_optimization_job(
    request: Request,
    body: SubmitJobRequest,
) -> dict:
    """Soumet un job WFO dans la queue.

    Returns:
        { job_id: str, status: "pending" }

    Errors:
        400: stratégie inconnue
        409: doublon (strategy, asset) pending/running
        429: queue pleine (max 5 pending jobs)
    """
    job_manager = request.app.state.job_manager

    try:
        job_id = await job_manager.submit_job(
            body.strategy_name, body.asset, body.params_override
        )
        return {"job_id": job_id, "status": "pending"}

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        if "déjà en cours" in str(exc):
            raise HTTPException(status_code=409, detail=str(exc))
        if "Queue pleine" in str(exc):
            raise HTTPException(status_code=429, detail=str(exc))
        raise


@router.get("/jobs")
async def list_optimization_jobs(
    request: Request,
    status: str | None = Query(default=None, description="Filtrer par status"),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    """Liste les jobs WFO avec filtre optionnel par status.

    Returns:
        { jobs: [{id, strategy_name, asset, timeframe, status, progress_pct, ...}] }
    """
    job_manager = request.app.state.job_manager
    jobs = await job_manager.list_jobs(status=status, limit=limit)

    return {
        "jobs": [
            {
                "id": j.id,
                "strategy_name": j.strategy_name,
                "asset": j.asset,
                "timeframe": j.timeframe,
                "status": j.status,
                "progress_pct": j.progress_pct,
                "current_phase": j.current_phase,
                "created_at": j.created_at.isoformat(),
                "started_at": j.started_at.isoformat() if j.started_at else None,
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                "duration_seconds": j.duration_seconds,
                "result_id": j.result_id,
                "error_message": j.error_message,
            }
            for j in jobs
        ]
    }


@router.get("/jobs/{job_id}")
async def get_optimization_job(request: Request, job_id: str) -> dict:
    """Détail d'un job WFO.

    Returns:
        Job complet avec tous les champs

    Errors:
        404: job inexistant
    """
    job_manager = request.app.state.job_manager
    job = await job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} non trouvé")

    return {
        "id": job.id,
        "strategy_name": job.strategy_name,
        "asset": job.asset,
        "timeframe": job.timeframe,
        "status": job.status,
        "progress_pct": job.progress_pct,
        "current_phase": job.current_phase,
        "params_override": job.params_override,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "duration_seconds": job.duration_seconds,
        "result_id": job.result_id,
        "error_message": job.error_message,
    }


@router.delete("/jobs/{job_id}")
async def cancel_optimization_job(request: Request, job_id: str) -> dict:
    """Annule un job pending ou running.

    Returns:
        { status: "cancelled" }

    Errors:
        404: job inexistant ou non annulable
    """
    job_manager = request.app.state.job_manager
    cancelled = await job_manager.cancel_job(job_id)

    if not cancelled:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} introuvable ou non annulable (déjà terminé)",
        )

    return {"status": "cancelled"}


@router.get("/param-grid/{strategy_name}")
async def get_param_grid(strategy_name: str) -> dict:
    """Retourne la grille de paramètres pour une stratégie.

    Returns:
        {
            "strategy": "envelope_dca",
            "params": {
                "ma_period": {"values": [5, 7, 10], "default": 7},
                "num_levels": {"values": [2, 3, 4], "default": 2},
                ...
            }
        }

    Errors:
        404: stratégie inconnue
    """
    import yaml
    from pathlib import Path
    from backend.optimization import STRATEGY_REGISTRY

    # Valider que la stratégie est optimisable
    if strategy_name not in STRATEGY_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Stratégie '{strategy_name}' non optimisable. "
            f"Disponibles : {list(STRATEGY_REGISTRY.keys())}",
        )

    # Charger param_grids.yaml
    with open(Path("config/param_grids.yaml"), encoding="utf-8") as f:
        grids = yaml.safe_load(f)
    strategy_grids = grids.get(strategy_name, {}).get("default", {})

    if not strategy_grids:
        raise HTTPException(
            status_code=404,
            detail=f"Aucune grille de paramètres définie pour '{strategy_name}'",
        )

    # Charger strategies.yaml pour les valeurs par défaut
    with open(Path("config/strategies.yaml"), encoding="utf-8") as f:
        strategies_cfg = yaml.safe_load(f)
    strategy_defaults = strategies_cfg.get(strategy_name, {})

    # Construire la réponse
    params = {}
    for param_name, values_list in strategy_grids.items():
        if not isinstance(values_list, list):
            continue
        params[param_name] = {
            "values": values_list,
            "default": strategy_defaults.get(param_name, values_list[0]),
        }

    return {"strategy": strategy_name, "params": params}


@router.get("/heatmap")
async def get_optimization_heatmap(
    strategy: str = Query(..., description="Nom de la stratégie"),
    asset: str = Query(..., description="Asset (ex: BTC/USDT)"),
    param_x: str = Query(..., description="Paramètre axe X"),
    param_y: str = Query(..., description="Paramètre axe Y"),
    metric: str = Query(default="total_score", description="Métrique couleur"),
) -> dict:
    """Retourne une matrice 2D des résultats WFO pour deux paramètres.

    Returns:
        {
            "x_param": "envelope_start",
            "y_param": "envelope_step",
            "metric": "total_score",
            "x_values": [0.05, 0.07, 0.10],
            "y_values": [0.02, 0.03, 0.05],
            "cells": [
                [{value: 45.2, grade: "C", result_id: 123}, {...}, ...],  # row y=0.02
                [{value: 52.1, grade: "B", result_id: 124}, {...}, ...],  # row y=0.03
                ...
            ]
        }

    Les cellules sans données sont représentées par {"value": null}.
    """
    import json
    import aiosqlite

    db_path = _get_db_path()

    # Charger tous les résultats pour (strategy, asset)
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute(
            """SELECT id, grade, total_score, oos_sharpe, consistency, dsr, best_params
               FROM optimization_results
               WHERE strategy_name = ? AND asset = ?
               ORDER BY created_at DESC""",
            (strategy, asset),
        )
        rows = await cursor.fetchall()

    if not rows:
        return {
            "x_param": param_x,
            "y_param": param_y,
            "metric": metric,
            "x_values": [],
            "y_values": [],
            "cells": [],
        }

    # Parser les best_params et extraire (param_x, param_y, metric)
    points: dict[tuple[float, float], dict] = {}  # (x, y) → {value, grade, result_id}

    for row in rows:
        row_dict = dict(row)  # Convertir Row en dict pour .get()

        try:
            params = json.loads(row_dict["best_params"])
        except (json.JSONDecodeError, TypeError):
            continue

        x_val = params.get(param_x)
        y_val = params.get(param_y)
        if x_val is None or y_val is None:
            continue

        # Extraire la valeur de la métrique
        metric_value = row_dict.get(metric)
        if metric_value is None and metric == "total_score":
            metric_value = row_dict["total_score"]

        coord = (float(x_val), float(y_val))
        # Garder le résultat le plus récent pour chaque (x, y)
        if coord not in points:
            points[coord] = {
                "value": metric_value,
                "grade": row_dict["grade"],
                "result_id": row_dict["id"],
            }

    # Construire les axes triés
    x_values = sorted(set(x for x, y in points.keys()))
    y_values = sorted(set(y for x, y in points.keys()))

    # Construire la matrice cells[y_idx][x_idx]
    cells = []
    for y_val in y_values:
        row = []
        for x_val in x_values:
            coord = (x_val, y_val)
            if coord in points:
                row.append(points[coord])
            else:
                row.append({"value": None})
        cells.append(row)

    return {
        "x_param": param_x,
        "y_param": param_y,
        "metric": metric,
        "x_values": x_values,
        "y_values": y_values,
        "cells": cells,
    }


# ─── GET /{result_id} (catch-all à la fin) ───────────────────────────────


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

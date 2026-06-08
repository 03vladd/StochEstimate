"""
Pairs router — watchlist management + analysis pipeline trigger.

Endpoints:
  POST   /api/v1/pairs                       — add pair to watchlist (triggers analysis)
  GET    /api/v1/pairs                       — list user's watchlist
  GET    /api/v1/pairs/{pair_id}             — full analysis detail
  GET    /api/v1/pairs/{pair_id}/job         — poll latest job status
  POST   /api/v1/pairs/{pair_id}/narration   — (re)generate Claude Haiku narration
  PATCH  /api/v1/pairs/{pair_id}/settings    — update alert thresholds
  DELETE /api/v1/pairs/{pair_id}             — remove from watchlist
"""

import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.deps import get_current_user
from app.models.pair import Pair, UserPair
from app.models.user import User
from app.schemas.pair import (
    PairDetailResponse,
    PairSettingsUpdateRequest,
    PairSubmitRequest,
    PairSubmitResponse,
    PairSummaryResponse,
    JobStatusResponse,
    UserPairSettings,
    CointegrationResponse,
    ValidationResponse,
    EstimationResponse,
    MLEResponse,
    SignalPoint,
)
from app.services import analysis_service

router = APIRouter(prefix="/pairs", tags=["pairs"])


# ── POST /pairs ────────────────────────────────────────────────────────────────

@router.post("", response_model=PairSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_pair(
    body: PairSubmitRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Add a ticker pair to the user's watchlist.
    If the pair has never been analysed (or a re-run is needed) an AnalysisJob
    is created and kicked off in the background.  Returns immediately.
    """
    pair = await analysis_service.get_or_create_pair(db, body.ticker1, body.ticker2)
    await analysis_service.get_or_create_user_pair(db, current_user.id, pair.id)

    # Check if we already have a completed analysis with the same window — if so, skip re-run
    existing_job = await analysis_service.get_latest_job(db, pair.id)
    if existing_job and existing_job.status == "complete" and existing_job.window_months == body.window_months:
        await db.commit()
        return PairSubmitResponse(
            pair_id=pair.id,
            job_id=existing_job.id,
            status="cached",
        )

    # Create a new job and run it in the background
    job = await analysis_service.create_analysis_job(db, pair.id, window_months=body.window_months)
    await db.commit()

    # We need a fresh session for the background task (can't share the request session)
    from app.database import AsyncSessionLocal

    async def _bg(job_id: uuid.UUID, pair_id: uuid.UUID, t1: str, t2: str, wm: int):
        async with AsyncSessionLocal() as bg_db:
            await analysis_service.run_analysis_job(job_id, pair_id, t1, t2, bg_db, window_months=wm)

    background_tasks.add_task(_bg, job.id, pair.id, pair.ticker1, pair.ticker2, body.window_months)

    return PairSubmitResponse(pair_id=pair.id, job_id=job.id, status="pending")


# ── GET /pairs ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[PairSummaryResponse])
async def list_pairs(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_pairs = await analysis_service.get_user_watchlist(db, current_user.id)
    summaries = []
    for up in user_pairs:
        p = up.pair
        confidence = None
        if p.validation_result:
            confidence = p.validation_result.confidence_level

        # Latest signal by date
        latest_signal = None
        if p.signals:
            latest_signal = max(p.signals, key=lambda s: s.date)

        summaries.append(
            PairSummaryResponse(
                pair_id=p.id,
                ticker1=p.ticker1,
                ticker2=p.ticker2,
                sector=p.sector,
                confidence_level=confidence,
                latest_z_score=latest_signal.z_score if latest_signal else None,
                latest_signal_type=latest_signal.signal_type if latest_signal else None,
                latest_signal_date=latest_signal.date if latest_signal else None,
                alert_enabled=up.alert_enabled,
            )
        )
    return summaries


# ── GET /pairs/{pair_id} ───────────────────────────────────────────────────────

@router.get("/{pair_id}", response_model=PairDetailResponse)
async def get_pair(
    pair_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Verify user has this pair on their watchlist
    wp_result = await db.execute(
        select(UserPair).where(
            UserPair.user_id == current_user.id,
            UserPair.pair_id == pair_id,
        )
    )
    user_pair = wp_result.scalar_one_or_none()
    if not user_pair:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pair not in your watchlist")

    pair = await analysis_service.get_pair_detail(db, pair_id)
    if not pair:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pair not found")

    signals_sorted = sorted(pair.signals, key=lambda s: s.date) if pair.signals else []

    latest_job = await analysis_service.get_latest_job(db, pair_id)
    return PairDetailResponse(
        pair_id=pair.id,
        ticker1=pair.ticker1,
        ticker2=pair.ticker2,
        sector=pair.sector,
        narration=pair.narration,
        window_months=latest_job.window_months if latest_job else None,
        settings=UserPairSettings(
            alert_enabled=user_pair.alert_enabled,
            long_threshold=user_pair.long_threshold,
            short_threshold=user_pair.short_threshold,
        ),
        cointegration=CointegrationResponse.model_validate(pair.cointegration_result)
        if pair.cointegration_result else None,
        validation=ValidationResponse.model_validate(pair.validation_result)
        if pair.validation_result else None,
        estimation=EstimationResponse.model_validate(pair.estimation_result)
        if pair.estimation_result else None,
        mle=MLEResponse.model_validate(pair.mle_result) if pair.mle_result else None,
        signals=[SignalPoint.model_validate(s) for s in signals_sorted],
    )


# ── GET /pairs/{pair_id}/job ───────────────────────────────────────────────────

@router.get("/{pair_id}/job", response_model=JobStatusResponse)
async def get_job_status(
    pair_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Verify ownership
    wp_result = await db.execute(
        select(UserPair).where(
            UserPair.user_id == current_user.id,
            UserPair.pair_id == pair_id,
        )
    )
    if not wp_result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pair not in your watchlist")

    job = await analysis_service.get_latest_job(db, pair_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No analysis job found for this pair")

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
    )


# ── POST /pairs/{pair_id}/refresh ─────────────────────────────────────────────

@router.post("/{pair_id}/refresh", response_model=JobStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def refresh_pair(
    pair_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Force a full re-analysis regardless of cached state (fetches fresh price data)."""
    wp_result = await db.execute(
        select(UserPair).where(
            UserPair.user_id == current_user.id,
            UserPair.pair_id == pair_id,
        )
    )
    if not wp_result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pair not in your watchlist")

    pair_result = await db.execute(select(Pair).where(Pair.id == pair_id))
    pair = pair_result.scalar_one_or_none()
    if not pair:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pair not found")

    # Reuse the window from the last job so Refresh doesn't change the analysis period
    last_job = await analysis_service.get_latest_job(db, pair_id)
    prev_window = last_job.window_months if last_job else 12

    job = await analysis_service.create_analysis_job(db, pair_id, window_months=prev_window)
    await db.commit()

    from app.database import AsyncSessionLocal

    async def _bg(job_id: uuid.UUID, p_id: uuid.UUID, t1: str, t2: str, wm: int):
        async with AsyncSessionLocal() as bg_db:
            await analysis_service.run_analysis_job(job_id, p_id, t1, t2, bg_db, window_months=wm)

    background_tasks.add_task(_bg, job.id, pair.id, pair.ticker1, pair.ticker2, prev_window)

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
    )


# ── POST /pairs/{pair_id}/narration ───────────────────────────────────────────

@router.post("/{pair_id}/narration")
async def regenerate_narration(
    pair_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    (Re)generate Claude Haiku narration for an already-analysed pair.
    Uses the estimation and validation data already stored in the DB.
    Returns immediately; the narration is written to pairs.narration in the background.
    """
    wp_result = await db.execute(
        select(UserPair).where(
            UserPair.user_id == current_user.id,
            UserPair.pair_id == pair_id,
        )
    )
    if not wp_result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pair not in your watchlist")

    pair = await analysis_service.get_pair_detail(db, pair_id)
    if not pair or not pair.estimation_result or not pair.validation_result:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Pair has no analysis results yet")

    from app.database import AsyncSessionLocal
    from app.services import narration_service
    import asyncio
    import logging
    log = logging.getLogger(__name__)

    # Extract primitive values now, while the session is still open.
    # The background task runs after the request session closes and SQLAlchemy
    # expires all ORM objects — accessing attributes on expired objects raises
    # DetachedInstanceError.
    e = pair.estimation_result
    v = pair.validation_result
    latest_signal = max(pair.signals, key=lambda s: s.date) if pair.signals else None

    t1 = pair.ticker1
    t2 = pair.ticker2
    confidence_level = v.confidence_level
    tests_passed = v.tests_passed_count
    theta = float(e.theta)
    mu = float(e.mu)
    sigma = float(e.sigma)
    theta_ci = (float(e.theta_ci_lower), float(e.theta_ci_upper))
    sigma_ci = (float(e.sigma_ci_lower), float(e.sigma_ci_upper))
    z_score = float(latest_signal.z_score) if latest_signal else None
    signal_type = latest_signal.signal_type if latest_signal else None

    async def _bg():
        try:
            text = await asyncio.to_thread(
                narration_service.generate_narration,
                t1, t2, confidence_level, tests_passed,
                theta, mu, sigma, theta_ci, sigma_ci,
                z_score, signal_type,
                settings.ANTHROPIC_API_KEY,
            )
            if text is None:
                log.error("Narration generation returned None for %s/%s", t1, t2)
                return
            async with AsyncSessionLocal() as bg_db:
                result = await bg_db.execute(select(Pair).where(Pair.id == pair_id))
                p = result.scalar_one_or_none()
                if p:
                    p.narration = text
                    await bg_db.commit()
                    log.info("Narration saved for pair %s", pair_id)
        except Exception:
            log.exception("Background narration task failed for pair %s", pair_id)

    background_tasks.add_task(_bg)
    return {"status": "generating"}


# ── PATCH /pairs/{pair_id}/settings ───────────────────────────────────────────

@router.patch("/{pair_id}/settings", response_model=UserPairSettings)
async def update_pair_settings(
    pair_id: uuid.UUID,
    body: PairSettingsUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    wp_result = await db.execute(
        select(UserPair).where(
            UserPair.user_id == current_user.id,
            UserPair.pair_id == pair_id,
        )
    )
    user_pair = wp_result.scalar_one_or_none()
    if not user_pair:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pair not in your watchlist")

    if body.alert_enabled is not None:
        user_pair.alert_enabled = body.alert_enabled
    if body.long_threshold is not None:
        user_pair.long_threshold = body.long_threshold
    if body.short_threshold is not None:
        user_pair.short_threshold = body.short_threshold

    await db.commit()
    return UserPairSettings(
        alert_enabled=user_pair.alert_enabled,
        long_threshold=user_pair.long_threshold,
        short_threshold=user_pair.short_threshold,
    )


# ── DELETE /pairs/{pair_id} ────────────────────────────────────────────────────

@router.delete("/{pair_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_pair(
    pair_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    wp_result = await db.execute(
        select(UserPair).where(
            UserPair.user_id == current_user.id,
            UserPair.pair_id == pair_id,
        )
    )
    user_pair = wp_result.scalar_one_or_none()
    if not user_pair:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pair not in your watchlist")

    await db.delete(user_pair)
    await db.commit()

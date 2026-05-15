from fastapi import APIRouter
from typing import Dict, Any
from transit_dashboard.backend import main as backend_main

router = APIRouter()


@router.get('/shap/jobs/{task_id}')
def shap_job_status(task_id: str) -> Dict[str, Any]:
    """Return lightweight status for SHAP Celery tasks. Falls back to in-memory store.
    """
    # Try Celery backend first
    try:
        from .tasks import celery_app
        res = celery_app.AsyncResult(task_id)
        status = res.status
        if status == 'SUCCESS':
            return {'task_id': task_id, 'status': 'completed', 'result': res.result}
        elif status in ('PENDING','RECEIVED','STARTED'):
            return {'task_id': task_id, 'status': 'running'}
        else:
            return {'task_id': task_id, 'status': status}
    except Exception:
        # Fallback to in-memory task map
        info = backend_main._SHAP_TASKS.get(task_id)
        if not info:
            return {'task_id': task_id, 'status': 'unknown'}
        return {'task_id': task_id, **info}

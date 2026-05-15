from celery import Celery
import os

CELERY_BROKER = os.getenv('CELERY_BROKER', 'redis://localhost:6379/0')
CELERY_BACKEND = os.getenv('CELERY_BACKEND', 'redis://localhost:6379/1')

celery_app = Celery('transit_dashboard', broker=CELERY_BROKER, backend=CELERY_BACKEND)

@celery_app.task(bind=True)
def compute_shap_task(self, route_id: str, n_background: int):
    """Celery task wrapper that calls the backend SHAP compute function.
    Returns the result dict (the same format as _compute_shap).
    This task also updates the in-memory _SHAP_TASKS map for local status when Celery is not used.
    """
    # Import here to avoid circular import at module load time
    from transit_dashboard.backend import main as backend_main
    # Register task in local map (best-effort)
    try:
        backend_main._SHAP_TASKS[self.request.id] = {
            'status': 'pending',
            'route_id': route_id,
            'n_background': n_background,
            'result': None,
        }
    except Exception:
        pass

    result = backend_main._compute_shap(route_id, n_background)

    # Store result in local map
    try:
        backend_main._SHAP_TASKS[self.request.id]['status'] = 'completed'
        backend_main._SHAP_TASKS[self.request.id]['result'] = result
    except Exception:
        pass

    return result

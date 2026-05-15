from fastapi.testclient import TestClient
from transit_dashboard.backend.main import app

client = TestClient(app)

def test_health_ok():
    r = client.get('/health')
    assert r.status_code == 200
    data = r.json()
    assert data.get('status') == 'ok'


def test_model_info():
    r = client.get('/model/info')
    assert r.status_code == 200
    data = r.json()
    assert 'active' in data

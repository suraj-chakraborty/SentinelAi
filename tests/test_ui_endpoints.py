import time
import unittest
from types import SimpleNamespace
from fastapi.testclient import TestClient

def make_app_with_dummy_orchestrator():
    # Minimal dummy orchestrator to satisfy Phase 5.5 UI tests
    class DummyUserProfiles:
        def __init__(self):
            self._store = {}
        def set_profile(self, user_id, persona=None, memory_ttl_hours=0, opt_in_privacy=True, preferred_tools=None):
            if preferred_tools is None:
                preferred_tools = []
            self._store[user_id] = SimpleNamespace(
                user_id=user_id,
                persona=persona or 'calm Jarvis',
                memory_ttl_hours=memory_ttl_hours,
                opt_in_privacy=opt_in_privacy,
                preferred_tools=preferred_tools,
            )
            return self._store[user_id]
        def get_profile(self, user_id):
            return self._store.get(user_id)

    class DummyOrchestrator:
        def __init__(self):
            self.user_profiles = DummyUserProfiles()
            self.current_user_id = None
            self.jarvis_plan = []
            self._jarvis_progress = 0
            self._jarvis_transcript = []
        def start_jarvis(self, goal):
            self.jarvis_plan = [SimpleNamespace(text=f'Execute: {goal}', done=False)]
            self._jarvis_progress = 0
            return {"status": "planned", "steps": [s.text for s in self.jarvis_plan]}
        def jarvis_execute_next(self):
            if not self.jarvis_plan:
                return {"status": "no_plan"}
            step = self.jarvis_plan.pop(0)
            step_text = step.text
            persona = getattr(self, 'current_user_persona', 'calm Jarvis')
            result = f"Execution with persona {persona}: {step_text}"
            self._jarvis_progress += 1
            self._jarvis_transcript.append({"step": step_text, "result": result, "timestamp": time.time()})
            return {"status": "executed", "step": step_text, "result": result, "remaining": []}
        def jarvis_reflect(self):
            remaining = [s.text for s in self.jarvis_plan]
            return {"status": "in_progress", "progress": self._jarvis_progress, "remaining_steps": remaining}
        def jarvis_status(self):
            return {"plan": [s.text for s in self.jarvis_plan], "progress": self._jarvis_progress}

    return DummyOrchestrator()

class TestUIEndpointsEndToEnd(unittest.TestCase):
    def test_end_to_end_user_profile_and_jarvis_flow(self):
        from sentinel.core.web_server import SentinelWebServer
        app_organizer = make_app_with_dummy_orchestrator()
        server = SentinelWebServer(orchestrator=app_organizer)
        client = TestClient(server.app)

        # 1) Create per-user profile
        resp = client.post('/user/profile', json={
            'persona': 'Calm Jarvis',
            'memory_ttl_hours': 24,
            'opt_in_privacy': True,
            'preferred_tools': ['notes','planner']
        }, headers={'X-User-Id': 'user123'})
        assert resp.status_code == 200
        # 2) Start a Jarvis plan
        resp2 = client.post('/jarvis/plan', json={'goal':'organize downloads'}, headers={'X-User-Id':'user123'})
        assert resp2.status_code == 200
        # 3) Execute next step
        resp3 = client.post('/jarvis/execute_next', headers={'X-User-Id':'user123'})
        assert resp3.status_code == 200
        # 4) Check status and transcript endpoints
        resp_status = client.get('/jarvis/status', headers={'X-User-Id':'user123'})
        assert resp_status.status_code == 200
        resp_transcript = client.get('/jarvis/transcript')
        assert resp_transcript.status_code == 200

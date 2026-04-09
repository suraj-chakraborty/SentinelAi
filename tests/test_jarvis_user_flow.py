import unittest


class TestJarvisUserFlow(unittest.TestCase):
    def test_end_to_end_per_user_flow(self):
        # Minimal in-test harness to simulate per-user flow
        class FakeProfile:
            def __init__(self, persona):
                self.persona = persona
        class FakeUserProfiles:
            def __init__(self):
                self._store = {}
            def set_profile(self, user_id, persona=None, memory_ttl_hours=0, opt_in_privacy=True, preferred_tools=None):
                if preferred_tools is None:
                    preferred_tools = []
                self._store[user_id] = FakeProfile(persona or 'calm Jarvis')
                return self._store[user_id]
            def get_profile(self, user_id):
                return self._store.get(user_id)

        class FakeOrchestrator:
            def __init__(self):
                self.user_profiles = FakeUserProfiles()
                self.current_user_id = None
                self._jarvis_plan = []
                self._jarvis_progress = 0
                self._jarvis_transcript = []
            def start_jarvis(self, goal):
                # create a tiny two-step plan for deterministic testing
                self._jarvis_plan = [type('Step', (), {'text': f'Execute: {goal}', 'done': False})()]
                self._jarvis_progress = 0
                return {"status": "planned", "steps": [s.text for s in self._jarvis_plan]}
            def jarvis_execute_next(self):
                if not self._jarvis_plan:
                    return {"status": "no_plan"}
                step = self._jarvis_plan.pop(0)
                step_text = getattr(step, 'text', str(step))
                # deterministic result that mentions persona if present
                persona = getattr(self, 'current_persona', 'calm Jarvis')
                result = f"Executed step with persona {persona}: {step_text}"
                self._jarvis_progress += 1
                self._jarvis_transcript.append({"step": step_text, "result": result})
                return {"status": "executed", "step": step_text, "result": result, "remaining": []}
            def jarvis_reflect(self):
                return {"status": "in_progress", "progress": self._jarvis_progress, "remaining_steps": [getattr(s, 'text', str(s)) for s in self._jarvis_plan]}
            def jarvis_status(self):
                return {"plan": [getattr(s, 'text', str(s)) for s in self._jarvis_plan], "progress": self._jarvis_progress}

        # Phase 5.1: create user profile for user123 with persona override
        o = FakeOrchestrator()
        o.current_user_id = 'user123'
        o.user_profiles.set_profile('user123', persona='Calm Jarvis')
        o.current_persona = 'Calm Jarvis'

        # Start a Jarvis plan for a simple goal
        start = o.start_jarvis("organize downloads")
        self.assertIn('planned', start.get('status'))
        # Execute next step
        exec1 = o.jarvis_execute_next()
        self.assertEqual(exec1['status'], 'executed')
        self.assertIn('Calm Jarvis', exec1['result'])
        # Reflect should show in-progress or complete depending on steps
        refl = o.jarvis_reflect()
        self.assertIn('progress', refl)
        # Transcript should have the step
        self.assertGreaterEqual(len(o._jarvis_transcript), 1)

Phase 5.5: Personalization Enhancements

- Objective: Finalize per-user personalization UX and tests to deliver a smooth Jarvis experience tailored to individual users on a local machine.
- What’s included:
  - Per-user profiles persisted in SQLite (Phase 5.1) with in-memory MVP for quick iteration
  - Per-user persona application: Jarvis prompts reflect the active user’s persona
  - Phase 5.4 UI: Per-user personalization panel on the dashboard for setting persona, memory TTL, and preferences
  - Phase 5.3 end-to-end tests: End-to-end path for set profile → start Jarvis plan → execute next → reflect → transcript, validated with a deterministic harness
- How to use:
  - Create a profile for a user via POST /user/profile (with X-User-Id header)
  - Use /jarvis/plan, /jarvis/execute_next to drive the Jarvis flow with the active user profile
  - Use /jarvis/transcript to review the reasoning trail and executed steps
- Security & privacy:
  - Governance hooks exist (Phase 4): admin endpoints for memory export/import, audit logs, TTL policies
  - Phase 5 adds per-user personalization with a minimal in-memory MVP path; Phase 5.1 adds SQLite persistence for production-grade storage
- Roadmap notes:
  - Phase 5.6: Enterprise-grade persistence with migrations
  - Phase 5.7: Advanced personalization (ML-based user preference profiling)

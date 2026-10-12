# Phase 4 Governance

- RBAC: admin vs user roles with protected admin endpoints
- Audit logs: track memory exports/imports, admin actions
- Memory TTL: per-topic TTL path introduced; ready for per-topic TTL in Phase 5
- Privacy: memory export/import; prepare for encryption in Phase 5
- UI: governance panel on the dashboard awaiting feedback

Usage guidance:
- Admin endpoints require an admin role or role header
- Use memory export/import to backup and restore memory
- TTL config controls how long memory remains in memory before pruning

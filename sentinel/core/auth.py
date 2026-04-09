from __future__ import annotations

from sentinel.app.config import AUDIT_LOG_PATH

def is_admin(request) -> bool:
    """Simple admin check for HTTP request. Looks for header or query param role=admin."""
    try:
        role = None
        if hasattr(request, 'headers'):
            role = request.headers.get('X-Role') or request.headers.get('X-ROLE')
        if not role:
            role = getattr(request, 'query_params', None).get('role') if request else None
        return str(role).lower() == 'admin'
    except Exception:
        return False

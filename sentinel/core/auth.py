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

def require_admin(func):
    """Lightweight decorator to enforce admin access on async endpoints.

    It looks for an HTTP Request object in the bound arguments and uses
    is_admin to validate. If not authorized, returns a 403 JSON response.
    """
    from functools import wraps
    try:
        from fastapi.responses import JSONResponse
        from fastapi import Request
    except Exception:
        JSONResponse = None
        Request = None

    @wraps(func)
    async def _wrapper(*args, **kwargs):
        request = None
        for a in args:
            if Request is not None and isinstance(a, Request):
                request = a
                break
        if request is None and 'request' in kwargs:
            request = kwargs['request']
        if request is None or not is_admin(request):
            if JSONResponse:
                return JSONResponse({"detail": "Forbidden"}, status_code=403)
            return {"detail": "Forbidden"}
        return await func(*args, **kwargs)

    return _wrapper

class Roles:
    ADMIN = 'admin'
    USER = 'user'

def require_role(role: str):
    """Decorator to require a specific role for an endpoint.
    Reads role from header 'X-Role' or query param 'role'.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            request = None
            for a in args:
                if isinstance(a, object) and hasattr(a, 'headers'):
                    request = a
                    break
            if request is None and 'request' in kwargs:
                request = kwargs['request']
            from fastapi.responses import JSONResponse
            try:
                r = request.headers.get('X-Role') if request and hasattr(request, 'headers') else None
                if not r:
                    r = request.query_params.get('role') if request else None
                if not r or str(r).lower() != str(role).lower():
                    return JSONResponse({"detail": "Forbidden"}, status_code=403)
            except Exception:
                return JSONResponse({"detail": "Forbidden"}, status_code=403)
            return await func(*args, **kwargs)
        return wrapper
    return decorator

#!/usr/bin/env python3
"""L4 HTTP 画像 mock（对接 HttpExternalMemoryAdapter）。

启动:
  python scripts/l4_http_mock_server.py --port 8765

配置:
  MEMORY_PROFILE=l4_http
  external_profiles_http_url: http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote


_PROFILES: dict[str, dict[str, Any]] = {
    "tenant1": {
        "user1": {
            "facts": [
                {"key": "部门", "value": "研发部", "source": "ldap"},
                {"key": "职位", "value": "工程师", "source": "hr"},
                {"key": "工号", "value": "E10001", "source": "crm"},
            ],
            "entities": {
                "武杰": {
                    "canonical_id": "u_wujie",
                    "display_name": "武杰",
                }
            },
        }
    }
}


def _tenant_users(tenant_id: str) -> list[str]:
    return sorted(_PROFILES.get(tenant_id, {}).keys())


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_json(self, code: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _parse_path(self) -> tuple[str, str, str]:
        # /tenants/{tenant}/users/{user}/profile
        parts = [unquote(p) for p in self.path.split("/") if p]
        tenant_id = parts[1] if len(parts) > 1 else ""
        user_id = parts[3] if len(parts) > 3 else ""
        action = parts[4] if len(parts) > 4 else ""
        return tenant_id, user_id, action

    def do_GET(self) -> None:
        tenant_id, user_id, action = self._parse_path()
        if self.path.endswith("/users") and "/tenants/" in self.path:
            self._send_json(200, {"users": _tenant_users(tenant_id)})
            return
        if action == "profile":
            profile = _PROFILES.get(tenant_id, {}).get(user_id)
            if profile is None:
                self._send_json(404, {"error": "not_found"})
                return
            self._send_json(200, profile)
            return
        self._send_json(404, {"error": "unknown_path", "path": self.path})

    def do_PUT(self) -> None:
        tenant_id, user_id, action = self._parse_path()
        if action != "profile":
            self._send_json(404, {"error": "unknown_path"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            profile = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid_json"})
            return
        _PROFILES.setdefault(tenant_id, {})[user_id] = profile
        self._send_json(200, {"ok": True})

    def do_DELETE(self) -> None:
        tenant_id, user_id, action = self._parse_path()
        if action != "profile":
            self._send_json(404, {"error": "unknown_path"})
            return
        removed = _PROFILES.get(tenant_id, {}).pop(user_id, None) is not None
        self._send_json(200, {"deleted": removed})


def main() -> int:
    parser = argparse.ArgumentParser(description="L4 HTTP profile mock")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"L4 mock listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

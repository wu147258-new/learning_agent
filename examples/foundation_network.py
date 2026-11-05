"""Loopback-only HTTP server and client test using Python standard library.

Runs an ephemeral local server on 127.0.0.1. Never contacts the internet.
Run: python examples/foundation_network.py
"""
from __future__ import annotations

import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from urllib.parse import urlsplit, parse_qs


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format_string, *args):
        return  # Do not print simulated requests to logs.

    def reply(self, status, document):
        data = json.dumps(document, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlsplit(self.path)
        if parsed.path == "/api/task":
            query = parse_qs(parsed.query)
            task_id = query.get("id", [""])[0]
            if task_id != "t-001":
                self.reply(404, {"ok": False, "error": "unknown task"})
                return
            self.reply(200, {"task_id": "t-001", "status": "running"})
        else:
            self.reply(404, {"ok": False, "error": "path not found"})

    def do_POST(self):
        self.reply(405, {"ok": False, "error": "method not allowed"})


def request(port, method, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        conn.request(method, path, headers={"Accept": "application/json"})
        response = conn.getresponse()
        status = response.status
        body = response.read()
        doc = json.loads(body.decode("utf-8"))
        return status, doc
    finally:
        conn.close()


def self_test():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    worker = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01},
                    daemon=True)
    worker.start()
    try:
        status, doc = request(port, "GET", "/api/task?id=t-001")
        assert status == 200
        assert doc["task_id"] == "t-001"
        assert doc["status"] == "running"  # HTTP 200 is not business completion!

        status, doc = request(port, "GET", "/api/task?id=unknown")
        assert status == 404 and doc["ok"] is False

        status, doc = request(port, "GET", "/missing")
        assert status == 404 and doc["error"] == "path not found"

        status, doc = request(port, "POST", "/api/task")
        assert status == 405 and doc["error"] == "method not allowed"

        parts = urlsplit("https://example.org:443/api/task?id=t-001")
        assert parts.scheme == "https"
        assert parts.hostname == "example.org"
        assert parts.port == 443
        assert parse_qs(parts.query)["id"] == ["t-001"]
        print("PASS: loopback HTTP, status codes, JSON and URL parsing")
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
        assert not worker.is_alive()


if __name__ == "__main__":
    self_test()

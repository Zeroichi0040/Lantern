import http.client
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app as lantern
from helpers import LanternProcess


class RemoteGuardTests(unittest.TestCase):
    origin = "https://desktop.example.ts.net"
    host = "desktop.example.ts.net"
    login = "owner@example.com"

    def guard(self, headers, method="GET", enabled=True):
        handler = object.__new__(lantern.Handler)
        handler.headers = headers
        handler.command = method
        handler._json = mock.Mock()
        with mock.patch.object(lantern, "REMOTE_ORIGIN", self.origin if enabled else ""), \
                mock.patch.object(lantern, "REMOTE_LOGIN", self.login if enabled else ""):
            result = handler._guard()
        if not result:
            self.assertEqual(handler._json.call_args.args[0], 403)
        return result

    def test_remote_opt_in_and_exact_identity(self):
        headers = {"Host": self.host, "Tailscale-User-Login": self.login}
        self.assertTrue(self.guard(headers))
        self.assertFalse(self.guard(headers, enabled=False))
        self.assertFalse(self.guard({"Host": self.host}))
        self.assertFalse(self.guard({**headers, "Tailscale-User-Login": "other@example.com"}))
        self.assertFalse(self.guard({**headers, "Host": "wrong.example.ts.net"}))

    def test_remote_post_requires_exact_https_origin(self):
        headers = {"Host": self.host, "Tailscale-User-Login": self.login}
        self.assertTrue(self.guard({**headers, "Origin": self.origin}, "POST"))
        for origin in (None, "null", "http://" + self.host, "https://evil.example"):
            h = dict(headers)
            if origin is not None:
                h["Origin"] = origin
            self.assertFalse(self.guard(h, "POST"))

    def test_rewritten_proxy_host_cannot_bypass_guard(self):
        headers = {"Host": f"127.0.0.1:{lantern.UI_PORT}",
                   "X-Forwarded-Host": self.host}
        self.assertFalse(self.guard(headers))
        headers["Tailscale-User-Login"] = self.login
        self.assertTrue(self.guard(headers))
        self.assertFalse(self.guard(headers, enabled=False))
        self.assertFalse(self.guard({**headers, "X-Forwarded-Host": "evil.example"}))

    def test_local_access_stays_available(self):
        host = f"127.0.0.1:{lantern.UI_PORT}"
        self.assertTrue(self.guard({"Host": host}, enabled=False))
        self.assertTrue(self.guard({"Host": host, "Origin": "http://" + host}, "POST"))
        self.assertFalse(self.guard({"Host": host, "Origin": self.origin}, "POST"))

    def test_remote_configuration_validation(self):
        for origin, login, valid in (
            ("", "", True), (self.origin, self.login, True),
            (self.origin, "", False), ("", self.login, False),
            ("http://" + self.host, self.login, False),
            ("https://evil.example", self.login, False),
            (self.origin + "/path", self.login, False),
        ):
            with self.subTest(origin=origin, login=login), \
                    mock.patch.object(lantern, "REMOTE_ORIGIN", origin), \
                    mock.patch.object(lantern, "REMOTE_LOGIN", login):
                if valid:
                    lantern.validate_remote_config()
                else:
                    with self.assertRaises(ValueError):
                        lantern.validate_remote_config()


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = LanternProcess(models=(
            "test-model.gguf",
            "sub/big-00001-of-00002.gguf",
            "sub/big-00002-of-00002.gguf",
            "mmproj-vision.gguf",
            "crash-me.gguf",
        ))

    @classmethod
    def tearDownClass(cls):
        cls.app.close()

    def setUp(self):
        self.app.request("POST", "/api/stop", {})

    # -- workspace configuration
    def test_default_home_detects_shared_llm_lab_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "models").mkdir()
            (workspace / "llama.cpp").mkdir()
            with mock.patch.object(lantern, "HERE", workspace / "lantern"):
                with mock.patch.dict(os.environ, {"LANTERN_HOME": ""}):
                    self.assertEqual(lantern.default_home(), workspace)

    def test_explicit_home_overrides_shared_workspace_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "models").mkdir()
            (workspace / "llama.cpp").mkdir()
            override = workspace / "custom-home"
            with mock.patch.object(lantern, "HERE", workspace / "lantern"):
                with mock.patch.dict(os.environ, {"LANTERN_HOME": str(override)}):
                    self.assertEqual(lantern.default_home(), override)

    # -- discovery
    def test_models_listing_hides_shards_and_mmproj(self):
        status, data = self.app.request("GET", "/api/models")
        self.assertEqual(status, 200)
        names = sorted(m["name"] for m in data["models"])
        self.assertEqual(names, ["crash-me.gguf", "sub/big-00001-of-00002.gguf", "test-model.gguf"])

    def test_index_page_is_served(self):
        status, body = self.app.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Lantern", body)

    def test_status_when_stopped(self):
        _, st = self.app.request("GET", "/api/status")
        self.assertEqual(st["state"], "stopped")
        self.assertTrue(st["server_found"])

    # -- security
    def test_rejects_foreign_host(self):
        status, body = self.app.request("GET", "/api/status", headers={"Host": "evil.example"})
        self.assertEqual(status, 403)

    def test_rejects_foreign_origin(self):
        status, _ = self.app.request("POST", "/api/stop", {}, headers={"Origin": "http://evil.example"})
        self.assertEqual(status, 403)

    def test_accepts_own_origin(self):
        status, _ = self.app.request("POST", "/api/stop", {}, headers={"Origin": self.app.base})
        self.assertEqual(status, 200)

    def test_path_traversal_is_rejected(self):
        status, body = self.app.load(model="../../etc/passwd")
        self.assertEqual(status, 404)

    # -- validation
    def test_invalid_numbers_are_rejected(self):
        self.assertEqual(self.app.load(ctx=10)[0], 400)
        self.assertEqual(self.app.load(threads=0)[0], 400)
        self.assertEqual(self.app.load(ctx="abc")[0], 400)
        self.assertEqual(self.app.load(mode="tpu")[0], 400)

    def test_bad_json_is_rejected(self):
        status, _ = self.app.request("POST", "/api/load", b"{not json")
        self.assertEqual(status, 400)

    def test_chat_before_load_is_503(self):
        status, body = self.app.request("POST", "/v1/chat/completions", {"messages": []})
        self.assertEqual(status, 503)

    # -- lifecycle
    def test_load_chat_stop(self):
        status, st = self.app.load()
        self.assertEqual(status, 200)
        self.assertIn(st["state"], ("loading", "ready"))
        self.app.wait_state("ready")

        conn = http.client.HTTPConnection("127.0.0.1", self.app.ui_port, timeout=15)
        conn.request("POST", "/v1/chat/completions", json.dumps({
            "messages": [{"role": "user", "content": "hello"}], "stream": True,
            "chat_template_kwargs": {"enable_thinking": True}}),
            {"Content-Type": "application/json"})
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        raw = resp.read().decode()
        conn.close()
        self.assertIn("reasoning_content", raw)
        self.assertIn("You said: hello", raw)
        self.assertIn("[DONE]", raw)

        _, st = self.app.request("POST", "/api/stop", {})
        self.assertEqual(st["state"], "stopped")

    def test_reload_switches_settings(self):
        self.app.load(ctx=2048)
        self.app.wait_state("ready")
        _, st = self.app.load(ctx=8192)
        self.assertEqual(st["ctx"], 8192)
        self.app.wait_state("ready")

    def test_crash_is_reported_with_log_tail(self):
        self.app.load(model="crash-me.gguf")
        st = self.app.wait_state("crashed")
        self.assertEqual(st["exit_code"], 3)
        self.assertIn("fake crash", st["log_tail"])

    def test_context_overflow_error_passes_through(self):
        self.app.load()
        self.app.wait_state("ready")
        status, body = self.app.request("POST", "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "OVERFLOW"}], "stream": True})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["type"], "exceed_context_size_error")

    def test_server_survives_client_disconnect_mid_stream(self):
        self.app.load()
        self.app.wait_state("ready")
        conn = http.client.HTTPConnection("127.0.0.1", self.app.ui_port, timeout=15)
        conn.request("POST", "/v1/chat/completions", json.dumps({
            "messages": [{"role": "user", "content": "LONG"}], "stream": True}),
            {"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read(64)
        conn.close()  # simulate the browser pressing Stop
        status, st = self.app.request("GET", "/api/status")
        self.assertEqual(status, 200)
        self.assertEqual(st["state"], "ready")


if __name__ == "__main__":
    unittest.main()

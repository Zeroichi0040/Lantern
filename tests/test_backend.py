import http.client
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import app as lantern
from helpers import LanternProcess, FAKE


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

    def test_foreign_server_blocks_load_and_chat_without_killing_it(self):
        foreign = subprocess.Popen([sys.executable, str(FAKE), "--port",
                                    str(self.app.llama_port), "-m", "foreign.gguf"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    if self.app.request("GET", "/health", port=self.app.llama_port)[0] == 200:
                        break
                except OSError:
                    time.sleep(0.1)
            else:
                self.fail("foreign server did not start")
            self.assertEqual(self.app.request("GET", "/api/status")[1]["state"], "blocked")
            self.assertEqual(self.app.load()[0], 409)
            self.assertEqual(self.app.request("POST", "/v1/chat/completions", {"messages": []})[0], 503)
            self.assertEqual(self.app.request("POST", "/api/stop", {})[1]["state"], "blocked")
            self.assertIsNone(foreign.poll())
        finally:
            foreign.terminate()
            foreign.wait(timeout=10)
        self.assertEqual(self.app.load()[0], 200)
        self.app.wait_state("ready")

    def test_switch_uses_new_verified_instance_and_rejects_stale_chat(self):
        self.app.load()
        before = self.app.wait_state("ready")
        self.app.load(model="sub/big-00001-of-00002.gguf")
        after = self.app.wait_state("ready")
        self.assertEqual(after["model"], "sub/big-00001-of-00002.gguf")
        self.assertNotEqual(before["instance"], after["instance"])
        self.assertEqual(self.app.request("POST", "/v1/chat/completions", {"messages": []},
                         headers={"X-Lantern-Instance": before["instance"]})[0], 409)
        self.assertEqual(self.app.request("POST", "/v1/chat/completions", {"messages": []},
                         headers={"X-Lantern-Instance": after["instance"]})[0], 200)

    def test_wrong_identity_or_model_path_never_becomes_ready(self):
        for name in ("wrong-id.gguf", "wrong-path.gguf"):
            path = self.app.home / "models" / name
            path.write_bytes(b"test")
            try:
                self.assertEqual(self.app.load(model=name)[0], 200)
                st = self.app.wait_state("blocked")
                self.assertNotIn("instance", st)
                self.assertEqual(self.app.request("POST", "/v1/chat/completions", {"messages": []})[0], 503)
            finally:
                self.app.request("POST", "/api/stop", {})
                path.unlink()

    def test_concurrent_loads_are_serialized(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.app.load, ctx=2048)
            second = pool.submit(self.app.load, ctx=8192)
            self.assertEqual(first.result()[0], 200)
            self.assertEqual(second.result()[0], 200)
        st = self.app.wait_state("ready")
        self.assertIn(st["ctx"], (2048, 8192))
        self.assertTrue(st["instance"])

    @unittest.skipUnless(os.name == "nt", "Windows job object")
    def test_windows_job_closes_owned_process(self):
        from process_job import WindowsJob
        job = WindowsJob()
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            job.assign(proc.pid)
            job.close()
            proc.wait(timeout=5)
            self.assertIsNotNone(proc.poll())
        finally:
            job.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait()

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

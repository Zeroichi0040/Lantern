"""Browser tests: drive the real page with Playwright against the fake llama-server.

Skipped automatically when Playwright isn't installed. To use an existing Chrome/Chromium
instead of Playwright's own download, set LANTERN_TEST_CHROME=/path/to/chrome.
"""
import os
import re
import unittest

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:  # pragma: no cover
    sync_playwright = None

from helpers import LanternProcess

READY = re.compile(r"\bready\b")


@unittest.skipIf(sync_playwright is None, "playwright is not installed")
class UITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = LanternProcess(models=("test-model.gguf", "crash-me.gguf"))
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch(
            executable_path=os.environ.get("LANTERN_TEST_CHROME") or None,
            args=["--no-sandbox"], headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls.app.close()

    def setUp(self):
        self.app.request("POST", "/api/stop", {})
        self.ctx = self.browser.new_context(viewport={"width": 1200, "height": 800})
        self.page = self.ctx.new_page()
        self.problems = []
        self.page.on("pageerror", lambda e: self.problems.append("pageerror: %s" % e))
        # HTTP error statuses (e.g. the overflow test's 400) are expected app behaviour,
        # asserted explicitly by the tests, so Chrome's resource-failure log isn't a problem.
        self.page.on("console", lambda m: self.problems.append("console: %s" % m.text)
                     if m.type == "error" and "Failed to load resource" not in m.text else None)
        self.page.goto(self.app.base)
        self.page.select_option("#model", "test-model.gguf")

    def tearDown(self):
        self.ctx.close()
        self.assertEqual(self.problems, [], "browser reported errors")

    # -- helpers
    def load_model(self):
        self.page.click("#load")
        expect(self.page.locator("#pill")).to_have_class(READY, timeout=20000)

    def send(self, text):
        self.page.fill("#input", text)
        self.page.keyboard.press("Enter")

    def last_reply(self):
        return self.page.locator(".msg.assistant").last

    def wait_done(self):
        expect(self.page.locator("#send")).to_have_text("Send", timeout=20000)

    # -- tests
    def test_sidebar_stays_in_viewport_while_chat_scrolls(self):
        # Simulate a long conversation and an over-height sidebar independently.
        self.page.evaluate("""
            () => {
                const wrap = document.querySelector("#wrap");
                wrap.innerHTML = Array.from({length: 50}, (_, i) =>
                    '<div class="msg"><div class="role">Assistant</div>' +
                    '<div class="body">Long conversation message ' + i + '</div></div>'
                ).join("");
                const extra = document.createElement("div");
                extra.style.height = "1200px";
                extra.setAttribute("aria-hidden", "true");
                document.querySelector(".side-scroll").appendChild(extra);
                document.querySelector("#chat").scrollTop =
                    document.querySelector("#chat").scrollHeight;
            }
        """)

        metrics = self.page.evaluate("""
            () => {
                const side = document.querySelector("#side");
                const sideScroll = document.querySelector(".side-scroll");
                const chat = document.querySelector("#chat");
                const rect = side.getBoundingClientRect();
                return {
                    sideTop: rect.top,
                    sideBottom: rect.bottom,
                    viewportHeight: window.innerHeight,
                    pageScrollY: window.scrollY,
                    chatScrollTop: chat.scrollTop,
                    chatCanScroll: chat.scrollHeight > chat.clientHeight,
                    sideCanScroll: sideScroll.scrollHeight > sideScroll.clientHeight
                };
            }
        """)
        self.assertEqual(metrics["sideTop"], 0)
        self.assertLessEqual(metrics["sideBottom"], metrics["viewportHeight"] + 1)
        self.assertEqual(metrics["pageScrollY"], 0)
        self.assertGreater(metrics["chatScrollTop"], 0)
        self.assertTrue(metrics["chatCanScroll"])
        self.assertTrue(metrics["sideCanScroll"])

    def test_assistant_response_can_be_edited_and_cancelled(self):
        self.load_model()
        self.send("original scene")
        self.wait_done()

        original = self.last_reply().locator(".body p").inner_text()
        self.last_reply().locator(".edit-assistant").click()
        editor = self.last_reply().locator("textarea.editbox")
        editor.fill("Rewritten scene with a different ending.")
        self.last_reply().locator(".cancel-assistant").click()
        expect(self.last_reply().locator(".body p")).to_have_text(original)

        self.last_reply().locator(".edit-assistant").click()
        editor = self.last_reply().locator("textarea.editbox")
        editor.fill("Rewritten scene with a different ending.")
        self.last_reply().locator(".save-assistant").click()
        expect(self.last_reply().locator(".body")).to_contain_text(
            "Rewritten scene with a different ending."
        )
        self.assertEqual(self.page.locator(".msg.user").count(), 1)

    def test_continue_creates_separate_bubbles_with_hidden_prompt(self):
        self.load_model()
        self.send("begin the story")
        self.wait_done()

        self.last_reply().locator(".edit-assistant").click()
        self.last_reply().locator("textarea.editbox").fill("The revised scene opens here.")
        self.last_reply().locator(".save-assistant").click()

        self.last_reply().locator(".continue").click()
        self.wait_done()
        expect(self.page.locator(".msg.assistant")).to_have_count(2)
        self.assertEqual(self.last_reply().locator(".body").inner_text(), "Continued scene.")

        _, request = self.app.request("GET", "/last-request", port=self.app.llama_port)
        self.assertNotIn("continue_final_message", request)
        self.assertNotIn("add_generation_prompt", request)
        self.assertEqual(request["messages"][-1], {"role": "user", "content": "Continue."})
        self.assertEqual(request["messages"][-2]["role"], "assistant")
        self.assertEqual(request["messages"][-2]["content"], "The revised scene opens here.")

        self.last_reply().locator(".continue").click()
        self.wait_done()
        expect(self.page.locator(".msg.assistant")).to_have_count(3)
        self.assertEqual(self.last_reply().locator(".body").inner_text(), "Continued scene.")

        _, request = self.app.request("GET", "/last-request", port=self.app.llama_port)
        self.assertNotIn("continue_final_message", request)
        self.assertNotIn("add_generation_prompt", request)
        self.assertEqual(request["messages"][-1], {"role": "user", "content": "Continue."})
        self.assertEqual(request["messages"][-2]["role"], "assistant")
        self.assertEqual(
            request["messages"][-2]["content"],
            "The revised scene opens here. Continued scene."
        )
        # Synthetic prompts are sent to the model but never become visible chat turns.
        expect(self.page.locator(".msg.user")).to_have_count(1)

    def test_settings_screen_persists_ui_preferences_and_contains_generation(self):
        self.load_model()
        self.page.click("#settingsToggle")
        expect(self.page.locator("#settingsPanel")).to_be_visible()
        self.assertEqual(self.page.locator("#side #generationSettings").count(), 0)
        expect(self.page.locator("#settingsPanel #generationSettings")).to_be_visible()
        expect(self.page.locator("#modelPlaceholder")).to_be_checked()
        expect(self.page.locator("#showEdit")).to_be_checked()
        expect(self.page.locator("#showContinue")).to_be_checked()
        expect(self.page.locator("#showDelete")).to_be_checked()
        self.page.locator("#showContinue").uncheck()
        self.page.locator("#modelPlaceholder").uncheck()
        self.page.reload()
        expect(self.page.locator("#settingsPanel")).to_be_hidden()
        expect(self.page.locator("#showContinue")).not_to_be_checked()
        expect(self.page.locator("#modelPlaceholder")).not_to_be_checked()

    def test_message_action_preferences_hide_message_controls(self):
        self.load_model()
        self.send("hide these controls")
        self.wait_done()
        self.page.click("#settingsToggle")
        self.page.locator("#showEdit").uncheck()
        self.page.locator("#showContinue").uncheck()
        self.page.locator("#showDelete").uncheck()
        self.page.click("#settingsToggle")

        expect(self.page.locator(".msg.user .edit")).to_have_count(0)
        expect(self.page.locator(".msg.user .delete-message")).to_have_count(0)
        expect(self.page.locator(".msg.assistant .edit-assistant")).to_have_count(0)
        expect(self.page.locator(".msg.assistant .delete-message")).to_have_count(0)
        expect(self.page.locator(".msg.assistant .continue")).to_have_count(0)

    def test_delete_message_keeps_following_bubbles_by_default(self):
        self.load_model()
        self.send("first scene")
        self.wait_done()
        self.send("second scene")
        self.wait_done()

        expect(self.page.locator(".msg")).to_have_count(4)
        first_user = self.page.locator(".msg.user").first
        first_user.locator(".delete-message").click()
        expect(self.page.locator(".msg").first.locator(".delete-question")).to_have_text(
            "Are you sure you want to delete this message?"
        )
        expect(self.page.locator(".msg").first.locator(".confirm-delete")).to_have_text("Delete")
        self.page.locator(".msg").first.locator(".confirm-delete").click()

        expect(self.page.locator(".msg")).to_have_count(3)
        expect(self.page.locator(".msg.assistant")).to_have_count(2)
        expect(self.page.locator(".msg.user")).to_have_count(1)
        expect(self.page.locator(".msg.assistant").first).to_contain_text("You said: first scene")
        expect(self.page.locator(".msg.user").first.locator(".body")).to_have_text("second scene")

    def test_delete_message_can_remove_following_bubbles(self):
        self.load_model()
        self.send("first scene")
        self.wait_done()
        self.send("second scene")
        self.wait_done()

        self.page.click("#settingsToggle")
        self.page.locator("#deleteFollowing").check()
        self.page.click("#settingsToggle")

        first_assistant = self.page.locator(".msg.assistant").first
        first_assistant.locator(".delete-message").click()
        expect(first_assistant.locator(".delete-question")).to_have_text(
            "Delete this message and all following messages?"
        )
        expect(first_assistant.locator(".confirm-delete")).to_have_text("Delete all")
        first_assistant.locator(".confirm-delete").click()

        expect(self.page.locator(".msg")).to_have_count(1)
        expect(self.page.locator(".msg.user")).to_have_count(1)
        expect(self.page.locator(".msg.assistant")).to_have_count(0)
        expect(self.page.locator(".msg.user").first.locator(".body")).to_have_text("first scene")

    def test_page_loads_and_lists_models(self):
        options = self.page.locator("#model option").all_text_contents()
        self.assertTrue(any("test-model.gguf" in o for o in options))
        expect(self.page.locator("#pill")).to_contain_text("No model loaded")
        expect(self.page.locator("#send")).to_be_disabled()

    def test_chat_with_thinking_code_and_stats(self):
        self.load_model()
        self.send("hi there")
        self.wait_done()
        reply = self.last_reply()
        expect(reply.locator(".body")).to_contain_text("You said: hi there")
        expect(reply.locator("details.think")).to_be_visible()
        expect(reply.locator("details.think .think-body")).to_contain_text("Let me think.")
        expect(reply.locator(".codeblock pre code")).to_contain_text("def hi():")
        expect(reply.locator(".stats")).to_contain_text("tok/s")

    def test_thinking_toggle_is_sent_to_server(self):
        self.load_model()
        self.page.click("#think")
        expect(self.page.locator("#think")).to_have_text("Thinking off")
        self.send("no thoughts")
        self.wait_done()
        expect(self.last_reply().locator("details.think")).to_be_hidden()
        _, last = self.app.request("GET", "/last-request", port=self.app.llama_port)
        self.assertIs(last["chat_template_kwargs"]["enable_thinking"], False)

    def test_stop_button_halts_generation(self):
        self.load_model()
        self.send("LONG")
        expect(self.page.locator("#send")).to_have_text("Stop")
        expect(self.last_reply().locator(".body")).to_contain_text("word3")
        self.page.click("#send")
        self.wait_done()
        expect(self.last_reply().locator(".stats")).to_contain_text("stopped")
        self.assertNotIn("word199", self.last_reply().locator(".body").inner_text())

    def test_context_overflow_error_is_shown(self):
        self.load_model()
        self.send("OVERFLOW")
        self.wait_done()
        expect(self.last_reply().locator(".err")).to_be_visible()

    def test_html_in_messages_is_escaped(self):
        self.load_model()
        self.send("<img src=x onerror=window.__xss=1> <b>bold</b>")
        self.wait_done()
        self.assertIsNone(self.page.evaluate("window.__xss"))
        expect(self.last_reply().locator(".body img")).to_have_count(0)

    def test_chats_persist_across_reload_and_can_be_deleted(self):
        self.load_model()
        self.send("remember me")
        self.wait_done()
        expect(self.page.locator("#convs .conv")).to_have_count(1)
        self.page.reload()
        expect(self.page.locator("#convs .conv").first).to_contain_text("remember me")
        expect(self.page.locator(".msg")).to_have_count(2)
        self.page.hover("#convs .conv")
        self.page.click("#convs .conv button")
        expect(self.page.locator("#convs .conv")).to_have_count(0)

    def test_regenerate_replaces_last_reply(self):
        self.load_model()
        self.send("again")
        self.wait_done()
        self.page.click(".regen")
        self.wait_done()
        expect(self.page.locator(".msg.assistant")).to_have_count(1)

    def test_crashed_model_shows_log(self):
        self.page.select_option("#model", "crash-me.gguf")
        self.page.click("#load")
        expect(self.page.locator("#pill")).to_have_class(re.compile("crashed"), timeout=20000)
        expect(self.page.locator("#hint")).to_contain_text("fake crash")

    # -- step 2 features
    def test_context_meter_shows_usage(self):
        self.load_model()
        expect(self.page.locator("#meter")).to_be_hidden()
        self.send("hello world")
        self.wait_done()
        expect(self.page.locator("#meter")).to_be_visible()
        expect(self.page.locator("#meter span")).to_contain_text("/ 4,096 tokens")

    def test_context_overflow_offers_trim_and_retry(self):
        self.load_model()
        for n in ("one", "two", "three"):
            self.send("BIGMSG %s" % n)
            self.wait_done()
        reply = self.last_reply()
        expect(reply.locator(".err")).to_contain_text("loaded context holds 4,096")
        expect(reply.locator(".trim")).to_be_visible()
        reply.locator(".trim").click()
        self.wait_done()
        expect(self.page.locator(".divider")).to_have_count(1)
        expect(self.last_reply().locator(".body")).to_contain_text("You said: BIGMSG three")
        expect(self.last_reply().locator(".err")).to_be_hidden()

    def test_edit_and_resend_message(self):
        self.load_model()
        self.send("first draft")
        self.wait_done()
        self.page.locator(".msg.user .edit").click()
        self.page.locator("textarea.editbox").fill("second draft")
        self.page.locator(".resend").click()
        self.wait_done()
        expect(self.page.locator(".msg")).to_have_count(2)
        expect(self.page.locator(".msg.user .body")).to_have_text("second draft")
        expect(self.last_reply().locator(".body")).to_contain_text("You said: second draft")

    def test_edit_can_be_cancelled(self):
        self.load_model()
        self.send("keep me")
        self.wait_done()
        self.page.locator(".msg.user .edit").click()
        self.page.locator("textarea.editbox").fill("changed my mind")
        self.page.locator(".cancel").click()
        expect(self.page.locator(".msg.user .body")).to_have_text("keep me")

    def test_rename_chat_from_sidebar_and_header(self):
        self.load_model()
        self.send("name me")
        self.wait_done()
        self.page.locator("#convs .conv span").dblclick()
        self.page.locator("#convs input.rename").fill("Project Notes")
        self.page.keyboard.press("Enter")
        expect(self.page.locator("#convs .conv span")).to_have_text("Project Notes")
        expect(self.page.locator("#title")).to_have_text("Project Notes")
        self.page.locator("#title").dblclick()
        self.page.locator(".top input.rename").fill("Renamed Again")
        self.page.keyboard.press("Enter")
        expect(self.page.locator("#title")).to_have_text("Renamed Again")
        self.page.reload()
        expect(self.page.locator("#convs .conv span")).to_have_text("Renamed Again")

    def test_search_chats(self):
        self.load_model()
        self.send("alpha topic")
        self.wait_done()
        self.page.click("#newChat")
        self.send("beta topic")
        self.wait_done()
        expect(self.page.locator("#convs .conv")).to_have_count(2)
        self.page.fill("#search", "beta")
        expect(self.page.locator("#convs .conv")).to_have_count(1)
        self.page.fill("#search", "zzz")
        expect(self.page.locator("#convs")).to_contain_text("No chats match")
        self.page.fill("#search", "You said: alpha")   # matches message text, not just titles
        expect(self.page.locator("#convs .conv")).to_have_count(1)

    def test_markdown_table_and_syntax_highlighting(self):
        self.load_model()
        self.send("TABLE please")
        self.wait_done()
        body = self.last_reply().locator(".body")
        expect(body.locator("table")).to_have_count(1)
        expect(body.locator("tbody tr")).to_have_count(2)
        expect(body.locator("th")).to_have_text(["Name", "Size"])
        expect(body.locator(".tk-k").first).to_have_text("def")
        expect(body.locator(".tk-c")).to_have_text("# greet")
        expect(body.locator(".tk-s")).to_have_text("'x'")

    def test_advanced_sampling_is_sent(self):
        self.load_model()
        self.page.locator("details.adv summary").click()
        self.page.fill("#topk", "12")
        self.page.fill("#minp", "0.1")
        self.page.fill("#repeat", "1.2")
        self.page.fill("#seed", "42")
        self.send("sampling")
        self.wait_done()
        _, last = self.app.request("GET", "/last-request", port=self.app.llama_port)
        self.assertEqual((last["top_k"], last["min_p"], last["repeat_penalty"], last["seed"]),
                         (12, 0.1, 1.2, 42))

    def test_settings_are_remembered_per_model(self):
        self.page.fill("#ctx", "2048")                       # test-model
        self.page.click("#think")                            # thinking off for test-model
        self.page.select_option("#model", "crash-me.gguf")
        self.assertEqual(self.page.input_value("#ctx"), "4096")
        expect(self.page.locator("#think")).to_have_text("Thinking on")
        self.page.fill("#ctx", "8192")
        self.page.select_option("#model", "test-model.gguf")
        self.assertEqual(self.page.input_value("#ctx"), "2048")
        expect(self.page.locator("#think")).to_have_text("Thinking off")
        self.page.select_option("#model", "crash-me.gguf")
        self.assertEqual(self.page.input_value("#ctx"), "8192")

    def test_escape_key_stops_generation(self):
        self.load_model()
        self.send("LONG")
        expect(self.last_reply().locator(".body")).to_contain_text("word3")
        self.page.keyboard.press("Escape")
        self.wait_done()
        expect(self.last_reply().locator(".stats")).to_contain_text("stopped")

    def test_mobile_layout_has_no_horizontal_scroll(self):
        self.page.set_viewport_size({"width": 390, "height": 800})
        overflow = self.page.evaluate(
            "document.documentElement.scrollWidth > document.documentElement.clientWidth")
        self.assertFalse(overflow)
        app = self.page.locator("#app")
        self.page.click("#menu")
        expect(app).to_have_class(re.compile("open"))
        self.page.click("#closeSide")                  # the open sidebar covers the ☰ button
        expect(app).not_to_have_class(re.compile("open"))
        self.page.click("#menu")
        expect(app).to_have_class(re.compile("open"))
        self.page.mouse.click(380, 400)                # the dimmed strip beside the sidebar
        expect(app).not_to_have_class(re.compile("open"))


if __name__ == "__main__":
    unittest.main()

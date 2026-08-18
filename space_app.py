"""Entrypoint for Hugging Face Spaces.

HF requires the container to listen on a port (7860). We run a tiny health
server there — it also doubles as the URL an uptime pinger hits to keep the
free Space from sleeping — and then start the Telegram bot in the main thread.
"""
import os
import threading
import http.server
import socketserver

PORT = int(os.environ.get("PORT", "7860"))


class _Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):  # silence access logs
        pass


def _serve_health():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), _Health) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    threading.Thread(target=_serve_health, daemon=True).start()
    import bot
    bot.main()

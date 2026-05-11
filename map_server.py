from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import webbrowser

from config import MapServerConfig
from landmarks import MAP_TEMPLATE_PATH, WEB_ROOT
from map_state import SharedMapState

SCREENSHOTS_DIR = Path(__file__).resolve().parent / "screenshots"
SCREENSHOT_PATH = SCREENSHOTS_DIR / "latest_map_crop.png"


def build_map_handler(shared_state: SharedMapState) -> type[BaseHTTPRequestHandler]:
    classifier = None

    class MapRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                self._send_html(MAP_TEMPLATE_PATH.read_text(encoding="utf-8"))
                return

            if self.path == "/state":
                self._send_json(shared_state.snapshot())
                return

            asset_path = self._resolve_asset_path(self.path)
            if asset_path is not None:
                self._send_file(asset_path)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/screenshots/latest_map_crop.png":
                self._save_screenshot()
                return

            if self.path == "/classify/latest_map_crop.png":
                self._classify_screenshot()
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        @staticmethod
        def _resolve_asset_path(request_path: str) -> Path | None:
            relative_path = request_path.lstrip("/")
            asset_path = (WEB_ROOT / relative_path).resolve()
            try:
                asset_path.relative_to(WEB_ROOT.resolve())
            except ValueError:
                return None

            if not asset_path.is_file():
                return None
            return asset_path

        def _send_html(self, content: str) -> None:
            payload = content.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_file(self, asset_path: Path) -> None:
            content_type = {
                ".css": "text/css; charset=utf-8",
                ".js": "application/javascript; charset=utf-8",
                ".html": "text/html; charset=utf-8",
            }.get(asset_path.suffix, "application/octet-stream")
            payload = asset_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, payload: dict[str, float | str]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _save_screenshot(self) -> None:
            content_length = int(self.headers.get("Content-Length", 0))
            payload = self.rfile.read(content_length)

            SCREENSHOTS_DIR.mkdir(exist_ok=True)
            SCREENSHOT_PATH.write_bytes(payload)

            self._send_json({"saved_to": str(SCREENSHOT_PATH)})

        def _classify_screenshot(self) -> None:
            nonlocal classifier

            if not SCREENSHOT_PATH.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Screenshot not found")
                return

            try:
                if classifier is None:
                    from land_cover_classifier import LandCoverClassifier

                    classifier = LandCoverClassifier()

                result = classifier.predict_image(SCREENSHOT_PATH)
            except Exception as error:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
                return

            self._send_json({
                "category": result["class"],
                "confidence": result["confidence"],
            })

    return MapRequestHandler


def start_map_server(config: MapServerConfig, shared_state: SharedMapState) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((config.host, config.port), build_map_handler(shared_state))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    return server


def open_map_browser(config: MapServerConfig) -> None:
    if not config.auto_open_browser:
        return
    webbrowser.open(f"http://{config.host}:{config.port}", new=1)
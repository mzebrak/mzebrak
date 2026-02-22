"""Record browser animation to GIF.

Prerequisites:
    sudo apt install xvfb ffmpeg

Setup (from repo root):
    cd banner/recorder && poetry install && poetry run playwright install chromium

Chromium is installed via Playwright (not apt) because it requires
a specific patched version compatible with the Playwright API.

Run (from repo root):
    cd banner/recorder && poetry run python record.py
"""

import asyncio
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from ffmpeg.asyncio import FFmpeg
from playwright.async_api import async_playwright
from xvfbwrapper import Xvfb

# --- Configuration ---

FPS = 50
DURATION = 13.5
LEAD_IN = 2.0  # seconds of recording before reload (ffmpeg startup buffer)
WIDTH = 822
HEIGHT = 140
CHROME_HEIGHT_BUDGET = 160  # maximum expected browser chrome height
XVFB_HEIGHT = HEIGHT + CHROME_HEIGHT_BUDGET
VIEWPORT_SETTLE = 0.3  # delay after viewport resize for layout stabilization
GIFSKI_QUALITY = 90
GIFSKI_VERSION = "1.34.0"
MAX_GIFSKI_BYTES = 20 * 1024 * 1024  # 20 MB safety limit for binary
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024  # 50 MB safety limit for archive download
DOWNLOAD_TIMEOUT = 30  # seconds
GIFSKI_TIMEOUT = 300  # seconds, generous limit for GIF conversion
VIDEO_READY_TIMEOUT = 8000  # ms, fallback if canplaythrough never fires

# Compensate for variable ffmpeg startup latency so the trim
# point falls just before the animation begins after page reload.
TRIM_BACKOFF = 0.3

RECORD_PADDING = 2.0  # extra seconds after animation to ensure full capture

RECORDER_DIR = Path(__file__).parent
BANNER_DIR = RECORDER_DIR.parent
OUTPUT_GIF = BANNER_DIR / "welcome.gif"
RAW_VIDEO = RECORDER_DIR / "raw_capture.mp4"
FRAMES_DIR = RECORDER_DIR / "frames_tmp"
HTML_FILE = BANNER_DIR / "index.html"
GIFSKI_DIR = RECORDER_DIR / ".gifski"
GIFSKI_BIN = GIFSKI_DIR / "gifski"
GIFSKI_URL = (
    f"https://github.com/ImageOptim/gifski/releases/download/"
    f"{GIFSKI_VERSION}/gifski-{GIFSKI_VERSION}.tar.xz"
)

CHROME_ARGS = [
    "--window-position=0,0",
    f"--window-size={WIDTH},{XVFB_HEIGHT}",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--autoplay-policy=no-user-gesture-required",
    "--disable-infobars",
    "--hide-scrollbars",
    "--disable-extensions",
    "--no-first-run",
    "--no-default-browser-check",
    "--force-device-scale-factor=1",
    "--use-angle=swiftshader",  # software rendering for headless Xvfb environment
]

# Resolves immediately if video is absent or already loaded;
# falls back to VIDEO_READY_TIMEOUT if canplaythrough never fires.
WAIT_FOR_VIDEO_JS = f"""() => new Promise(resolve => {{
    const video = document.querySelector("#background-video");
    if (!video || video.readyState >= 3) return resolve();
    video.addEventListener("canplaythrough", () => resolve(), {{ once: true }});
    setTimeout(() => resolve(), {VIDEO_READY_TIMEOUT});
}})"""


# --- Errors ---


class RecorderError(RuntimeError):
    """Raised when recording prerequisites or steps fail."""


# --- Helpers ---


def check_system_deps() -> None:
    """Verify required system tools are installed."""
    missing = [cmd for cmd in ("ffmpeg", "Xvfb") if shutil.which(cmd) is None]
    if missing:
        raise RecorderError(
            f"Missing system dependencies: {', '.join(missing)}. "
            "Install with: sudo apt install xvfb ffmpeg"
        )


def ensure_gifski() -> Path:
    """Download gifski binary from GitHub releases if not present."""
    if GIFSKI_BIN.exists():
        return GIFSKI_BIN

    print(f"Downloading gifski {GIFSKI_VERSION}...")
    GIFSKI_DIR.mkdir(parents=True, exist_ok=True)

    system = platform.system().lower()
    archive_subpath = {"linux": "linux/gifski", "darwin": "mac/gifski"}.get(system)
    if not archive_subpath:
        raise RecorderError(f"Unsupported platform: {system}")

    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / "gifski.tar.xz"
        try:
            with urllib.request.urlopen(GIFSKI_URL, timeout=DOWNLOAD_TIMEOUT) as resp:
                data = resp.read(MAX_ARCHIVE_BYTES + 1)
                if len(data) > MAX_ARCHIVE_BYTES:
                    raise RecorderError("gifski archive download exceeds size limit")
                archive.write_bytes(data)
        except (urllib.error.URLError, OSError) as exc:
            raise RecorderError(f"Failed to download gifski: {exc}") from exc

        with tarfile.open(archive, "r:xz") as tar:
            member = tar.extractfile(archive_subpath)
            if not member:
                raise RecorderError(f"{archive_subpath} not found in archive")
            data = member.read(MAX_GIFSKI_BYTES + 1)
            if len(data) > MAX_GIFSKI_BYTES:
                raise RecorderError(
                    f"gifski binary exceeds {MAX_GIFSKI_BYTES // 1024 // 1024} MB"
                )
            GIFSKI_BIN.write_bytes(data)
            GIFSKI_BIN.chmod(GIFSKI_BIN.stat().st_mode | stat.S_IEXEC)

    print(f"gifski installed at {GIFSKI_BIN}")
    return GIFSKI_BIN


async def record_screen(
    display: str, chrome_height: int, total_duration: float
) -> None:
    """Record content area of Xvfb screen to lossless intermediate video."""
    recorder = (
        FFmpeg()
        .option("y")
        .input(
            f"{display}+0,{chrome_height}",
            f="x11grab",
            video_size=f"{WIDTH}x{HEIGHT}",
            framerate=str(FPS),
            draw_mouse="0",
        )
        .output(
            str(RAW_VIDEO),
            vcodec="libx264",
            preset="ultrafast",
            qp="0",
            pix_fmt="yuv444p",
            t=str(total_duration),
        )
    )
    await recorder.execute()


async def extract_frames(trim_offset: float) -> list[Path]:
    """Extract PNG frames from raw video, skipping initial offset."""
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True)

    extractor = (
        FFmpeg()
        .option("y")
        .input(str(RAW_VIDEO), ss=str(trim_offset))
        .output(
            f"{FRAMES_DIR}/frame%04d.png",
            vf=f"fps={FPS}",
            t=str(DURATION),
        )
    )
    await extractor.execute()
    return sorted(FRAMES_DIR.glob("frame*.png"))


def convert_to_gif(gifski: Path, frames: list[Path]) -> None:
    """Convert PNG frames to GIF using gifski."""
    try:
        result = subprocess.run(
            [
                str(gifski),
                "--fps", str(FPS),
                "--quality", str(GIFSKI_QUALITY),
                "--width", str(WIDTH),
                "-o", str(OUTPUT_GIF),
                *(str(f) for f in frames),
            ],
            check=True,
            capture_output=True,
            timeout=GIFSKI_TIMEOUT,
        )
        if result.stderr:
            print(result.stderr.decode(errors="replace").strip())
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise RecorderError(f"gifski failed (exit {exc.returncode}): {stderr}") from exc
    except subprocess.TimeoutExpired:
        raise RecorderError(f"gifski timed out after {GIFSKI_TIMEOUT}s") from None


# --- Main flow ---


async def main() -> None:
    check_system_deps()
    gifski = ensure_gifski()

    trim_offset = LEAD_IN  # overwritten after reload; safe fallback for errors

    with Xvfb(width=WIDTH, height=XVFB_HEIGHT, colordepth=24) as xvfb:
        display = f":{xvfb.new_display}"

        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=False,
                args=CHROME_ARGS,
            )

            try:
                # no_viewport=True uses real Xvfb display; viewport is set
                # explicitly after measuring browser chrome height.
                page = await browser.new_page(no_viewport=True)

                print("Loading page...")
                await page.goto(
                    HTML_FILE.resolve().as_uri(), wait_until="networkidle"
                )

                chrome_height = await page.evaluate(
                    "window.outerHeight - window.innerHeight"
                )
                if not 0 < chrome_height < XVFB_HEIGHT - HEIGHT:
                    raise RecorderError(
                        f"Unexpected chrome height: {chrome_height}px "
                        f"(expected 1..{XVFB_HEIGHT - HEIGHT - 1})"
                    )
                print(f"Browser chrome: {chrome_height}px")

                await page.set_viewport_size({"width": WIDTH, "height": HEIGHT})
                await asyncio.sleep(VIEWPORT_SETTLE)
                await page.evaluate(WAIT_FOR_VIDEO_JS)
                print("Page ready.")

                total_duration = LEAD_IN + DURATION + RECORD_PADDING
                print(f"Recording {total_duration}s at {FPS}fps...")

                t_start = time.monotonic()
                record_task = asyncio.create_task(
                    record_screen(display, chrome_height, total_duration)
                )

                try:
                    await asyncio.sleep(LEAD_IN)
                    if record_task.done():
                        record_task.result()  # surface early ffmpeg failure

                    print("Reloading to sync animation...")
                    await page.reload(wait_until="networkidle")

                    trim_offset = max(0.0, time.monotonic() - t_start - TRIM_BACKOFF)
                    print(f"Trim offset: {trim_offset:.2f}s")

                    await record_task
                finally:
                    if not record_task.done():
                        record_task.cancel()
                        try:
                            await record_task
                        except asyncio.CancelledError:
                            pass

                print("Recording complete.")
            finally:
                await browser.close()

    try:
        print("Extracting frames...")
        frames = await extract_frames(trim_offset)
        if not frames:
            raise RecorderError("No frames extracted -- check trim offset and duration")

        print(f"Converting {len(frames)} frames to GIF...")
        convert_to_gif(gifski, frames)
    finally:
        shutil.rmtree(FRAMES_DIR, ignore_errors=True)
        RAW_VIDEO.unlink(missing_ok=True)

    size_mb = OUTPUT_GIF.stat().st_size / 1024 / 1024
    print(f"\nDone! {OUTPUT_GIF.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RecorderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

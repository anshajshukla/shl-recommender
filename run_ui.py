"""Launch the Streamlit UI with configurable options.

Environment variables / CLI flags:
    API_BASE_URL            Base URL for the FastAPI service (default: http://localhost:8000)
    STREAMLIT_HOST          Host interface for Streamlit (default: 0.0.0.0)
    STREAMLIT_PORT          Port for Streamlit (default: 8501)
    STREAMLIT_HEADLESS      Run Streamlit headless (true/false, default: true)

Usage examples:
    python run_ui.py
    python run_ui.py --api-url https://api.example.com --host 0.0.0.0 --port 8501
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def str_to_bool(value: Optional[str], default: bool = True) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Streamlit UI for the SHL Recommender.")
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_BASE_URL", "http://localhost:8000"),
        help="Base URL for the recommendation API (default: %(default)s)",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("STREAMLIT_HOST", "0.0.0.0"),
        help="Host interface for Streamlit (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", os.getenv("STREAMLIT_PORT", "8501"))),
        help="Port for Streamlit (default: %(default)s)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=str_to_bool(os.getenv("STREAMLIT_HEADLESS"), default=True),
        help="Run Streamlit in headless mode (default: true)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ui_file = Path(__file__).parent / "src" / "ui_simple.py"
    if not ui_file.exists():
        logger.error("UI file not found: %s", ui_file)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("Starting SHL Recommender Streamlit UI")
    logger.info("=" * 70)
    logger.info("Streamlit host: %s", args.host)
    logger.info("Streamlit port: %s", args.port)
    logger.info("API base URL: %s", args.api_url)
    logger.info("=" * 70)

    env = os.environ.copy()
    env["API_BASE_URL"] = args.api_url
    env["STREAMLIT_SERVER_ADDRESS"] = args.host
    env["STREAMLIT_SERVER_PORT"] = str(args.port)
    env["STREAMLIT_SERVER_HEADLESS"] = "true" if args.headless else "false"

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(ui_file),
                "--server.port", str(args.port),
                "--server.address", args.host,
                "--server.headless", "true" if args.headless else "false",
                "--server.enableCORS", "false",
                "--server.enableXsrfProtection", "false",
            ],
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        logger.error("Failed to start Streamlit UI: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("UI server stopped by user")

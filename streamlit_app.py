"""Entry point for Streamlit deployment.

Render and local developers can run `streamlit run streamlit_app.py`
to launch the production UI defined in `src/ui_simple.py`.
"""

from src.ui_simple import main


def run() -> None:
    """Run the production Streamlit UI."""
    main()


if __name__ == "__main__":
    run()


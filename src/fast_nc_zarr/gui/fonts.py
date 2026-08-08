from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

FONT_PATH = Path(__file__).resolve().parent / "assets" / "NotoSansSC-VF.ttf"


def configure_application_font(application: QApplication) -> str:
    """Load the bundled Simplified Chinese font and apply it globally."""
    if not FONT_PATH.is_file():
        raise RuntimeError(f"Bundled font resource is missing: {FONT_PATH}")

    font_id = QFontDatabase.addApplicationFont(str(FONT_PATH))
    if font_id < 0:
        raise RuntimeError(f"Failed to register bundled font resource: {FONT_PATH}")

    families = QFontDatabase.applicationFontFamilies(font_id)
    if not families:
        raise RuntimeError(
            f"Bundled font registered without an available family: {FONT_PATH}"
        )

    family = families[0]
    font = QFont(application.font())
    font.setFamily(family)
    application.setFont(font)
    return family

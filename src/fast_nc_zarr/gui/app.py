from __future__ import annotations

import faulthandler
import os
import platform
import sys

# WSLg may expose a DXG adapter that cannot complete Qt's graphics feature
# probes.  The application is a QWidget UI and gains nothing from GPU OpenGL;
# use the stable raster path on WSL before Qt loads its platform plugin.
if "microsoft" in platform.release().lower():
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
from PySide6.QtWidgets import QApplication

from .fonts import configure_application_font
from .theme import apply_theme
from .main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    faulthandler.enable(all_threads=True)
    application = QApplication(sys.argv if argv is None else argv)
    application.setApplicationName("快速 Zarr 转换器")
    application.setOrganizationName("fast-nc-zarr")
    configure_application_font(application)
    apply_theme(application)
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

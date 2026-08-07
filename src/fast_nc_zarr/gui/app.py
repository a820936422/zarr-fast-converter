from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    application = QApplication(sys.argv if argv is None else argv)
    application.setApplicationName("快速 Zarr 转换器")
    application.setOrganizationName("fast-nc-zarr")
    application.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

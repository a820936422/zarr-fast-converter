from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True, slots=True)
class ThemeTokens:
    """Semantic colors and dimensions shared by the desktop interface."""

    canvas: str = "#F1F5F9"
    surface: str = "#FFFFFF"
    surface_alt: str = "#E2E8F0"
    surface_dark: str = "#111827"
    text_primary: str = "#000000"
    text_secondary: str = "#111827"
    text_muted: str = "#334155"
    border: str = "#AAB7C4"
    border_strong: str = "#475569"

    primary: str = "#1D4ED8"
    primary_hover: str = "#1E40AF"
    primary_pressed: str = "#1E3A8A"
    focus: str = "#2563EB"
    success: str = "#15803D"
    success_bg: str = "#DCFCE7"
    warning: str = "#B45309"
    warning_bg: str = "#FEF3C7"
    danger: str = "#B91C1C"
    danger_bg: str = "#FEE2E2"
    accent: str = "#7C3AED"
    accent_bg: str = "#EDE9FE"
    info_bg: str = "#DBEAFE"


TOKENS = ThemeTokens()


def build_stylesheet(tokens: ThemeTokens = TOKENS) -> str:
    """Build the application-wide Fusion stylesheet from semantic tokens."""

    return f"""
    QWidget {{
        color: {tokens.text_primary};
        font-size: 14px;
        font-weight: 500;
    }}
    QMainWindow {{ background: {tokens.canvas}; }}
    QMenuBar {{
        background: {tokens.surface};
        color: {tokens.text_primary};
        border-bottom: 1px solid {tokens.border};
        font-weight: 600;
    }}
    QMenuBar::item:selected, QMenu::item:selected {{
        background: {tokens.info_bg};
        color: {tokens.text_primary};
    }}
    QMenu {{
        background: {tokens.surface};
        color: {tokens.text_primary};
        border: 1px solid {tokens.border};
        padding: 6px;
        font-weight: 500;
    }}
    QLabel, QCheckBox, QRadioButton {{
        color: #000000;
        font-weight: 700;
    }}
    QWidget:disabled, QLabel:disabled, QCheckBox:disabled, QRadioButton:disabled,
    QGroupBox:disabled, QGroupBox:disabled QLabel, QGroupBox:disabled QComboBox {{
        color: #000000;
        font-weight: 700;
    }}
    QLabel#pageTitle {{
        color: {tokens.text_primary};
        font-size: 20px;
        font-weight: 800;
        padding: 4px 0 2px;
    }}
    QLabel#pageSubtitle {{
        color: {tokens.text_primary};
        font-size: 14px;
        font-weight: 700;
        padding-bottom: 4px;
    }}
    QLabel#helperText {{
        color: #000000;
        font-size: 13px;
        font-weight: 700;
    }}
    QLabel#topContext {{
        color: {tokens.text_primary};
        background: {tokens.surface};
        border-bottom: 1px solid {tokens.border};
        padding: 10px 16px;
        font-weight: 700;
    }}
    QFrame#actionBar {{
        background: {tokens.surface};
        border-top: 1px solid {tokens.border};
    }}
    QLabel#actionHint {{ color: {tokens.text_primary}; font-size: 14px; font-weight: 700; }}
    QPushButton#primaryAction {{
        background: {tokens.primary};
        color: #FFFFFF;
        border: 1px solid {tokens.primary};
        font-weight: 700;
        min-width: 116px;
    }}
    QPushButton#primaryAction:hover {{ background: {tokens.primary_hover}; }}
    QPushButton#primaryAction:disabled {{
        background: {tokens.surface_alt};
        color: {tokens.text_muted};
        border-color: {tokens.border};
    }}
    QFrame#navPanel {{ background: {tokens.surface_dark}; }}
    QLabel#brandLabel {{
        color: #FFFFFF;
        font-size: 18px;
        font-weight: 700;
        padding: 8px 14px 14px;
    }}
    QLabel#navVersion {{ color: #CBD5E1; padding: 8px 14px 0; font-weight: 600; }}
    QGroupBox {{
        border: 1px solid {tokens.border};
        border-radius: 8px;
        margin-top: 12px;
        padding: 14px 10px 10px;
        background: {tokens.surface};
        color: #000000;
        font-weight: 700;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        color: #000000;
        font-size: 14px;
        font-weight: 800;
    }}
    QGroupBox#chooserPanel {{
        background: {tokens.surface};
        border-color: {tokens.border_strong};
    }}
    QGroupBox#chooserPanel QLabel {{
        color: #000000;
        font-weight: 700;
    }}
    QLabel#pathStatus, QLabel#statusBadge {{
        border-radius: 5px;
        padding: 4px 7px;
        font-size: 12px;
        font-weight: 700;
    }}
    QLabel#stepperItem {{
        border: 1px solid {tokens.border};
        border-radius: 6px;
        background: {tokens.surface};
        color: {tokens.text_primary};
        padding: 5px 8px;
        font-weight: 700;
    }}
    QLabel#stepperItem[stepState="current"] {{
        border-color: {tokens.focus};
        background: {tokens.info_bg};
        color: {tokens.primary};
        font-weight: 700;
    }}
    QLabel#stepperItem[stepState="complete"] {{
        border-color: #86EFAC;
        background: {tokens.success_bg};
        color: {tokens.success};
        font-weight: 700;
    }}
    QTextBrowser#summaryCard {{
        border: 1px solid {tokens.border};
        border-radius: 9px;
        background: {tokens.surface};
        color: {tokens.text_primary};
        padding: 12px;
    }}
    QListWidget {{
        background: {tokens.surface_dark};
        color: #F8FAFC;
        border: 0;
        padding: 14px 8px;
        outline: 0;
        font-weight: 600;
    }}
    QListWidget::item {{
        padding: 12px 14px;
        margin: 3px 0;
        border-radius: 7px;
        color: #F8FAFC;
        font-weight: 600;
    }}
    QListWidget::item:hover {{ background: #25364A; color: #FFFFFF; }}
    QListWidget::item:selected {{
        background: {tokens.primary};
        color: #FFFFFF;
        font-weight: 800;
    }}
    QListWidget::item:disabled {{ color: #FFFFFF; font-weight: 700; }}
    QListWidget#mainNavigation::item {{
        color: #FFFFFF;
        font-weight: 700;
    }}
    QListWidget#mainNavigation::item:disabled {{
        color: #FFFFFF;
        background: #182235;
        font-weight: 700;
    }}
    QListWidget#chooserLocations {{
        background: {tokens.surface};
        color: {tokens.text_primary};
        border: 1px solid {tokens.border};
        padding: 5px;
    }}
    QListWidget#chooserLocations::item {{
        color: {tokens.text_primary};
        padding: 9px 8px;
        margin: 1px 0;
        font-weight: 600;
    }}
    QListWidget#chooserLocations::item:hover {{
        background: {tokens.info_bg};
        color: {tokens.text_primary};
    }}
    QListWidget#chooserLocations::item:selected {{
        background: {tokens.primary};
        color: #FFFFFF;
        font-weight: 800;
    }}
    QListWidget#chooserLocations::item:disabled {{
        color: {tokens.text_secondary};
        font-weight: 800;
    }}
    QGroupBox#sectionCard, QGroupBox#summaryCard {{
        border: 1px solid {tokens.border};
        border-radius: 9px;
        margin-top: 14px;
        padding: 16px 12px 12px;
        background: {tokens.surface};
        color: {tokens.text_primary};
    }}
    QGroupBox#sectionCard::title, QGroupBox#summaryCard::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        color: {tokens.text_primary};
        font-weight: 800;
    }}
    QGroupBox#operationCard {{
        border: 1px solid {tokens.border};
        border-radius: 8px;
        margin-top: 10px;
        padding: 12px;
        background: {tokens.surface};
    }}
    QPushButton {{
        background: {tokens.primary};
        color: #FFFFFF;
        border: 1px solid {tokens.primary};
        border-radius: 6px;
        padding: 8px 16px;
        min-height: 18px;
        font-weight: 700;
    }}
    QPushButton:hover {{ background: {tokens.primary_hover}; border-color: {tokens.primary_hover}; }}
    QPushButton:pressed {{ background: {tokens.primary_pressed}; }}
    QPushButton:disabled {{
        background: {tokens.surface_alt};
        color: {tokens.text_secondary};
        border-color: {tokens.border};
        font-weight: 700;
    }}
    QPushButton#secondaryButton, QPushButton#pathPickerAuxButton {{
        background: {tokens.surface_alt};
        color: {tokens.text_primary};
        border: 1px solid {tokens.border_strong};
        font-weight: 700;
    }}
    QPushButton#secondaryButton:hover, QPushButton#pathPickerAuxButton:hover {{
        background: #D9E3EF;
        color: {tokens.text_primary};
    }}
    QPushButton#dangerButton {{
        background: {tokens.danger_bg};
        color: {tokens.danger};
        border-color: #FCA5A5;
    }}
    QPushButton#dangerButton:hover {{ background: #FECACA; }}
    QPushButton#pathPickerAuxButton:checked {{
        background: {tokens.info_bg};
        color: {tokens.primary};
        border-color: {tokens.focus};
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit,
    QPlainTextEdit, QTextEdit, QTextBrowser, QTableWidget {{
        border: 1px solid {tokens.border_strong};
        border-radius: 6px;
        background: {tokens.surface};
        color: {tokens.text_primary};
        padding: 6px;
        selection-background-color: {tokens.info_bg};
        selection-color: {tokens.text_primary};
        font-weight: 600;
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QDateEdit:focus, QPlainTextEdit:focus, QTextBrowser:focus, QTableWidget:focus,
    QPushButton:focus {{
        border: 2px solid {tokens.focus};
    }}
    QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,
    QDoubleSpinBox:disabled, QDateEdit:disabled {{
        background: {tokens.surface_alt};
        color: {tokens.text_primary};
        font-weight: 700;
    }}
    QHeaderView::section {{
        background: {tokens.surface_alt};
        color: {tokens.text_primary};
        border: 0;
        border-right: 1px solid {tokens.border};
        border-bottom: 1px solid {tokens.border};
        padding: 7px;
        font-weight: 800;
    }}
    QTableWidget {{ gridline-color: {tokens.border}; }}
    QToolTip {{
        color: {tokens.text_primary};
        background: #FFFDF2;
        border: 1px solid {tokens.border_strong};
        padding: 5px;
    }}
    QProgressBar {{
        border: 1px solid {tokens.border};
        border-radius: 6px;
        text-align: center;
        background: {tokens.surface};
        color: {tokens.text_primary};
        min-height: 18px;
    }}
    QProgressBar::chunk {{ background: {tokens.success}; border-radius: 5px; }}
    QSplitter::handle {{ background: {tokens.border}; }}
    QScrollArea {{ border: 0; background: transparent; }}
    QMenuBar {{
        background: {tokens.surface};
        color: {tokens.text_primary};
        font-weight: 700;
    }}
    QMenuBar::item {{
        background: transparent;
        color: {tokens.text_primary};
        padding: 4px 8px;
    }}
    QMenuBar::item:selected {{
        background: {tokens.info_bg};
        color: {tokens.text_primary};
    }}
    QMenu {{
        background: {tokens.surface};
        color: {tokens.text_primary};
        border: 1px solid {tokens.border};
        font-weight: 700;
    }}
    QMenu::item {{ padding: 6px 22px 6px 10px; color: {tokens.text_primary}; }}
    QMenu::item:selected {{ background: {tokens.info_bg}; color: {tokens.text_primary}; }}
    QStatusBar {{ background: {tokens.surface}; color: {tokens.text_primary}; font-weight: 600; }}
    QLabel#statusBadge[status="success"], QLabel#pathStatus[status="success"] {{ color: {tokens.success}; background: {tokens.success_bg}; }}
    QLabel#statusBadge[status="warning"], QLabel#pathStatus[status="warning"] {{ color: {tokens.warning}; background: {tokens.warning_bg}; }}
    QLabel#statusBadge[status="danger"], QLabel#pathStatus[status="danger"] {{ color: {tokens.danger}; background: {tokens.danger_bg}; }}
    QLabel#statusBadge[status="info"], QLabel#pathStatus[status="info"] {{ color: {tokens.primary}; background: {tokens.info_bg}; }}
    QLabel#statusBadge[status="neutral"], QLabel#pathStatus[status="neutral"] {{ color: {tokens.text_secondary}; background: {tokens.surface_alt}; }}
    QFrame#metricCard {{
        background: {tokens.surface};
        border: 1px solid {tokens.border};
        border-radius: 8px;
    }}
    QLabel#metricTitle {{ color: {tokens.text_muted}; font-size: 13px; font-weight: 700; }}
    QLabel#metricValue {{ color: {tokens.text_primary}; font-size: 17px; font-weight: 800; }}
    QLabel#metricDetail {{ color: {tokens.text_secondary}; font-size: 13px; font-weight: 600; }}
    """


def apply_theme(application: QApplication) -> None:
    """Apply the v1.6.8 visual system to a QApplication."""

    application.setStyle("Fusion")
    application.setStyleSheet(build_stylesheet())
    palette = application.palette()
    text_roles = (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.PlaceholderText,
    )
    for group in (
        QPalette.ColorGroup.Active,
        QPalette.ColorGroup.Inactive,
        QPalette.ColorGroup.Disabled,
    ):
        for role in text_roles:
            palette.setColor(group, role, QColor(TOKENS.text_primary))
    application.setPalette(palette)

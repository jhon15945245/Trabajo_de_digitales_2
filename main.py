"""Punto de entrada de la aplicación."""

import sys
import traceback

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from gui.main_window import MainWindow


def _install_exception_hook() -> None:
    """Muestra cualquier excepción no capturada en un diálogo (RNF1) sin cerrar la app."""
    def hook(exc_type, exc, tb):
        # Guarda el detalle del error y avisa al usuario sin cerrar la ventana.
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        sys.stderr.write(text)
        if QApplication.instance() is not None:
            QMessageBox.critical(None, "Error inesperado",
                                 f"{exc_type.__name__}: {exc}\n\nLa aplicación sigue activa.")
    sys.excepthook = hook


def main() -> None:
    # Mejora la presentación en pantallas de alta resolución.
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Crea la aplicación y define la información que muestra el sistema.
    app = QApplication(sys.argv)
    app.setApplicationName("Simulador de Comunicaciones Digitales")
    app.setOrganizationName("Universidad del Cauca - Telecomunicaciones")
    _install_exception_hook()

    # Muestra la ventana principal y mantiene la interfaz en funcionamiento.
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    # Ejecuta la aplicación cuando este archivo se inicia directamente.
    main()

"""
main.py
=======
Punto de entrada de la aplicación.

Uso:
    python main.py

Requisitos del sistema:
    Python >= 3.9
    PyQt5, matplotlib, numpy  (ver requirements.txt)
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from gui.main_window import MainWindow


def main():
    # Habilitar escala para pantallas de alta resolución (HiDPI)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Simulador de Comunicaciones Digitales")
    app.setOrganizationName("Universidad - Telecomunicaciones")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

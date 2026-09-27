#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Wärmebildkamera Installer ==="
echo ""

# ------------------------------------------------------------------ #
#  Python-Pakete installieren                                          #
# ------------------------------------------------------------------ #

echo "[1/2] Installiere Abhängigkeiten..."

sudo apt update
sudo apt install -y \
    python3-opencv \
    python3-pyside6.qtwidgets \
    python3-pyside6.qtcore \
    python3-pyside6.qtgui

echo "      Pakete installiert."
echo ""

# ------------------------------------------------------------------ #
#  .desktop-Datei erstellen                                            #
# ------------------------------------------------------------------ #

echo "[2/2] Erstelle Anwendungsstarter..."

DESKTOP_FILE="$HOME/.local/share/applications/t7.desktop"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Wärmebildkamera
Comment=Wärmebildkamera Anwendung
Exec=python3 $SCRIPT_DIR/t7.py
Icon=camera-on
Terminal=false
Categories=Graphics;
StartupNotify=true
EOF

chmod +x "$DESKTOP_FILE"

# Plasma Anwendungsmenü aktualisieren
update-desktop-database "$HOME/.local/share/applications/" 2>/dev/null || true
kbuildsycoca6 --noincremental 2>/dev/null || kbuildsycoca5 --noincremental 2>/dev/null || true

echo "      Anwendungsstarter erstellt: $DESKTOP_FILE"
echo ""
echo "=== Installation abgeschlossen! ==="
echo "    'Wärmebildkamera' ist jetzt im Startmenü unter 'Grafik' verfügbar."

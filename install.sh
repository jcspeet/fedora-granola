#!/usr/bin/env bash
# eatmo - System dependency installer for Fedora
set -e

echo "Installing system dependencies..."
sudo dnf install -y \
    python3-pip \
    python3-gobject \
    python3-gobject-devel \
    gtk4 \
    libadwaita \
    python3-devel \
    gcc \
    portaudio \
    portaudio-devel \
    pipewire \
    pipewire-pulseaudio \
    pipewire-utils \
    pulseaudio-utils \
    gobject-introspection-devel \
    cairo-gobject-devel \
    pkg-config

echo "Creating virtual environment..."
python3 -m venv --system-site-packages venv

echo "Installing Python dependencies..."
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

echo "Installing desktop launcher and icon..."
ICON_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"
DESKTOP_DIR="$HOME/.local/share/applications"
mkdir -p "$ICON_DIR" "$DESKTOP_DIR"
cp "$(dirname "$0")/eatmo.svg" "$ICON_DIR/eatmo.svg"
cp "$(dirname "$0")/eatmo.desktop" "$DESKTOP_DIR/eatmo.desktop"
gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true

echo ""
echo "Installation complete!"
echo ""
echo "eatmo should now appear in your application launcher."
echo "On first launch, open Settings and enter your API key."
echo ""
echo "Or add the API key to ~/.config/eatmo/config.env"

#!/usr/bin/env bash
# Fedora Granola - System dependency installer for Fedora
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

echo ""
echo "Installation complete!"
echo ""
echo "To run Fedora Granola:"
echo "  1. Set your Anthropic API key: export ANTHROPIC_API_KEY=your_key_here"
echo "  2. Run: venv/bin/python main.py"
echo ""
echo "Or add the API key to ~/.config/fedora-granola/config.env"

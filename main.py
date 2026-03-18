#!/usr/bin/env python3
"""
Fedora Granola — entry point
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from app import GranolaApp

if __name__ == "__main__":
    app = GranolaApp()
    sys.exit(app.run(sys.argv))

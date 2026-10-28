here#!/bin/sh
set -e
python tests/test_apexnet.py
python tests/test_extended.py
python tests/test_pro.py
python tests/test_hardware.py
python tests/test_interop.py
python tests/test_distributed.py
python -m apexnet.selftest --quick

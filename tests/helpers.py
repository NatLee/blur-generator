"""Helpers shared by the tests."""
import contextlib
import io
import os
import tempfile
from typing import NamedTuple

import cv2
import numpy as np

from blurgenerator.cli import main as cli_main


def random_image(seed=0, size=(40, 40)):
    """Noisy 3-channel test image without pure black pixels."""
    return np.random.RandomState(seed).randint(1, 255, size=size + (3,), dtype=np.uint8)


def gradient_depth_map(size=(40, 40)):
    """3-channel depth map that goes from 0 on the left to 255 on the right."""
    row = np.linspace(0, 255, size[1]).astype(np.uint8)
    gray = np.tile(row, (size[0], 1))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class CliResult(NamedTuple):
    code: int
    stdout: str
    stderr: str


def run_cli(*args):
    """Run the blurgenerator command in-process; return its exit code and what it printed."""
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            cli_main([str(arg) for arg in args])
        except SystemExit as exc:
            if isinstance(exc.code, str):
                # sys.exit('message') prints the message on stderr and exits with 1.
                err.write(exc.code + '\n')
                code = 1
            else:
                code = exc.code or 0
    return CliResult(code, out.getvalue(), err.getvalue())


class TempDirTestCase:
    """Mixin: ``self.tmp`` is a fresh folder; ``self.write_image(name, img)`` saves an image into it."""

    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp, name)

    def write_image(self, name, img):
        path = self.path(name)
        assert cv2.imwrite(path, img)
        return path

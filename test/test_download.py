import subprocess
import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from bonito.cli.download import models, training, Downloader


@mock.patch("sys.stdout", new=StringIO())
@mock.patch("sys.stderr", new=StringIO())
class TestDownload(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = TemporaryDirectory()
        self.save_path = Path(self._tmp_dir.name)

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_list(self):
        bonito_args = ["bonito", "download", "--list"]
        proc = subprocess.Popen(bonito_args + ["--models"], stderr=subprocess.PIPE)
        _, stderr = proc.communicate()
        self.assertTrue(len(stderr.decode().split('\n')) - 1, len(models))

        proc = subprocess.Popen(bonito_args + ["--training"], stderr=subprocess.PIPE)
        _, stderr = proc.communicate()
        self.assertTrue(len(stderr.decode().split('\n')) - 1, len(training))

        proc = subprocess.Popen(bonito_args + ["--all"], stderr=subprocess.PIPE)
        _, stderr = proc.communicate()
        self.assertTrue(len(stderr.decode().split('\n')) - 1, len(models) + len(training))

    def test_download(self):
        downloader = Downloader(out_dir=self.save_path)
        model_subset = models[:3]  # Pick the most recent models as an example
        for model in model_subset:
            downloader.download(model)
            self.assertTrue((self.save_path / model).exists())

        # Check we've cleaned up any other files
        assert len(list(self.save_path.iterdir())) == len(model_subset)

    def test_force(self):
        downloader = Downloader(out_dir=self.save_path, force=False)
        mock_out = self.save_path / "mock_output"
        mock_out.write_text("placeholder_text")

        # This should run without error despite the url not existing
        _ = downloader.download(mock_out.name)

        # This should fail to download from the CDN
        downloader.force = True
        with self.assertRaises(FileNotFoundError):
            _ = downloader.download(mock_out.name)

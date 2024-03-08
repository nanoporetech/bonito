import subprocess
import unittest

from bonito import modules

class TestCli(unittest.TestCase):
    """
    Minimal test case that each of the CLI entry points can be called from terminal
    """

    def test_tool_gets_help(self):
        for tool in modules:
            help_message = subprocess.check_output(["bonito", tool, "-h"])
            self.assertTrue(f"usage: bonito {tool}".encode() in help_message)

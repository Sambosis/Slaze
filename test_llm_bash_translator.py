import unittest
from unittest.mock import patch, mock_open
import platform as std_platform # Keep a reference to the original platform module

# Import functions to be tested
from llm_bash_translator import (
    get_llm_modified_command,
    parse_llm_response,
    translate_bash_command
)

class TestLLMBashTranslator(unittest.TestCase):

    def test_parse_llm_response(self):
        self.assertEqual(parse_llm_response("  echo hello  "), "echo hello")
        self.assertEqual(parse_llm_response("cmd"), "cmd")
        self.assertEqual(parse_llm_response(""), "")
        self.assertEqual(parse_llm_response(None), "") # Check None handling

    def test_get_llm_modified_command_inputs(self):
        with self.assertRaises(ValueError):
            get_llm_modified_command(None, {"os": "Linux"})
        with self.assertRaises(ValueError):
            get_llm_modified_command("", {"os": "Linux"}) # Empty string is now a ValueError
        with self.assertRaises(ValueError):
            get_llm_modified_command("ls", None)
        with self.assertRaises(ValueError):
            get_llm_modified_command("ls", {}) # Empty dict is now a ValueError

    def test_get_llm_modified_command_logic(self):
        # Test macOS simulation
        mac_info = {"os": "Darwin", "architecture": "arm64"}
        self.assertEqual(get_llm_modified_command("apt-get install xyz", mac_info), "brew install xyz")
        self.assertEqual(get_llm_modified_command("yum install xyz", mac_info), "brew install xyz")
        self.assertEqual(get_llm_modified_command("ls", mac_info), "ls") # No change rule

        # Test Linux simulation
        linux_info = {"os": "Linux", "architecture": "x86_64", "distro": "Ubuntu"}
        self.assertEqual(get_llm_modified_command("brew install xyz", linux_info), "apt-get install xyz")
        self.assertEqual(get_llm_modified_command("ls", linux_info), "ls")

        # Test Windows simulation
        windows_info = {"os": "Windows", "architecture": "AMD64"}
        self.assertEqual(get_llm_modified_command("ls", windows_info), "dir")
        self.assertEqual(get_llm_modified_command("cat file.txt", windows_info), "type file.txt")
        self.assertEqual(get_llm_modified_command("apt-get install xyz", windows_info), "apt-get install xyz") # No change

        # Test Unknown OS
        unknown_os_info = {"os": "AmigaOS", "architecture": "m68k"}
        self.assertEqual(get_llm_modified_command("apt-get install xyz", unknown_os_info), "apt-get install xyz")

    @patch('llm_bash_translator.platform')
    @patch('llm_bash_translator.open', new_callable=mock_open)
    def test_translate_bash_command_linux(self, mock_file_open, mock_platform):
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "x86_64"
        # Simulate /etc/os-release content
        mock_file_open.return_value.readlines.return_value = ['ID=ubuntu', 'VERSION_ID="20.04"']

        self.assertEqual(translate_bash_command("brew install git"), "apt-get install git")
        self.assertEqual(translate_bash_command("ls -l"), "ls -l") # No change

    @patch('llm_bash_translator.platform')
    def test_translate_bash_command_macos(self, mock_platform):
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "arm64"
        # No need to mock 'open' for macOS if /etc/os-release is only checked for Linux

        self.assertEqual(translate_bash_command("apt-get install git"), "brew install git")
        self.assertEqual(translate_bash_command("yum install git"), "brew install git")
        self.assertEqual(translate_bash_command("ls -l"), "ls -l")

    @patch('llm_bash_translator.platform')
    def test_translate_bash_command_windows(self, mock_platform):
        mock_platform.system.return_value = "Windows"
        mock_platform.machine.return_value = "AMD64"

        self.assertEqual(translate_bash_command("ls"), "dir")
        self.assertEqual(translate_bash_command("cat file.txt"), "type file.txt")
        self.assertEqual(translate_bash_command("git clone repo"), "git clone repo") # No change

    @patch('llm_bash_translator.platform')
    @patch('llm_bash_translator.open', new_callable=mock_open)
    def test_translate_bash_command_linux_no_os_release(self, mock_file_open, mock_platform):
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "x86_64"
        mock_file_open.side_effect = FileNotFoundError # Simulate /etc/os-release not found

        # Should still work, 'distro' will be 'N/A'
        self.assertEqual(translate_bash_command("brew install git"), "apt-get install git")


    def test_translate_bash_command_invalid_inputs(self):
        # Test None input
        self.assertEqual(translate_bash_command(None), "") # Handled by type check, returns ""
        # Test non-string input
        self.assertEqual(translate_bash_command(123), "123") # Handled by type check, returns str(input)
        # Test empty string
        self.assertEqual(translate_bash_command(""), "") # Returns as is
        # Test whitespace string
        self.assertEqual(translate_bash_command("   "), "   ") # Returns as is

    @patch('llm_bash_translator.get_llm_modified_command')
    @patch('llm_bash_translator.platform')
    def test_translate_bash_command_llm_call_exception(self, mock_platform, mock_get_llm_cmd):
        mock_platform.system.return_value = "Linux" # Any OS
        mock_platform.machine.return_value = "x86_64"
        mock_get_llm_cmd.side_effect = Exception("LLM API Down")

        # Should fallback to original command
        self.assertEqual(translate_bash_command("some command"), "some command")

    @patch('llm_bash_translator.platform')
    def test_translate_bash_command_platform_call_empty(self, mock_platform):
        # Test when platform.system() or platform.machine() return empty strings (or None)
        mock_platform.system.return_value = ""
        mock_platform.machine.return_value = ""

        # The system_info passed to get_llm_modified_command will have "Unknown" for os and arch
        # Our simple get_llm_modified_command will then likely return the original command
        # For "ls", if OS is "Unknown", it returns "ls".
        self.assertEqual(translate_bash_command("ls"), "ls")
        # For "apt-get install", if OS is "Unknown", it returns "apt-get install"
        self.assertEqual(translate_bash_command("apt-get install tool"), "apt-get install tool")


if __name__ == '__main__':
    unittest.main()

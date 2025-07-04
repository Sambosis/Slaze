import platform
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Placeholder for a real LLM API client
# For example, if using OpenAI:
# import openai
# openai.api_key = "YOUR_API_KEY"

# --- LLM Interaction ---
def get_llm_modified_command(original_command: str, system_info: dict) -> str:
    """
    Constructs a prompt and (simulates) sending it to an LLM
    to get a modified bash command suitable for the given system.
    """
    if not original_command or not isinstance(original_command, str):
        logging.error("Original command is empty or not a string: %s", original_command)
        raise ValueError("Original command must be a non-empty string.")
    if not system_info or not isinstance(system_info, dict):
        logging.error("System info is empty or not a dictionary: %s", system_info)
        raise ValueError("System info must be a non-empty dictionary.")

    prompt = f"""Given the following bash command:
'{original_command}'

And the following system information:
OS: {system_info.get('os', 'Unknown')}
Architecture: {system_info.get('architecture', 'Unknown')}
Distribution (if Linux): {system_info.get('distro', 'N/A')}

Please provide an equivalent command that will work on this system.
Return ONLY the modified command. Do not include any other text, explanations, or formatting.
If the command is already compatible, return the original command.
"""
    logging.info(f"Generated LLM prompt for command '{original_command}' on OS {system_info.get('os', 'Unknown')}")
    logging.debug(f"Full LLM prompt:\n{prompt}") # DEBUG level for potentially long prompt

    # --- SIMULATED LLM CALL ---
    # In a real scenario, you would make an API call to an LLM here.
    # This block should include error handling for the API call (e.g., network issues, API errors).
    try:
        # Example for a real API call:
        # response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=50, timeout=10)
        # llm_output = response.choices[0].text.strip()

        # Simulated logic:
        llm_output = ""
        os_name = system_info.get('os')

        if os_name == "Darwin": # macOS
            if "apt-get install" in original_command:
                llm_output = original_command.replace("apt-get install", "brew install")
            elif "yum install" in original_command:
                llm_output = original_command.replace("yum install", "brew install")
            else:
                llm_output = original_command
        elif os_name == "Linux":
            if "brew install" in original_command:
                llm_output = original_command.replace("brew install", "apt-get install") # Simplified
            else:
                llm_output = original_command
        elif os_name == "Windows":
            if original_command.strip() == "ls":
                llm_output = "dir"
            elif "cat " in original_command:
                llm_output = original_command.replace("cat ", "type ")
            else:
                llm_output = original_command
        else:
            llm_output = original_command
            logging.warning(f"No specific translation logic for OS '{os_name}', returning original command '{original_command}'.")

    except Exception as e: # Replace with more specific exceptions for a real API call
        logging.error(f"Error during simulated LLM call for command '{original_command}': {e}", exc_info=True)
        # Fallback: return the original command if the LLM call fails
        return original_command

    logging.info(f"Simulated LLM response for '{original_command}': '{llm_output}'")
    return llm_output

# --- Parsing Logic ---
def parse_llm_response(llm_response: str) -> str:
    """
    Parses the LLM's response to extract the modified bash command.
    Currently assumes the LLM returns only the command.
    """
    if llm_response is None:
        logging.warning("LLM response was None, returning empty string.")
        return ""

    modified_command = llm_response.strip()
    logging.info(f"Parsed command from LLM response: '{modified_command}'")
    return modified_command

# --- Wrapper Function ---
def translate_bash_command(original_command: str) -> str:
    """
    Translates a bash command for the current operating system using an LLM (simulated).
    """
    logging.info(f"Attempting to translate command for current system: '{original_command}'")

    if original_command is None:
        logging.error(f"Invalid input: Original command is None. Returning empty string.")
        return ""
    if not isinstance(original_command, str):
        logging.error(f"Invalid input type: Original command must be a string, got {type(original_command)}. Converting to string.")
        try:
            original_command = str(original_command)
        except Exception as e:
            logging.error(f"Could not convert original_command to string: {e}. Returning empty string.", exc_info=True)
            return "" # Failsafe if str() conversion itself fails for some reason

    if not original_command.strip(): # Handle empty or whitespace-only strings
        logging.info("Original command is empty or whitespace after potential conversion. Returning as is.")
        return original_command

    try:
        system_name = platform.system()
        architecture = platform.machine()
        distro = "N/A" # Default

        if not system_name:
            logging.warning("Could not determine OS type (platform.system() returned empty). Defaulting OS to 'Unknown'.")
            system_name = "Unknown"
        if not architecture:
            logging.warning("Could not determine architecture (platform.machine() returned empty). Defaulting to 'Unknown'.")
            architecture = "Unknown"

        if system_name == "Linux":
            try:
                # Attempt to read /etc/os-release for distro info
                with open("/etc/os-release") as f:
                    lines = f.readlines()
                os_release = {}
                for line in lines:
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        os_release[key] = value.strip('"').strip("'") # Remove potential quotes
                distro = os_release.get('ID', 'Unknown')
            except FileNotFoundError:
                logging.info("'/etc/os-release' not found. Distro information will be 'N/A'.")
            except Exception as e:
                logging.warning(f"Could not parse /etc/os-release: {e}. Distro info will be 'N/A'.", exc_info=True)

        current_system_info = {
            "os": system_name,
            "architecture": architecture,
            "distro": distro
        }
        logging.info(f"Current system info: {current_system_info}")

        llm_response = get_llm_modified_command(original_command, current_system_info)
        modified_command = parse_llm_response(llm_response)

        logging.info(f"Translation for '{original_command}' complete. Modified: '{modified_command}'")
        return modified_command

    except ValueError as ve: # Catch specific ValueErrors from get_llm_modified_command
        logging.error(f"ValueError during translation process for '{original_command}': {ve}", exc_info=True)
        return original_command # Return original on known input validation errors
    except Exception as e:
        logging.error(f"Unexpected error during command translation process for '{original_command}': {e}", exc_info=True)
        # Fallback: return the original command if translation fails unexpectedly
        return original_command

# --- Main Execution for Testing ---
if __name__ == '__main__':
    # To see debug logs:
    # logging.getLogger().setLevel(logging.DEBUG)

    test_commands_for_current_system = [
        "apt-get install python3",
        "ls -la",
        "cat myfile.txt",
        "brew install tree",
        "yum install mc",
        "",                        # Test empty command
        "   ",                     # Test whitespace only command
        "unknown_command_for_llm"  # Test command with no specific rule
    ]

    print("\n--- Testing command translation on current system ---")
    for command in test_commands_for_current_system:
        print(f"  Original: '{command}'")
        translated_cmd = translate_bash_command(command)
        print(f"  Translated for current system: '{translated_cmd}'")

    # Test None input (should be handled by translate_bash_command type check)
    print(f"\n  Original: None")
    translated_cmd_none = translate_bash_command(None) # type: ignore
    print(f"  Translated for current system (from None): '{translated_cmd_none}'")

    print(f"\n  Original: 123 (int)")
    translated_cmd_int = translate_bash_command(123) # type: ignore
    print(f"  Translated for current system (from int): '{translated_cmd_int}'")


    # Example with a specific simulated system (macOS)
    print("\n--- Simulating translation for macOS ---")
    mac_os_info = {"os": "Darwin", "architecture": "arm64"}
    for command in test_commands_for_current_system:
        if not command.strip(): # Skip empty or whitespace for this direct test part
            continue
        print(f"  Original: '{command}'")
        llm_response_mac = get_llm_modified_command(command, mac_os_info)
        translated_cmd_mac = parse_llm_response(llm_response_mac)
        print(f"  Translated for macOS: '{translated_cmd_mac}'")

    # Example with a specific simulated system (Windows)
    print("\n--- Simulating translation for Windows ---")
    windows_os_info = {"os": "Windows", "architecture": "AMD64"}
    for command in test_commands_for_current_system:
        if not command.strip(): # Skip empty or whitespace for this direct test part
            continue
        print(f"  Original: '{command}'")
        llm_response_win = get_llm_modified_command(command, windows_os_info)
        translated_cmd_win = parse_llm_response(llm_response_win)
        print(f"  Translated for Windows: '{translated_cmd_win}'")

    # Test with invalid system_info for get_llm_modified_command
    print("\n--- Testing get_llm_modified_command with invalid system_info ---")
    try:
        get_llm_modified_command("ls", None) # type: ignore
    except ValueError as e:
        print(f"  Caught expected error for None system_info: {e}")
    try:
        get_llm_modified_command("ls", {}) # system_info.get('os') will be None
        # This will now log a warning in get_llm_modified_command for 'Unknown' OS
        # and proceed, rather than raising a ValueError directly if keys are missing
        # but the dict itself is provided. The ValueError is for `system_info` being None or not a dict.
        print("  Call with empty dict system_info (check logs for warnings).")
    except ValueError as e:
        print(f"  Caught unexpected error for empty dict system_info: {e}")

    print("\n--- Testing get_llm_modified_command with invalid original_command ---")
    try:
        get_llm_modified_command(None, {"os": "Linux"}) # type: ignore
    except ValueError as e:
        print(f"  Caught expected error for None original_command: {e}")
    try:
        get_llm_modified_command("", {"os": "Linux"})
    except ValueError as e: # This is now allowed, but will be returned as is by translate_bash_command
        print(f"  Caught expected error for empty original_command: {e}")


    logging.info("End of __main__ test execution.")

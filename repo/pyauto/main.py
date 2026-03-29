import os
from selenium.webdriver.remote.webdriver import WebDriver
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import subprocess
import pyautogui
import time

GOOGLE_URL: str = 'https://www.google.com'
FILENAME: str = 'googlesource.html'


def setup_chrome_driver() -> WebDriver:
    """
    Initializes and returns a Chrome WebDriver instance using webdriver-manager.
    """
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    print('Browser opened...')
    return driver


def navigate_and_copy_source(driver: WebDriver) -> None:
    """
    Navigates to Google, sends Ctrl+U, switches tabs, selects and copies source to clipboard.
    """
    driver.get(GOOGLE_URL)
    print('Navigated to site.')
    time.sleep(2)
    body = driver.find_element(By.TAG_NAME, 'body')
    body.send_keys(Keys.CONTROL + 'u')
    print('Source tab opened.')
    time.sleep(3)
    driver.switch_to.window(driver.window_handles[-1])
    time.sleep(2)
    body = driver.find_element(By.TAG_NAME, 'body')
    body.send_keys(Keys.CONTROL + 'a')
    body.send_keys(Keys.CONTROL + 'c')
    print('Source copied to clipboard.')
    driver.quit()


def open_notepad_paste_and_save() -> None:
    """
    Launches Notepad, pastes clipboard content, saves as specified filename using keyboard shortcuts.
    """
    full_path = os.path.join(os.getcwd(), FILENAME)
    subprocess.Popen(['notepad.exe'])
    time.sleep(3)
    print('Notepad opened')
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(1)
    print('Pasted content')
    pyautogui.hotkey('ctrl', 's')
    time.sleep(1)
    print('Saving file...')
    pyautogui.write(full_path)
    pyautogui.press('enter')


def main() -> None:
    """
    Orchestrates the entire automation sequence: browser actions, copy, Notepad launch, paste, and save.
    """
    print('Starting automation...')
    driver = setup_chrome_driver()
    navigate_and_copy_source(driver)
    print('Source copied...')
    open_notepad_paste_and_save()
    print('File saved.')


if __name__ == '__main__':
    main()
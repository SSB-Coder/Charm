"""
Project:Charm — Mouse Controller
"""

import logging

import pyautogui

import config

logger = logging.getLogger(__name__)

pyautogui.PAUSE = config.PYAUTOGUI_PAUSE
pyautogui.FAILSAFE = config.PYAUTOGUI_FAILSAFE


class MouseController:
    """Abstraction over PyAutoGUI for mouse and keyboard control."""

    def __init__(self) -> None:
        logger.info(
            "MouseController initialized (PAUSE=%.2f, FAILSAFE=%s)",
            pyautogui.PAUSE, pyautogui.FAILSAFE,
        )

    def move(self, x: int, y: int) -> None:
        try:
            pyautogui.moveTo(x, y, duration=0)
        except pyautogui.PyAutoGUIException as exc:
            logger.warning("Mouse move failed: %s", exc)

    def left_click(self) -> None:
        try:
            pyautogui.click(button="left")
            logger.debug("Left click")
        except pyautogui.PyAutoGUIException as exc:
            logger.warning("Left click failed: %s", exc)

    def right_click(self) -> None:
        try:
            pyautogui.click(button="right")
            logger.debug("Right click")
        except pyautogui.PyAutoGUIException as exc:
            logger.warning("Right click failed: %s", exc)

    def middle_click(self) -> None:
        try:
            pyautogui.click(button="middle")
            logger.debug("Middle click")
        except pyautogui.PyAutoGUIException as exc:
            logger.warning("Middle click failed: %s", exc)

    def double_click(self) -> None:
        try:
            pyautogui.doubleClick(button="left")
            logger.debug("Double left-click")
        except pyautogui.PyAutoGUIException as exc:
            logger.warning("Double left-click failed: %s", exc)

    def copy(self) -> None:
        try:
            pyautogui.hotkey("ctrl", "c")
            logger.debug("Copy (Ctrl+C)")
        except pyautogui.PyAutoGUIException as exc:
            logger.warning("Copy failed: %s", exc)

    def paste(self) -> None:
        try:
            pyautogui.hotkey("ctrl", "v")
            logger.debug("Paste (Ctrl+V)")
        except pyautogui.PyAutoGUIException as exc:
            logger.warning("Paste failed: %s", exc)

    def undo(self) -> None:
        try:
            pyautogui.hotkey("ctrl", "z")
            logger.debug("Undo (Ctrl+Z)")
        except pyautogui.PyAutoGUIException as exc:
            logger.warning("Undo failed: %s", exc)

    def scroll(self, amount: int) -> None:
        try:
            pyautogui.scroll(amount)
            logger.debug("Scroll %d", amount)
        except pyautogui.PyAutoGUIException as exc:
            logger.warning("Scroll failed: %s", exc)

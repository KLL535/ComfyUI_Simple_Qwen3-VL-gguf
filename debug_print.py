# debug_print.py
import os
import sys
import time

try:
    from colorama import just_fix_windows_console, Fore, Back, Style
    if os.name == "nt":
        just_fix_windows_console()
except ImportError:
    class Fore:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ''
        RESET = ''
    class Back:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ''
        RESET = ''
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ''

def _debug_print(debug, stage, start_time, message = "", file = sys.stdout):
    if not debug:
        return

    elapsed = time.perf_counter() - start_time
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} {stage}: {Fore.YELLOW}{elapsed:.3f}s{Style.RESET_ALL} {message}", file=file)


def _debug_result(debug, stage, start_time, text, file = sys.stdout):
    if not debug:
        return

    speed_info = ""
    if text:
        elapsed = time.perf_counter() - start_time
        words = len(text.strip().split())
        if elapsed > 0:
            speed_info = f"| {words} word ({Fore.GREEN}{words / elapsed:.1f} word/sec{Style.RESET_ALL})"
    _debug_print(debug, stage, start_time, speed_info, file=file)

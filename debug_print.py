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

def _debug_print(debug, stage, start_time, message = "", text = "", file = sys.stdout):
    if not debug:
        return

    elapsed = time.perf_counter() - start_time
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} {stage} {Style.BRIGHT}{Fore.GREEN}{text}{Style.RESET_ALL}: {Fore.YELLOW}{elapsed:.3f}s{Style.RESET_ALL} {message}", file=file)



# debug_print.py
import os
import sys
import time

class C:
    MAGENTA = "\033[95m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BRIGHT = "\033[1m"
    RESET = "\033[0m"

def _debug_print(debug, stage, start_time, message="", text="", file=sys.stdout):
    if not debug:
        return
    elapsed = time.perf_counter() - start_time
    # Пишем напрямую, без обертки colorama
    out = f"{C.MAGENTA}[DEBUG]{C.RESET} {stage} {C.BRIGHT}{C.GREEN}{text}{C.RESET}: {C.YELLOW}{elapsed:.3f}s{C.RESET} {message}"
    print(out, file=file)

def _debug_info(debug, stage, text="", file=sys.stdout):
    if not debug:
        return
    out = f"{C.MAGENTA}[DEBUG]{C.RESET} {stage}: {C.BRIGHT}{C.GREEN}{text}{C.RESET}"
    print(out, file=file)



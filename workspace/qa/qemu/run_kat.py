#!/usr/bin/env python3
"""Run one KAT image on one board profile.

Exit codes: 0 = the target exited with 0 (pass); non-zero = fail/timeout.
Board profiles live in ./boards/*.json and carry the machine, the QEMU binary,
the load kind (raw bin for M-profile, ELF elsewhere) and the exit protocol.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BOARDS = HERE / "boards"


def load_board(name):
    path = BOARDS / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"run_kat: unknown board '{name}' (no {path})")
    return json.loads(path.read_text(encoding="utf-8"))


def qemu_argv(board, image, extra=()):
    if board.get("mode") == "userspace":
        # User-mode emulation runs the ELF directly: no machine, no loader, no
        # semihosting. The process exit status is the verdict, same as on the host.
        #
        # 用户态模拟直接执行 ELF：没有 machine/loader/半主机；进程退出码即判定，与宿主一致。
        # `extra` carries the board's own additions: qemu-user needs `-cpu` to turn on an
        # extension the host profile does not model by default (RVV on rv64-virt is exactly
        # that case, measured: the default `-cpu rv64` has misa.V clear and traps every vector
        # instruction).
        #
        # `extra` 带的是板型自己的附加项：qemu-user 需要 `-cpu` 才能打开宿主 profile 默认不建模的
        # 扩展（rv64-virt 的 RVV 正是这种情况，实测：默认 `-cpu rv64` 的 misa.V 为 0，每条向量
        # 指令都陷入）。
        return [board["qemu"], *extra, image]
    argv = [board["qemu"], "-M", board["machine"], "-nographic", "-monitor", "none"]
    argv += [*extra]
    if board.get("serial_none", True):
        argv += ["-serial", "none"]
    if board["load"] == "raw":
        argv += ["-kernel", image]
    elif board["load"] == "loader":
        # Raw image placed by address: -kernel for these machines always loads at
        # the model's own base, so a board whose reset vector is elsewhere (AN505
        # fetches from 0x10000000) must use the generic loader instead.
        argv += ["-device", f"loader,file={image},addr={board['load_addr']},force-raw=on"]
    else:
        argv += ["-bios", "none", "-kernel", image]
    if board["exit"] == "semihost-sys-exit":
        argv += ["-semihosting-config", "enable=on,target=native"]
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("board", help="board profile name (file stem in ./boards)")
    parser.add_argument("image", help="path to the .bin (raw boards) or .elf image")
    parser.add_argument("label", nargs="?", default=None,
                        help="display name for the run (board:variant); defaults to the board id")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--qemu-arg", action="append", default=[], metavar="ARG",
                        help="extra argument for qemu (repeatable), for a capability the "
                             "board profile does not model by default")
    parser.add_argument("--tier", default=None,
                        help="the kernel tier this image compares (k3); printed in the verdict so "
                             "a pass says which tier passed")
    args = parser.parse_args()

    board = load_board(args.board)
    image = Path(args.image)
    if not image.is_file():
        raise SystemExit(f"run_kat: image not found: {image}")

    label = args.label or board["id"]
    argv = qemu_argv(board, str(image), args.qemu_arg)
    if args.verbose:
        print(f"[run_kat] {label}: {' '.join(argv)}", flush=True)
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=board.get("timeout_s", 20))
    except subprocess.TimeoutExpired:
        print(f"[run_kat] FAIL {label}: timeout after {board.get('timeout_s', 20)} s", file=sys.stderr)
        return 1
    if done.returncode == 0:
        print(f"[run_kat] PASS {label}")
        return 0
    print(f"[run_kat] FAIL {label}: qemu exit {done.returncode}", file=sys.stderr)
    for stream, text in (("stdout", done.stdout), ("stderr", done.stderr)):
        tail = "\n".join(text.strip().splitlines()[-5:])
        if tail:
            print(f"[run_kat] {stream} tail:\n{tail}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

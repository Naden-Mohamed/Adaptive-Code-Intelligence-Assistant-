from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from core.config import get_settings
from core.schemas import ExecutionResult

logger = logging.getLogger(__name__)

# Best-effort static blocklist -- defense-in-depth, NOT a full sandbox guarantee.
# See README "Known Limitations": use a container-based sandbox for untrusted,
# multi-tenant, production use.
BLOCKED_PATTERNS = (
    "import os",
    "import socket",
    "import shutil",
    "import subprocess",
    "__import__",
    "eval(",
    "exec(",
    "open(",
    "ctypes",
    "importlib",
)


class SandboxViolation(Exception):
    pass


def _preflight_check(code: str) -> None:
    lowered = code.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in lowered:
            raise SandboxViolation(f"Blocked pattern detected: '{pattern}'")


# def _limit_resources(memory_mb: int):
#     """
#     Applied in the child process via subprocess `preexec_fn` (POSIX only).

#     Which RLIMIT_* constants exist -- and whether the OS actually lets you
#     set them -- varies by platform. RLIMIT_NPROC in particular is commonly
#     missing or unsupported on macOS even though it's standard on Linux, and
#     some container/sandbox environments restrict RLIMIT_AS too. Each limit
#     is therefore applied independently and best-effort: an unsupported or
#     unsettable limit is skipped and logged rather than crashing the whole
#     sandboxed execution.
#     """

#     def _apply():
#         try:
#             import resource
#         except ImportError:
#             # resource module not available (e.g., on Windows)
#             return

#         mem_bytes = memory_mb * 1024 * 1024
#         desired_limits = (
#             ("RLIMIT_AS", (mem_bytes, mem_bytes)),
#             ("RLIMIT_NPROC", (32, 32)),
#             ("RLIMIT_FSIZE", (1024 * 1024, 1024 * 1024)),  # 1MB max file writes
#         )

#         for limit_name, values in desired_limits:
#             limit_const = getattr(resource, limit_name, None)
#             if limit_const is None:
#                 # Not defined on this platform (e.g. RLIMIT_NPROC on macOS) -- skip.
#                 continue
#             try:
#                 resource.setrlimit(limit_const, values)
#             except (ValueError, OSError, AttributeError):
#                 # Defined but not settable in this environment -- skip rather than fail.
#                 continue

#     return _apply


def run_code(code: str, language: str = "python") -> ExecutionResult:
    settings = get_settings()

    if language.lower() != "python":
        return ExecutionResult(success=False, stderr=f"Execution not supported for language: {language}")

    try:
        _preflight_check(code)
    except SandboxViolation as exc:
        return ExecutionResult(success=False, stderr=str(exc))

    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / "generated_solution.py"
        script_path.write_text(code, encoding="utf-8")

        start = time.perf_counter()
        try:
            preexec = _limit_resources(settings.sandbox_memory_limit_mb) if sys.platform != "win32" else None
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=settings.sandbox_timeout_seconds,
                preexec_fn=preexec,
                env={"PATH": "/usr/bin:/bin"},  # minimal env, no inherited secrets
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ExecutionResult(
                success=proc.returncode == 0,
                stdout=proc.stdout[-5000:],
                stderr=proc.stderr[-5000:],
                execution_time_ms=elapsed_ms,
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return ExecutionResult(
                success=False,
                stderr=f"Execution timed out after {settings.sandbox_timeout_seconds}s",
                execution_time_ms=elapsed_ms,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Sandbox execution failed: %s", exc)
            return ExecutionResult(success=False, stderr=str(exc))
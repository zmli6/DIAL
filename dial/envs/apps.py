"""
APPS Environment Adapter
=========================
Wraps the APPS (Automated Programming Progress Standard) dataset
as an interactive code-generation environment for DIAL.

Uses the 'introductory' difficulty subset — expected base SR ~30-60%
on Qwen3-4B, which avoids the ceiling effect seen in MBPP/HumanEval.

Each episode = one APPS problem:
  Step 0: LLM generates initial code (temperature=0)
  Step 1+: LLM sees test failures, modifies code
  ...repeat until all tests pass or max_steps

Rollout: K-variant generation (temperature=0.7, K=5)
Utility: max(variant pass_rate) - base pass_rate

Interface: follows BaseEnv exactly (same as MBPPEnv).
  - reset(seed=None) → (obs_text, info)
  - step(action: int) → (obs, reward, terminated, truncated, info)
  - get_state() / set_state() for snapshot/restore
  - get_actions() returns discrete action indices
  - get_text_description() for LLM proposer

Performance:
  IO-style tests are batched into a single subprocess (all test cases
  run inside one Python process) to avoid per-test subprocess overhead
  (~100ms each). Max test cases capped at MAX_IO_TESTS to avoid
  pathological 250-test problems.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import traceback
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseEnv, EnvState

logger = logging.getLogger("DIAL")

# Maximum IO test cases per problem to cap pathological cases
MAX_IO_TESTS = 25


# ── Safe code execution ─────────────────────────────────────────

def _safe_exec(code: str, test_code: str, timeout: int = 10) -> Dict:
    """
    Run generated code + test assertions in a subprocess.

    Follows the same pattern as MBPP's _safe_exec.
    Returns:
        {
            "passed": bool,       # all tests passed?
            "pass_rate": float,
            "num_passed": int,
            "num_total": int,
            "error": str | None,
            "error_type": str | None,
        }
    """
    full_code = f"{code}\n\n{test_code}"

    try:
        proc = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return {
                "passed": True,
                "pass_rate": 1.0,
                "num_passed": 1,
                "num_total": 1,
                "error": None,
                "error_type": None,
            }
        else:
            err = proc.stderr.strip()[:500]
            # Try to detect partial pass from assertion errors
            return {
                "passed": False,
                "pass_rate": 0.0,
                "num_passed": 0,
                "num_total": 1,
                "error": err,
                "error_type": _classify_error(err),
            }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "pass_rate": 0.0,
            "num_passed": 0,
            "num_total": 1,
            "error": f"TimeoutError ({timeout}s)",
            "error_type": "TimeoutError",
        }
    except Exception as e:
        return {
            "passed": False,
            "pass_rate": 0.0,
            "num_passed": 0,
            "num_total": 1,
            "error": str(e)[:500],
            "error_type": type(e).__name__,
        }


def _safe_exec_io(code: str, test_cases: List[Dict],
                  timeout: int = 10) -> Dict:
    """
    Run code against stdin/stdout test cases (APPS IO style).

    **Optimised**: all test cases run inside a single subprocess via a
    harness script.  This avoids per-test subprocess startup overhead
    (~100-150 ms each), giving ~10-50× speed-up on typical problems.

    Test cases are capped at MAX_IO_TESTS to avoid pathological problems
    (some APPS introductory problems have 250 tests).
    """
    # Cap test count
    capped = test_cases[:MAX_IO_TESTS]
    num_total = len(capped)
    if num_total == 0:
        return {"passed": False, "pass_rate": 0.0, "num_passed": 0,
                "num_total": 0, "error": "no test cases", "error_type": None}

    # Total timeout: per_test × num_tests, but at least 10s
    total_timeout = max(timeout * num_total, 10)

    # Build a harness that runs each test inside the same process
    # using exec() and StringIO to capture stdin/stdout.
    test_data_json = json.dumps([
        {"input": tc.get("input", ""), "output": tc.get("output", "").strip()}
        for tc in capped
    ])

    # The harness injects user code via exec(), redirects stdin/stdout
    # per test, and prints a JSON summary at the end.
    harness = textwrap.dedent(f'''\
import sys, io, json

_CODE = {repr(code)}
_TESTS = json.loads({repr(test_data_json)})
_TIMEOUT = {timeout}

results = []
for _i, _tc in enumerate(_TESTS):
    _old_stdin, _old_stdout = sys.stdin, sys.stdout
    sys.stdin = io.StringIO(_tc["input"])
    sys.stdout = io.StringIO()
    _ok = False
    _err = ""
    try:
        exec(compile(_CODE, "<solution>", "exec"), {{}})
        _actual = sys.stdout.getvalue().strip()
        _expected = _tc["output"]
        _ok = (_actual == _expected)
        if not _ok:
            _err = f"Expected {{repr(_expected[:60])}} got {{repr(_actual[:60])}}"
    except Exception as _e:
        _err = f"{{type(_e).__name__}}: {{str(_e)[:200]}}"
    finally:
        sys.stdin, sys.stdout = _old_stdin, _old_stdout
    results.append({{"ok": _ok, "err": _err}})

# Print JSON on original stdout
print(json.dumps(results))
''')

    try:
        proc = subprocess.run(
            [sys.executable, "-c", harness],
            capture_output=True,
            text=True,
            timeout=total_timeout,
        )

        if proc.returncode != 0 and not proc.stdout.strip():
            # Total failure (syntax error in user code, etc.)
            err = proc.stderr.strip()[:500]
            return {
                "passed": False,
                "pass_rate": 0.0,
                "num_passed": 0,
                "num_total": num_total,
                "error": err,
                "error_type": _classify_error(err),
            }

        # Parse results JSON
        try:
            results = json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            err = proc.stderr.strip()[:500] or proc.stdout.strip()[:200]
            return {
                "passed": False,
                "pass_rate": 0.0,
                "num_passed": 0,
                "num_total": num_total,
                "error": f"harness output parse error: {err}",
                "error_type": "HarnessError",
            }

        num_passed = sum(1 for r in results if r["ok"])
        errors = [f"Test {i}: {r['err']}" for i, r in enumerate(results) if not r["ok"]]
        pass_rate = num_passed / num_total if num_total > 0 else 0.0

        return {
            "passed": num_passed == num_total,
            "pass_rate": pass_rate,
            "num_passed": num_passed,
            "num_total": num_total,
            "error": "; ".join(errors[:3]) if errors else None,
            "error_type": None,
        }

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "pass_rate": 0.0,
            "num_passed": 0,
            "num_total": num_total,
            "error": f"TimeoutError (total {total_timeout}s)",
            "error_type": "TimeoutError",
        }
    except Exception as e:
        return {
            "passed": False,
            "pass_rate": 0.0,
            "num_passed": 0,
            "num_total": num_total,
            "error": str(e)[:500],
            "error_type": type(e).__name__,
        }


def _classify_error(err_text: str) -> str:
    """Classify error type from stderr text."""
    for etype in ["SyntaxError", "IndentationError", "NameError",
                  "TypeError", "ValueError", "IndexError", "KeyError",
                  "AttributeError", "RuntimeError", "AssertionError"]:
        if etype in err_text:
            return etype
    return "UnknownError"


# ── APPS Environment ────────────────────────────────────────────

class APPSEnv(BaseEnv):
    """
    APPS coding environment for DIAL.

    Follows the same interface as MBPPEnv:
      - Discrete action space (pre-generated candidate codes)
      - Snapshot/restore for rollout evaluation
      - Same reset/step signatures as BaseEnv

    Each episode = one coding problem from the APPS dataset.
    Agent generates/refines code over multiple steps.
    Rollout = K-variant code generation with temperature sampling.
    """

    ENV_TYPE = "apps"

    def __init__(self, env_name: str, **kwargs):
        super().__init__(env_name, **kwargs)

        self.max_steps = kwargs.get("max_steps", 5)
        self.num_candidates = kwargs.get("num_candidates", 5)
        self.difficulty = kwargs.get("difficulty", "introductory")
        self.max_problems = kwargs.get("max_problems", None)
        self.timeout_per_test = kwargs.get("timeout_per_test", 10)

        # State
        self._task_id: int = 0
        self._prompt: str = ""
        self._test_cases: List[Dict] = []
        self._test_code: str = ""          # constructed assert-style test string
        self._fn_name: Optional[str] = None
        self._starter_code: str = ""
        self._solution: str = ""
        self._current_code: str = ""
        self._test_result: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []
        self._step_count: int = 0
        self._done: bool = False
        self._reward: float = 0.0
        self._action_texts: List[str] = []

        # Dataset
        self._problems: List[Dict] = []
        self._problem_idx: int = 0
        self._load_data()

    def _load_data(self):
        """Load APPS problems from HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Please install 'datasets': pip install datasets"
            )

        diff_label = self.difficulty
        logger.info(f"Loading APPS dataset (difficulty={diff_label})...")
        # Use parquet revision for compatibility with datasets >= 4.x
        # (script-based loading is no longer supported)
        ds = load_dataset(
            "codeparrot/apps", split="test",
            revision="refs/convert/parquet",
        )

        problems = []
        for item in ds:
            if item.get("difficulty", "") != diff_label:
                continue

            # Parse test cases
            test_cases = []
            fn_name = None
            try:
                io_raw = json.loads(item.get("input_output", "{}"))
                inputs = io_raw.get("inputs", [])
                outputs = io_raw.get("outputs", [])
                fn_name = io_raw.get("fn_name", None)

                for inp, out in zip(inputs, outputs):
                    test_cases.append({
                        "input": inp if isinstance(inp, str) else str(inp),
                        "output": out if isinstance(out, str) else str(out),
                    })
            except (json.JSONDecodeError, KeyError):
                continue

            if len(test_cases) == 0:
                continue

            # Build an assert-style test string for _safe_exec compatibility
            test_code = self._build_test_code(fn_name, test_cases)

            # Extract first solution for oracle (if available)
            solutions_raw = item.get("solutions", "")
            solution = ""
            if solutions_raw:
                try:
                    sols_list = json.loads(solutions_raw)
                    if isinstance(sols_list, list) and sols_list:
                        solution = sols_list[0]
                except (json.JSONDecodeError, TypeError):
                    solution = solutions_raw if isinstance(solutions_raw, str) else ""

            problems.append({
                "problem_id": item.get("problem_id", len(problems)),
                "question": item.get("question", ""),
                "difficulty": diff_label,
                "test_cases": test_cases,
                "test_code": test_code,
                "fn_name": fn_name,
                "starter_code": item.get("starter_code", ""),
                "solution": solution,
            })

            if self.max_problems and len(problems) >= self.max_problems:
                break

        if not problems:
            raise ValueError(
                f"No APPS problems found for difficulty={diff_label}. "
                f"Check dataset availability."
            )

        self._problems = problems
        logger.info(f"APPSEnv: loaded {len(self._problems)} {diff_label} problems")

    @staticmethod
    def _build_test_code(fn_name: Optional[str], test_cases: List[Dict]) -> str:
        """Build an assert-style test string from IO test cases."""
        # For function-call style problems with fn_name
        if fn_name:
            lines = []
            for i, tc in enumerate(test_cases[:5]):  # limit to 5 tests
                inp = tc["input"].strip()
                out = tc["output"].strip()
                lines.append(f"assert str({fn_name}({inp})).strip() == {repr(out)}, "
                             f"'Test {i} failed'")
            return "\n".join(lines)

        # For stdin/stdout style — we handle these via _safe_exec_io at runtime
        return ""  # empty = use IO-style execution

    # ── lifecycle ─────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict]:
        """Reset to a new APPS problem. Returns (observation, info)."""
        if seed is not None:
            import numpy as np
            np.random.seed(seed)

        problem = self._problems[self._problem_idx % len(self._problems)]
        self._problem_idx += 1

        self._task_id = problem["problem_id"]
        self._prompt = problem["question"]
        self._test_cases = problem["test_cases"]
        self._test_code = problem.get("test_code", "")
        self._fn_name = problem.get("fn_name")
        self._starter_code = problem.get("starter_code", "")
        self._solution = problem.get("solution", "")
        self._current_code = ""
        self._test_result = {}
        self._history = []
        self._step_count = 0
        self._done = False
        self._reward = 0.0
        self._action_texts = []

        self._generate_candidate_actions()

        obs = self._build_observation(initial=True)
        info = {
            "problem_id": self._task_id,
            "difficulty": self.difficulty,
            "has_fn_name": self._fn_name is not None,
        }
        return obs, info

    def step(self, action: int) -> Tuple[Any, float, bool, bool, Dict]:
        """Execute action (select a candidate code). Returns (obs, reward, terminated, truncated, info)."""
        if action < len(self._action_texts):
            new_code = self._action_texts[action]
        else:
            new_code = self._current_code or "pass"

        self._current_code = new_code
        self._step_count += 1

        # Execute code against tests
        self._test_result = self._run_tests(new_code)
        pass_rate = self._test_result["pass_rate"]
        reward = pass_rate
        all_passed = self._test_result["passed"]

        self._history.append({
            "step": self._step_count,
            "action": action,
            "num_candidates": len(self._action_texts),
            "code": new_code,
            "code_lines": len(new_code.splitlines()),
            "test_result": dict(self._test_result),
            "pass_rate": pass_rate,
            "all_passed": all_passed,
            "state_category": self.classify_state(),
        })

        max_reached = self._step_count >= self.max_steps

        terminated = all_passed
        truncated = max_reached and not all_passed
        self._done = terminated or truncated
        self._reward = reward

        obs = self._build_observation(initial=False)

        if not self._done:
            self._generate_candidate_actions()

        return obs, float(reward), terminated, truncated, {
            "pass_rate": pass_rate,
            "all_passed": all_passed,
            "error": self._test_result.get("error"),
        }

    def _run_tests(self, code: str) -> Dict:
        """Run code against tests, choosing function-call or IO style."""
        if self._test_code:
            # Function-call style (has fn_name with assert-based tests)
            return _safe_exec(code, self._test_code, timeout=self.timeout_per_test)
        else:
            # IO style (stdin/stdout)
            return _safe_exec_io(code, self._test_cases, timeout=self.timeout_per_test)

    def _build_observation(self, initial: bool = False) -> str:
        """Build the observation string seen by the agent."""
        lines = [f"Task: {self._prompt}"]

        if self._starter_code:
            lines.append(f"\nStarter code:\n```python\n{self._starter_code}\n```")

        if initial:
            lines.append("\nGenerate a Python solution for the task.")
            if self._fn_name:
                lines.append(f"The function should be named: {self._fn_name}")
            # Show example test for reference
            if self._test_cases:
                tc = self._test_cases[0]
                lines.append(f"\nExample input:\n{tc['input'][:200]}")
                lines.append(f"Expected output:\n{tc['output'][:200]}")
        else:
            lines.append(f"\nCurrent code (step {self._step_count}):")
            lines.append(f"```python\n{self._current_code}\n```")
            if self._test_result:
                tr = self._test_result
                lines.append(f"\nTest results: {tr['num_passed']}/{tr['num_total']} passed")
                if tr["error"]:
                    lines.append(f"Error: {tr['error']}")
                if tr["passed"]:
                    lines.append("✅ All tests passed!")

        return "\n".join(lines)

    def _generate_candidate_actions(self):
        """
        Generate candidate code variants for the discrete action interface.

        Same pattern as MBPPEnv:
          - Action 0: oracle solution (if available)
          - Remaining: current code / mutations
        """
        candidates = []

        # Oracle solution
        if self._solution:
            candidates.append(self._solution)

        # Current code (no-change option)
        if self._current_code:
            candidates.append(self._current_code)

            # Simple mutation for indentation errors
            if self._test_result.get("error_type") == "IndentationError":
                fixed = textwrap.dedent(self._current_code)
                candidates.append(fixed)
        else:
            # First step: provide a skeleton
            if self._fn_name:
                skeleton = f"def {self._fn_name}(*args, **kwargs):\n    pass"
            elif self._starter_code:
                skeleton = self._starter_code
            else:
                skeleton = "# TODO: implement solution\npass"
            candidates.append(skeleton)

        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            c_stripped = c.strip()
            if c_stripped and c_stripped not in seen:
                seen.add(c_stripped)
                unique.append(c)

        if not unique:
            unique = ["pass"]

        self._action_texts = unique

    # ── snapshot / restore ────────────────────────────────────────

    def get_state(self) -> EnvState:
        return EnvState(
            data={
                "task_id": self._task_id,
                "prompt": self._prompt,
                "test_cases": self._test_cases,
                "test_code": self._test_code,
                "fn_name": self._fn_name,
                "starter_code": self._starter_code,
                "solution": self._solution,
                "current_code": self._current_code,
                "test_result": dict(self._test_result) if self._test_result else {},
                "history": list(self._history),
                "step_count": self._step_count,
                "done": self._done,
                "reward": self._reward,
                "action_texts": list(self._action_texts),
                "problem_idx": self._problem_idx,
            },
            env_type="apps",
        )

    def set_state(self, state: EnvState):
        d = state.data
        self._task_id = d["task_id"]
        self._prompt = d["prompt"]
        self._test_cases = d.get("test_cases", [])
        self._test_code = d.get("test_code", "")
        self._fn_name = d.get("fn_name")
        self._starter_code = d.get("starter_code", "")
        self._solution = d.get("solution", "")
        self._current_code = d["current_code"]
        self._test_result = d.get("test_result", {})
        self._history = list(d.get("history", []))
        self._step_count = d["step_count"]
        self._done = d["done"]
        self._reward = d["reward"]
        self._action_texts = list(d.get("action_texts", []))
        self._problem_idx = d.get("problem_idx", self._problem_idx)

    # ── action space ──────────────────────────────────────────────

    def get_actions(self) -> List[int]:
        return list(range(len(self._action_texts)))

    def get_action_name(self, action: int) -> str:
        if action < len(self._action_texts):
            code = self._action_texts[action]
            first_line = code.split("\n")[0][:80]
            return f"code_{action}: {first_line}..."
        return f"action_{action}"

    def get_action_code(self, action: int) -> str:
        """Return the full code text for an action."""
        if action < len(self._action_texts):
            return self._action_texts[action]
        return ""

    # ── text description ──────────────────────────────────────────

    def get_text_description(self) -> str:
        parts = [f"Task: {self._prompt}"]
        parts.append(f"Step: {self._step_count}")

        if self._fn_name:
            parts.append(f"Function name: {self._fn_name}")
        if self._starter_code:
            parts.append(f"Starter code:\n```python\n{self._starter_code}\n```")

        if self._current_code:
            parts.append(f"\nCurrent code:\n```python\n{self._current_code}\n```")
        if self._test_result:
            tr = self._test_result
            parts.append(f"Test results: {tr['num_passed']}/{tr['num_total']} passed")
            if tr.get("error"):
                parts.append(f"Error: {tr['error']}")

        parts.append(f"\nAvailable actions (code variants):")
        for i, code in enumerate(self._action_texts):
            preview = code.split("\n")[0][:60]
            parts.append(f"  {i}: {preview}...")

        return "\n".join(parts)

    # ── forward value (heuristic) ─────────────────────────────────

    def forward_value(self, action: int) -> float:
        """Run the code and measure pass rate (same as MBPP — code exec is cheap)."""
        if action >= len(self._action_texts):
            return -1.0
        code = self._action_texts[action]
        result = self._run_tests(code)
        return result["pass_rate"]

    # ── success ───────────────────────────────────────────────────

    def is_success(self, reward: float, terminated: bool, info: Dict) -> bool:
        return terminated and info.get("all_passed", False)

    # ── state classification ──────────────────────────────────────

    def classify_state(self) -> str:
        """Classify based on test pass rate progress."""
        if not self._test_result:
            return "no_attempt"
        pr = self._test_result.get("pass_rate", 0.0)
        if pr == 0.0:
            return "all_failing"
        elif pr < 1.0:
            return "partial_pass"
        else:
            return "all_passing"

    # ── state key ─────────────────────────────────────────────────

    def make_state_key(self) -> str:
        code_hash = hashlib.md5(
            f"{self._task_id}:{self._step_count}:{self._current_code}".encode()
        ).hexdigest()[:8]
        return f"apps_{self._task_id}_s{self._step_count}_{code_hash}"

    # ── greedy oracle action ─────────────────────────────────────

    def greedy_oracle_action(self) -> int:
        """Oracle = choose the solution code (action 0 if loaded)."""
        if self._solution and self._action_texts and self._action_texts[0] == self._solution:
            return 0
        # Fallback: evaluate all candidates, pick best pass rate
        best_action = 0
        best_rate = -1.0
        for i, code in enumerate(self._action_texts):
            result = self._run_tests(code)
            if result["pass_rate"] > best_rate:
                best_rate = result["pass_rate"]
                best_action = i
        return best_action

    # ── trajectory ────────────────────────────────────────────────

    def get_trajectory(self) -> Dict[str, Any]:
        """
        Return the full trajectory of the current (or just-finished) episode.

        Designed for post-hoc analysis of code environment paths:
          - problem metadata (id, difficulty, fn_name, starter_code)
          - per-step records: action chosen, code submitted, test results,
            pass rate progression, state category
          - outcome: success, final reward, total steps
        """
        return {
            "problem_id": self._task_id,
            "difficulty": self.difficulty,
            "has_fn_name": self._fn_name is not None,
            "fn_name": self._fn_name,
            "has_starter_code": bool(self._starter_code),
            "num_test_cases": len(self._test_cases),
            "has_test_code": bool(self._test_code),
            "prompt_length": len(self._prompt),
            "solution_length": len(self._solution) if self._solution else 0,
            # Episode outcome
            "steps": self._step_count,
            "done": self._done,
            "final_reward": self._reward,
            "final_pass_rate": self._test_result.get("pass_rate", 0.0) if self._test_result else 0.0,
            "success": self._test_result.get("passed", False) if self._test_result else False,
            # Per-step trajectory (the core data)
            "history": list(self._history),
        }

    # ── convenience ──────────────────────────────────────────────

    def get_num_episodes(self) -> int:
        """Total number of available problems."""
        return len(self._problems)

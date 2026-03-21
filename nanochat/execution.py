"""
Sandboxed execution utilities for running Python code that comes out of an LLM.
沙盒执行工具，用于运行 LLM 生成的 Python 代码。
Adapted from OpenAI HumanEval code:
改编自 OpenAI HumanEval 代码：
https://github.com/openai/human-eval/blob/master/human_eval/execution.py

What is covered:
覆盖内容：
- Each execution runs in its own process (can be killed if it hangs or crashes)
- 每次执行在其自己的进程中运行（如果挂起或崩溃可以被终止）
- Execution is limited by a timeout to stop infinite loops
- 执行受超时限制以停止无限循环
- Memory limits are enforced by default (256MB)
- 默认强制执行内存限制（256MB）
- stdout and stderr are captured and returned
- stdout 和 stderr 被捕获并返回
- Code runs in a temporary directory that is deleted afterwards
- 代码在临时目录中运行，之后被删除
- Dangerous functions are disabled (examples: os.system, os.kill, shutil.rmtree, subprocess.Popen)
- 危险函数被禁用（例如：os.system、os.kill、shutil.rmtree、subprocess.Popen）

What is not covered:
未覆盖内容：
- Not a true security sandbox
- 不是真正的安全沙盒
- Network access is not blocked (e.g. sockets could be opened)
- 网络访问未被阻止（例如可以打开套接字）
- Python's dynamic features (e.g. ctypes) could bypass restrictions
- Python 的动态特性（例如 ctypes）可能绕过限制
- No kernel-level isolation (no seccomp, no containers, no virtualization)
- 没有内核级隔离（没有 seccomp、容器、虚拟化）

Overall this sandbox is good for evaluation of generated code and protects against
总的来说，这个沙盒适用于评估生成的代码并防止
accidental destructive behavior, but it is not safe against malicious adversarial code.
意外的破坏性行为，但对恶意对抗性代码不安全。
"""

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
from dataclasses import dataclass
from typing import Optional

# -----------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of executing Python code in a sandbox."""
    """在沙盒中执行 Python 代码的结果。"""
    success: bool
    stdout: str
    stderr: str
    error: Optional[str] = None
    timeout: bool = False
    memory_exceeded: bool = False

    def __repr__(self):
        parts = []
        parts.append(f"ExecutionResult(success={self.success}")
        if self.timeout:
            parts.append(", timeout=True")
        if self.memory_exceeded:
            parts.append(", memory_exceeded=True")
        if self.error:
            parts.append(f", error={self.error!r}")
        if self.stdout:
            parts.append(f", stdout={self.stdout!r}")
        if self.stderr:
            parts.append(f", stderr={self.stderr!r}")
        parts.append(")")
        return "".join(parts)


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def capture_io():
    """Capture stdout and stderr, and disable stdin."""
    """捕获 stdout 和 stderr，并禁用 stdin。"""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    stdin_block = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stdout_capture):
        with contextlib.redirect_stderr(stderr_capture):
            with redirect_stdin(stdin_block):
                yield stdout_capture, stderr_capture


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""
    """读取时会抛出异常的 StringIO"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        """如果 IO 对象可读则返回 True。"""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    这会禁用各种破坏性函数并防止生成的代码
    from interfering with the test (e.g. fork bomb, killing other processes,
    干扰测试（例如 fork 炸弹、杀死其他进程、
    removing filesystem files, etc.)
    删除文件系统文件等）

    WARNING
    警告
    This function is NOT a security sandbox. Untrusted code, including, model-
    此函数不是安全沙盒。不受信任的代码，包括模型
    generated code, should not be blindly executed outside of one. See the
    生成的代码，不应在沙盒之外盲目执行。请参阅
    Codex paper for more information about OpenAI's code sandbox, and proceed
    Codex 论文以获取有关 OpenAI 代码沙盒的更多信息，并
    with caution.
    谨慎操作。
    """

    if platform.uname().system != "Darwin":
        # These resource limit calls seem to fail on macOS (Darwin), skip?
        # 这些资源限制调用在 macOS (Darwin) 上似乎会失败，跳过？
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def _unsafe_execute(code: str, timeout: float, maximum_memory_bytes: Optional[int], result_dict):
    """Execute code in a subprocess with safety guards. Results are written to result_dict."""
    """在子进程中执行代码并带有安全保护。结果写入 result_dict。"""
    with create_tempdir():

        # These system calls are needed when cleaning up tempdir.
        # 清理 tempdir 时需要这些系统调用。
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        unlink = os.unlink

        # Disable functionalities that can make destructive changes to the test.
        # 禁用可能对测试造成破坏性更改的功能。
        reliability_guard(maximum_memory_bytes=maximum_memory_bytes)

        # Default to failure
        # 默认为失败
        result_dict.update({
            "success": False,
            "stdout": "",
            "stderr": "",
            "timeout": False,
            "memory_exceeded": False,
            "error": None,
        })

        try:
            exec_globals = {}
            with capture_io() as (stdout_capture, stderr_capture):
                with time_limit(timeout):
                    # WARNING
                    # 警告
                    # This program exists to execute untrusted model-generated code. Although
                    # 此程序用于执行不受信任的模型生成代码。虽然
                    # it is highly unlikely that model-generated code will do something overtly
                    # 模型生成的代码极不可能做出明显的
                    # malicious in response to this test suite, model-generated code may act
                    # 恶意行为来响应此测试套件，但模型生成的代码可能
                    # destructively due to a lack of model capability or alignment.
                    # 由于缺乏模型能力或对齐而表现出破坏性。
                    # Users are strongly encouraged to sandbox this evaluation suite so that it
                    # 强烈建议用户将此评估套件沙盒化，以便它
                    # does not perform destructive actions on their host or network. For more
                    # 不会对其主机或网络执行破坏性操作。有关
                    # information on how OpenAI sandboxes its code, see the accompanying paper.
                    # OpenAI 如何沙盒其代码的更多信息，请参阅附带的论文。
                    # Once you have read this disclaimer and taken appropriate precautions,
                    # 一旦您阅读了此免责声明并采取了适当的预防措施，
                    # uncomment the following line and proceed at your own risk:
                    # 取消注释以下行并自行承担风险：
                    exec(code, exec_globals)

            result_dict.update({
                "success": True,
                "stdout": stdout_capture.getvalue(),
                "stderr": stderr_capture.getvalue(),
            })

        except TimeoutException:
            result_dict.update({
                "timeout": True,
                "error": "Execution timed out",
            })

        except MemoryError as e:
            result_dict.update({
                "memory_exceeded": True,
                "error": f"Memory limit exceeded: {e}",
            })

        except BaseException as e:
            result_dict.update({
                "error": f"{type(e).__name__}: {e}",
            })

        # Needed for cleaning up.
        # 清理所需。
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir
        os.unlink = unlink


def execute_code(
    code: str,
    timeout: float = 5.0, # 5 seconds default
                            # 默认 5 秒
    maximum_memory_bytes: Optional[int] = 256 * 1024 * 1024, # 256MB default
                                                              # 默认 256MB
) -> ExecutionResult:
    """
    Execute Python code in a sandboxed environment.
    在沙盒环境中执行 Python 代码。

    Args:
    参数：
        code: Python code to execute as a string
        code: 要执行的 Python 代码字符串
        timeout: Maximum execution time in seconds (default: 5.0)
        timeout: 最大执行时间（秒）（默认：5.0）
        maximum_memory_bytes: Memory limit in bytes (default: 256MB, None to disable)
        maximum_memory_bytes: 内存限制（字节）（默认：256MB，None 禁用）

    Returns:
    返回：
        ExecutionResult with success status, stdout/stderr, and error information
        ExecutionResult 包含成功状态、stdout/stderr 和错误信息

    Example:
    示例：
        >>> result = execute_code("print('hello world')")
        >>> result.success
        True
        >>> result.stdout
        'hello world\\n'
    """

    manager = multiprocessing.Manager()
    result_dict = manager.dict()

    p = multiprocessing.Process(
        target=_unsafe_execute,
        args=(code, timeout, maximum_memory_bytes, result_dict)
    )
    p.start()
    p.join(timeout=timeout + 1)

    if p.is_alive():
        p.kill()
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            error="Execution timed out (process killed)",
            timeout=True,
            memory_exceeded=False,
        )

    if not result_dict:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            error="Execution failed (no result returned)",
            timeout=True,
            memory_exceeded=False,
        )

    return ExecutionResult(
        success=result_dict["success"],
        stdout=result_dict["stdout"],
        stderr=result_dict["stderr"],
        error=result_dict["error"],
        timeout=result_dict["timeout"],
        memory_exceeded=result_dict["memory_exceeded"],
    )


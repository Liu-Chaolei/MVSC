import logging
import subprocess
import threading
import time
from queue import Empty, Queue
from typing import Iterator, List, Optional
from .logging import get_logger


class BaseTask:
    """
    A class to manage command execution as a subprocess with output streaming.

    This class handles running a command as a subprocess, capturing its output,
    and providing methods to access that output either line by line or as a stream.
    Each task can only be started once and can have only one consumer of its output.
    """

    def __init__(self, command: List[str], logger: Optional[logging.Logger] = None):
        """
        Initialize a new BaseTask.

        Args:
            command: List of command arguments to execute
            logger: Optional logger for task status messages
        """
        self.command = command
        self.logger = logger or get_logger(__name__)

        # Process information
        self.pid: Optional[int] = None
        self.process: Optional[subprocess.Popen] = None

        # State tracking
        self.started = False
        self.finished = False
        self.error = None
        self.return_code: Optional[int] = None

        # Output handling
        self.output_queue = Queue()
        self.complete_output = []
        self.output_thread = None
        self.process_completed = threading.Event()

    def run(self) -> None:
        """
        Run the task as a subprocess and start capturing its output.

        Raises:
            RuntimeError: If the task has already been started
            subprocess.SubprocessError: If the subprocess fails to start
        """
        if self.started:
            raise RuntimeError("Task has already been started")

        self.started = True

        command_str = " ".join(self.command)
        self.logger.info(f"Running command:\n{command_str}")

        try:
            # Start the process
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            self.pid = self.process.pid
            self.logger.info(f"Started process with PID: {self.pid}")

            # Start a thread to read the output
            self.output_thread = threading.Thread(
                target=self._read_output,
                args=(self.process, self.output_queue, self.process_completed),
            )
            self.output_thread.daemon = True
            self.output_thread.start()

        except Exception as e:
            self.error = e
            self.finished = True
            self.logger.error(f"Failed to start process: {e}")
            raise

    def _read_output(
        self, process: subprocess.Popen, queue: Queue, completed_event: threading.Event
    ) -> None:
        """
        Read output from process and put it in the queue.

        Args:
            process: The subprocess to read from
            queue: Queue to store output lines
            completed_event: Event to signal when process is complete
        """
        for line in iter(process.stdout.readline, ""):
            if line:
                line = line.strip()
                self.complete_output.append(line)
                queue.put(line)

        # Process has completed
        self.return_code = process.wait()
        self.finished = True
        queue.put("")  # Empty string signals end of output
        self.logger.info(f"Process completed with return code: {self.return_code}")
        completed_event.set()

    def join(self) -> None:
        """Wait for the task to complete."""
        if not self.started:
            raise RuntimeError("Task has not been started")

        self.process_completed.wait()
        if self.output_thread:
            self.output_thread.join()

    def getline(self) -> str:
        """
        Get a single line of output from the task.

        Returns:
            A line of output, or empty string if task has finished and queue is empty
        """
        if not self.started:
            raise RuntimeError("Task has not been started")

        if self.finished and self.output_queue.empty():
            return ""

        try:
            return self.output_queue.get(timeout=0.1)
        except Empty:
            return "" if self.finished else self.getline()

    def iter_output(self) -> Iterator[str]:
        """
        Iterate over the output of the task.

        Yields:
            Lines of output from the task
        """
        if not self.started:
            raise RuntimeError("Task has not been started")

        while not self.process_completed.is_set() or not self.output_queue.empty():
            try:
                line = self.output_queue.get(timeout=0.1)
                if line:  # Skip empty string which signals end
                    yield line
            except Empty:
                pass
            time.sleep(0.01)

    def get_output(self) -> str:
        """
        Get the complete output of the task.

        Returns:
            The complete output as a string

        Raises:
            RuntimeError: If the task has not finished
        """
        if not self.finished:
            self.join()

        return "\n".join(self.complete_output)

    def is_finished(self) -> bool:
        """Check if the task has finished."""
        return self.finished

    def get_pid(self) -> Optional[int]:
        """Get the PID of the subprocess."""
        return self.pid

    def get_return_code(self) -> Optional[int]:
        """Get the return code of the subprocess."""
        return self.return_code


import time
from typing import Optional, Dict
from contextlib import contextmanager

class Timer:
    """High-precision timer for measuring execution time (seconds)."""
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self) -> float:
        """Start the timer and return start timestamp (perf_counter)."""
        self.start_time = time.perf_counter()
        self.end_time = None
        return self.start_time

    def stop(self) -> float:
        """
        Stop the timer and return elapsed time in seconds.
        Raises RuntimeError if timer wasn't started.
        """
        if self.start_time is None:
            raise RuntimeError("Timer was not started")
        self.end_time = time.perf_counter()
        return self.elapsed_time

    @property
    def elapsed_time(self) -> float:
        """Elapsed time in seconds (uses current time if not stopped yet)."""
        if self.start_time is None:
            raise RuntimeError("Timer was not started")
        end_time = self.end_time or time.perf_counter()
        return end_time - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_time * 1000.0

    def reset(self):
        """Reset timer to initial state."""
        self.start_time = None
        self.end_time = None

class StepTimer:
    
    def __init__(self):
        self.steps: Dict[str, float] = {}
        self.current_step: Optional[str] = None
        self.current_timer: Optional[Timer] = None
        self.overall_timer = Timer()

    def start_overall(self):
        """Reset state and start the overall timer."""
        self.steps = {}
        self.current_step = None
        self.current_timer = None
        self.overall_timer.reset()
        self.overall_timer.start()

    def start_step(self, step_name: str):
        """Start timing a named step (stops previous running step)."""
        if self.current_timer is not None:
            self.stop_current_step()
        self.current_step = step_name
        self.current_timer = Timer()
        self.current_timer.start()

    def stop_current_step(self) -> Optional[float]:
        """
        Stop the current step and return its duration in milliseconds.
        If no step is running, returns None.
        """
        if self.current_timer is None or self.current_step is None:
            return None
        try:
            duration_ms = self.current_timer.stop() * 1000.0
        except RuntimeError:
            duration_ms = 0.0
        prev = self.steps.get(self.current_step, 0.0)
        self.steps[self.current_step] = prev + duration_ms
        self.current_step = None
        self.current_timer = None
        return duration_ms

    def stop_overall(self) -> float:
        """
        Stop the overall timer (and any running step). Return total duration in seconds.
        This is what LLMMonitor.__exit__ expects (seconds).
        """
        if self.current_timer is not None:
            self.stop_current_step()
        try:
            total_seconds = self.overall_timer.stop()
        except RuntimeError:
            total_seconds = 0.0
        return total_seconds

    def get_step_duration(self, step_name: str) -> Optional[float]:
        """Get duration of a specific step in milliseconds (or None if not present)."""
        return self.steps.get(step_name)

    def get_all_steps(self) -> Dict[str, float]:
        """
        Return a copy of all step durations (milliseconds).
        If a step is currently running, include its up-to-now duration.
        """
        steps_copy = dict(self.steps)
        if self.current_step and self.current_timer:
            try:
                now_ms = (time.perf_counter() - self.current_timer.start_time) * 1000.0
                steps_copy[self.current_step] = steps_copy.get(self.current_step, 0.0) + now_ms
            except Exception:
                pass
        return steps_copy

    @property
    def total_duration_ms(self) -> float:
        """
        Current total duration in milliseconds.
        If overall timer hasn't been started, returns 0.0.
        """
        try:
            return self.overall_timer.elapsed_ms
        except RuntimeError:
            return 0.0

    def reset(self):
        """Reset all timers and step state."""
        self.steps.clear()
        self.current_step = None
        self.current_timer = None
        self.overall_timer.reset()

@contextmanager
def timed_step(step_timer: StepTimer, step_name: str):
    """Context manager for timing a step."""
    step_timer.start_step(step_name)
    try:
        yield
    finally:
        step_timer.stop_current_step()

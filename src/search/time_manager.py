import time

class TimeManager:
    def __init__(self):
        self._start = 0.0
        self._limit_ms = 0.0
        self.stopped = False

    def start(self, movetime_ms: float):
        self._start = time.time()
        self._limit_ms = movetime_ms
        self.stopped = False

    def elapsed_ms(self) -> float:
        return (time.time() - self._start) * 1000.0

    def should_stop(self) -> bool:
        if self.stopped:
            return True
        if self.elapsed_ms() >= self._limit_ms:
            self.stopped = True
        return self.stopped

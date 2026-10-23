"""Monotonic per-side match clock; independent of browser timers and CPU clocks."""
import time


class MatchClock:
    def __init__(self, minutes=0, now=None):
        if type(minutes) is not int or not 0 <= minutes <= 180:
            raise ValueError('每方总时间必须是 0～180 分钟的整数')
        self.minutes = minutes
        self.now = now or time.monotonic
        self.remaining = {1: minutes * 60., -1: minutes * 60.}
        self.active = None
        self.since = None
        self.started = False

    def settle(self):
        if self.active is not None:
            now = self.now()
            self.remaining[self.active] = max(0., self.remaining[self.active] - (now - self.since))
            self.since = now

    def start(self, side):
        self.settle()
        if self.minutes:
            self.active, self.since, self.started = side, self.now(), True

    def pause(self):
        self.settle()
        self.active = self.since = None

    def expired(self):
        self.settle()
        return next((side for side in (1, -1) if self.minutes and self.remaining[side] <= 0), None)

    def budget(self, side, empty_cells, move_limit=0):
        """Reserve 30s globally and allocate across all possibly remaining turns."""
        self.settle()
        if not self.minutes:
            return move_limit or 0
        available = max(.001, (self.remaining[side] - min(30., self.remaining[side] * .1))
                        / max(1, (empty_cells + 1) // 2))
        return min(available, move_limit) if move_limit else available

    def state(self):
        self.settle()
        return dict(minutes=self.minutes, redMs=round(self.remaining[1] * 1000),
                    blueMs=round(self.remaining[-1] * 1000), active=self.active,
                    started=self.started, enabled=bool(self.minutes))

    def restore(self, snapshot):
        self.pause()
        self.remaining = {1: snapshot['redMs'] / 1000, -1: snapshot['blueMs'] / 1000}
        self.started = snapshot.get('started', True)

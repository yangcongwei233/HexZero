"""双方独立的单调比赛计时，不依赖浏览器计时器或处理器时钟。"""
import time


class MatchClock:
    def __init__(self, minutes=0, now=None):
        """校验每方总时限并初始化剩余秒数；可注入单调时钟函数以便验证，零分钟表示禁用计时。"""
        if type(minutes) is not int or not 0 <= minutes <= 180:
            raise ValueError('每方总时间必须是 0～180 分钟的整数')
        self.minutes = minutes
        self.now = now or time.monotonic
        self.remaining = {1: minutes * 60., -1: minutes * 60.}
        self.active = None
        self.since = None
        self.started = False

    def settle(self):
        """将上次结算以来的时间扣除到当前行动方，剩余时间最低为零，并刷新结算时刻。"""
        if self.active is not None:
            now = self.now()
            self.remaining[self.active] = max(0., self.remaining[self.active] - (now - self.since))
            self.since = now

    def start(self, side):
        """先结算上一方，再切换当前计时方；未启用比赛时限时不启动时钟。"""
        self.settle()
        if self.minutes:
            self.active, self.since, self.started = side, self.now(), True

    def pause(self):
        """结算当前已用时间并暂停计时，不重置双方余额。"""
        self.settle()
        self.active = self.since = None

    def expired(self):
        """结算后返回时间耗尽的棋色，均未超时或禁用计时时返回空值。"""
        self.settle()
        return next((side for side in (1, -1) if self.minutes and self.remaining[side] <= 0), None)

    def budget(self, side, empty_cells, move_limit=0):
        """全局预留 30 秒，再按可能的剩余回合分配用时。"""
        self.settle()
        if not self.minutes:
            return move_limit or 0
        available = max(.001, (self.remaining[side] - min(30., self.remaining[side] * .1))
                        / max(1, (empty_cells + 1) // 2))
        return min(available, move_limit) if move_limit else available

    def state(self):
        """结算棋钟并返回毫秒级显示快照，包含当前计时方、是否启动及是否启用。"""
        self.settle()
        return dict(minutes=self.minutes, redMs=round(self.remaining[1] * 1000),
                    blueMs=round(self.remaining[-1] * 1000), active=self.active,
                    started=self.started, enabled=bool(self.minutes))

    def restore(self, snapshot):
        """暂停计时并恢复快照中的双方余额；恢复后不会自行继续扣时。"""
        self.pause()
        self.remaining = {1: snapshot['redMs'] / 1000, -1: snapshot['blueMs'] / 1000}
        self.started = snapshot.get('started', True)

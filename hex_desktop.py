"""Tk 桌面对弈：主线程绘图，单工作线程处理棋局和推理。
运行：python hex_desktop.py --models models"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import math
import os
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont

# 导入 NumPy 前限制 BLAS 线程；界面线程数只控制 ONNX intra_op。
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
os.environ['HEX_FORCE_CPU'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

ROOT = Path(__file__).resolve().parent
PRESETS = (800, 1200, 2000, 3200, 6400, 12800, 51200)
RED, BLUE = '#ce594a', '#3979c7'
ACTORS = {'human': '人类', 'trained': 'HexZero', 'record': '棋谱'}
MARKS = {-2: '大劣', -1: '小劣', 0: '一般', 1: '小好', 2: '大好'}


def coordinate(row, col, size=11):
    """仅用于显示，不能把屏幕坐标直接传给网络的 canonical 坐标。"""
    return f'{chr(65 + col)}{size - row}'


def score_label(value):
    """Q 代表当前执棋方的局面估值，不是经过校准的获胜概率。"""
    if value is None:
        return '未搜索'
    mark = -2 if value < -.6 else -1 if value < -.2 else 0 if value <= .2 else 1 if value <= .6 else 2
    return f'{value:+.3f}  /  {mark:+d} {MARKS[mark]}'


class DesktopWindow:
    """界面保存只读快照；工作线程独占搜索树，结果由 Tk.after 轮询。"""

    def __init__(self, root, models, data, model=None, threads=None, competition=False):
        self.root, self.models, self.data = root, Path(models), Path(data)
        self.competition = bool(competition)
        self.model_controls = []
        self.ai_controls = []
        self.requested_model, self.requested_threads = model, threads
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='hex-engine')
        self.service = self.future = None
        self.operation = None
        self.snapshot = self.state = self.config = None
        self.reply_pending = self.closing = False
        self.pending_analysis = None
        self.preview = []
        self.centers = {}
        self.controls = []
        self.last_clock_poll = self.last_catalog_poll = 0.
        self.history_key = self.analysis_key = None
        self.root.title('HexZero · 3250 比赛版' if self.competition else 'HexZero · 桌面对弈')
        self.root.geometry('1250x830')
        self.root.minsize(1000, 720)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self._build()
        self._load()
        self.root.after(40, self._poll)

    def _button(self, parent, text, command, **grid):
        button = ttk.Button(parent, text=text, command=command)
        button.grid(padx=4, pady=4, sticky='ew', **grid)
        self.controls.append(button)
        return button

    def _build(self):
        """用原生控件组织操作；棋盘使用 Canvas 六边形，不依赖网页渲染。"""
        families = set(tkfont.families(self.root))
        self.font_family = next((name for name in ('Microsoft YaHei UI', 'Microsoft YaHei',
            'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'song ti', 'Arial') if name in families), 'TkDefaultFont')
        for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont'):
            tkfont.nametofont(name).configure(family=self.font_family, size=10)
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', font=(self.font_family, 10))
        style.configure('TFrame', background='#f4f6f2')
        style.configure('TLabel', background='#f4f6f2')
        style.configure('TButton', padding=6)
        style.configure('Treeview', rowheight=27)
        self.root.configure(background='#f4f6f2')
        top = ttk.Frame(self.root, padding=(18, 12))
        top.pack(fill='x')
        ttk.Label(top, text='HexZero 11 × 11', font=(self.font_family, 20, 'bold')).pack(side='left')
        ttk.Label(top, text='红方先手 · 连接上下     蓝方后手 · 连接左右').pack(side='right')
        layout = ttk.Panedwindow(self.root, orient='horizontal')
        layout.pack(fill='both', expand=True, padx=12)
        left, right = ttk.Frame(layout), ttk.Frame(layout, width=405)
        layout.add(left, weight=3)
        layout.add(right, weight=2)
        self.turn = tk.StringVar(value='正在加载模型与环境…')
        ttk.Label(left, textvariable=self.turn, font=(self.font_family, 13, 'bold'), padding=10).pack(fill='x')
        self.canvas = tk.Canvas(left, background='#f4f6f2', highlightthickness=0, width=660, height=560)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda event: self.draw_board())
        self.canvas.bind('<Button-1>', self.board_click)
        self.clock_text = tk.StringVar(value='棋钟未启用')
        ttk.Label(left, textvariable=self.clock_text, font=(self.font_family, 14), padding=8).pack()
        self.model_info = tk.StringVar(value='正在准备对弈环境…')
        ttk.Label(left, textvariable=self.model_info, wraplength=640, padding=8).pack(fill='x')
        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill='both', expand=True)
        self.play_tab, self.analysis_tab, self.record_tab, self.settings_tab = [ttk.Frame(self.tabs, padding=10) for _ in range(4)]
        for tab, title in zip((self.play_tab, self.analysis_tab, self.record_tab, self.settings_tab), ('对局', '候选分析', '棋谱', '设置')):
            self.tabs.add(tab, text=title)
        self._build_play()
        self._build_analysis()
        self._build_record()
        self._build_settings()
        self.status = tk.StringVar(value='正在检查模型；未放入模型时可人人对弈。')
        ttk.Label(self.root, textvariable=self.status, padding=10, wraplength=1180).pack(fill='x')
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(fill='x')

    def _build_play(self):
        tab = self.play_tab
        tab.columnconfigure((0, 1), weight=1)
        self.mode = tk.StringVar(value='人机对弈' if self.competition else '人人对弈')
        self.seat = tk.StringVar(value='先手红方')
        self.simulations = tk.StringVar(value='12800')
        for row, title, var, values in ((0, '对弈模式', self.mode, ('人人对弈', '人机对弈')),
                                       (2, '人机对弈时人的先后手', self.seat, ('先手红方', '后手蓝方')),
                                       (4, '每步搜索上限', self.simulations, PRESETS)):
            ttk.Label(tab, text=title).grid(row=row, column=0, columnspan=2, sticky='w', pady=(8, 2))
            combo = ttk.Combobox(tab, textvariable=var, values=values, state='readonly')
            combo.grid(row=row+1, column=0, columnspan=2, sticky='ew')
            if var is self.mode:
                self.mode_combo = combo
        ttk.Label(tab, text='人人模式由双方轮流点击落子；人机模式由模型应手。\n设置在新开局时生效。', wraplength=350).grid(row=6, column=0, columnspan=2, pady=12, sticky='w')
        self._button(tab, '新开一局', lambda: self.new_game(False), row=7, column=0)
        self._button(tab, '手动设置开局', lambda: self.new_game(True), row=7, column=1)
        self._button(tab, '完成开局', lambda: self.command('start'), row=8, column=0)
        self.step_button = self._button(tab, '让模型落子', lambda: self.command('step'), row=8, column=1)
        self.reply_button = self._button(tab, '继续模型应手', self.toggle_reply, row=9, column=0)
        self._button(tab, '悔棋', lambda: self.command('undo'), row=9, column=1)
        self._button(tab, '暂停 / 继续棋钟', self.toggle_clock, row=10, column=0, columnspan=2)
        ttk.Label(tab, text='手动开局时，点击空位替红蓝双方交替落子。\n暂停模型应手会等待正在计算的一手结束。', wraplength=350).grid(row=11, column=0, columnspan=2, pady=14, sticky='w')

    def _build_analysis(self):
        tab = self.analysis_tab
        tab.columnconfigure(0, weight=1)
        ttk.Label(tab, text='HexZero · 当前执棋方视角').grid(row=0, column=0, sticky='w')
        self.analyze_button = self._button(tab, '分析当前局面', self.analyze, row=1, column=0)
        self.candidates = ttk.Treeview(tab, columns=('move', 'visits', 'value'), show='headings', height=3, selectmode='browse')
        for name, title, width in (('move', '候选', 60), ('visits', '访问', 75), ('value', 'Q / 局面等级', 165)):
            self.candidates.heading(name, text=title)
            self.candidates.column(name, width=width, minwidth=50, stretch=True)
        self.candidates.grid(row=2, column=0, sticky='ew', pady=8)
        self.candidates.bind('<<TreeviewSelect>>', self.select_candidate)
        self.pv_text = tk.StringVar(value='显示前三个候选及已搜索的后续变化。')
        ttk.Label(tab, textvariable=self.pv_text, wraplength=350, justify='left').grid(row=3, column=0, sticky='w', pady=8)
        self.preview_depth = tk.IntVar(value=0)
        self.preview_scale = tk.Scale(tab, from_=0, to=6, orient='horizontal', variable=self.preview_depth,
                                     command=lambda value: self.update_preview(), label='预览后续手数', background='#f4f6f2', highlightthickness=0)
        self.preview_scale.grid(row=4, column=0, sticky='ew')
        self._button(tab, '返回实战棋盘', self.clear_preview, row=5, column=0)
        self._button(tab, '采用选中的候选', self.play_candidate, row=6, column=0)
        generated = self._button(tab, '用分析生成上一手评注', lambda: self.command('annotate', generated=True), row=7, column=0)
        self.ai_controls.extend((self.analyze_button, generated))
        ttk.Label(tab, text='Q 为当前执棋方视角，范围 −1～+1，不是胜率。\nMARK −2～+2 表示局面优劣。\n候选按访问次数、Q、先验排序；预览不落子。', wraplength=350).grid(row=8, column=0, sticky='w', pady=14)

    def _build_record(self):
        tab = self.record_tab
        tab.columnconfigure((0, 1), weight=1)
        self._button(tab, '导入 TXT / JSON', self.import_record, row=0, column=0)
        self._button(tab, '导出 JSON', lambda: self.export_record('json'), row=0, column=1)
        self._button(tab, '导出比赛 TXT', lambda: self.export_record('txt'), row=1, column=0)
        self._button(tab, '队名 / 赛事 / 赛果', self.edit_metadata, row=1, column=1)
        self.history = ttk.Treeview(tab, columns=('ply', 'color', 'move', 'actor', 'mark'), show='headings', height=10, selectmode='browse')
        for name, title, width in (('ply', '手', 30), ('color', '方', 30), ('move', '坐标', 45), ('actor', '落子者', 80), ('mark', '评分', 40)):
            self.history.heading(name, text=title)
            self.history.column(name, width=width, stretch=True)
        self.history.grid(row=2, column=0, columnspan=2, sticky='nsew')
        tab.rowconfigure(2, weight=1)
        self.history.bind('<Double-1>', self.history_seek)
        self._button(tab, '上一手', lambda: self.seek_relative(-1), row=3, column=0)
        self._button(tab, '下一手', lambda: self.seek_relative(1), row=3, column=1)
        self._button(tab, '回到开局', lambda: self.command('seek', ply=0), row=4, column=0)
        self._button(tab, '跳到末手', lambda: self.command('seek', ply=self.state['replayTotal']), row=4, column=1)
        self._button(tab, '从此局面续弈', lambda: self.command('continue'), row=5, column=0)
        self._button(tab, '编辑当前手评注', self.edit_annotation, row=5, column=1)
        self.comment_text = tk.StringVar(value='双击棋谱中的一手查看局面。')
        ttk.Label(tab, textvariable=self.comment_text, wraplength=350).grid(row=6, column=0, columnspan=2, sticky='w', pady=8)

    def _build_settings(self):
        tab = self.settings_tab
        tab.columnconfigure(0, weight=1)
        ttk.Label(tab, text='比赛版 · 固定 3250 模型' if self.competition else '当前 HexZero 模型').grid(row=0, column=0, sticky='w')
        self.model_choice = tk.StringVar()
        self.model_combo = ttk.Combobox(tab, textvariable=self.model_choice, state='readonly', width=38)
        self.model_combo.grid(row=1, column=0, sticky='ew', pady=5)
        self.model_controls.extend((
            self._button(tab, '加载模型并新开局', self.change_model, row=2, column=0),
            self._button(tab, '刷新模型目录', lambda: self.command('refresh'), row=3, column=0)))
        # 公开版允许在没有模型时选择目录；比赛版在 UI 和服务层均拒绝换模。
        choose = ttk.Button(tab, text='选择 models 文件夹', command=self.choose_models)
        choose.grid(row=4, column=0, sticky='ew', padx=4, pady=4)
        self.model_controls.append(choose)
        if self.competition:
            self.model_combo.grid_remove()
            for control in self.model_controls:
                control.grid_remove()
            ttk.Label(tab, text='已固定 3250 权重，启动时校验内容。\n支持人人对弈及与 3250 人机对弈。', wraplength=350).grid(row=1, column=0, sticky='w', pady=8)
        else:
            ttk.Label(tab, text='公开版不附带模型。将 .onnx 和同名 .onnx.json\n放入 models 后，加载即可启用人机与分析。', wraplength=350).grid(row=9, column=0, sticky='w', pady=6)
        self.threads = tk.StringVar(value=str(self.requested_threads or 4))
        ttk.Label(tab, text='CPU 推理线程数').grid(row=5, column=0, sticky='w', pady=(12, 2))
        ttk.Combobox(tab, textvariable=self.threads, values=tuple(range(4, 17)), state='readonly').grid(row=6, column=0, sticky='ew')
        self._button(tab, '应用线程数（保留棋局）', lambda: self.command('threads', threads=int(self.threads.get())), row=7, column=0)
        advanced = ttk.LabelFrame(tab, text='高级设置 · 新开局生效', padding=8)
        advanced.grid(row=8, column=0, sticky='ew', pady=12)
        advanced.columnconfigure(1, weight=1)
        self.seed, self.move_seconds, self.clock_minutes = tk.StringVar(value='0'), tk.StringVar(value='0'), tk.StringVar(value='0')
        for row, text, var in ((0, '随机种子', self.seed), (1, '单步秒数（0 不限）', self.move_seconds), (2, '每方总分钟（0 不限）', self.clock_minutes)):
            ttk.Label(advanced, text=text).grid(row=row, column=0, sticky='w', pady=4)
            ttk.Entry(advanced, textvariable=var, width=12).grid(row=row, column=1, sticky='ew', padx=5)
        ttk.Label(advanced, text='种子用于并列走法的选择与复现，不保证改变开局。\n比赛默认 0；丰富开局请使用“手动设置开局”。\n启用用时限制后，机器速度也会影响搜索结果。', wraplength=330).grid(row=3, column=0, columnspan=2, pady=8, sticky='w')

    def _load(self):
        """模型初始化也在工作线程执行，首次 Numba 编译不会冻结窗口。"""
        def initialize():
            from hex.desktop_service import DesktopService
            self.service = DesktopService(self.models, self.data, self.requested_model, self.requested_threads, self.competition)
            return self.service.snapshot()
        self.operation = 'load'
        self.future = self.executor.submit(initialize)
        self.progress.start(12)
        self._control_states()

    def _submit(self, operation, **params):
        if self.future is not None or self.service is None:
            return
        self.operation = operation
        self.future = self.executor.submit(self.service.perform, operation, **params)
        if operation not in ('state', 'refresh'):
            self.status.set({'step': '模型正在搜索…', 'analyze': '正在分析当前局面…', 'model': '正在校验并切换模型…', 'close': '正在保存棋局…'}.get(operation, '正在处理…'))
            self.progress.start(12)
        self._control_states()

    def command(self, operation, **params):
        """在 UI 主线程读取 revision，再把普通 Python 数据传给工作线程。"""
        if not self.state or self.future is not None:
            return
        if operation not in ('step', 'state', 'refresh'):
            self.reply_pending = False
        if operation not in ('state', 'refresh', 'threads', 'save', 'export'):
            self.clear_preview()
        self._submit(operation, revision=self.state['revision'], **params)

    def _poll(self):
        """处理异步结果、人机回应和时钟刷新；永远只有一个在途命令。"""
        if self.future is not None and self.future.done():
            operation, future = self.operation, self.future
            self.future = None
            self.progress.stop()
            try:
                result = future.result()
                if operation == 'close':
                    self.executor.shutdown(wait=False)
                    self.root.destroy()
                    return
                if result is not None:
                    first = self.state is None
                    self.snapshot, self.state, self.config = result, result['session'], result['config']
                    if first or operation in ('model', 'load'):
                        self.simulations.set(str(self.state['mctsSims']))
                        self.mode.set('人人对弈' if self.state['mode'] == 'pvp' else '人机对弈')
                        self.seat.set('先手红方' if self.state['myPlayer'] == 1 else '后手蓝方')
                    self.render()
                if operation in ('new', 'move', 'start') and self.state['mode'] == 'human':
                    self.reply_pending = True
                if operation not in ('state', 'refresh'):
                    warnings = self.state.get('recordWarnings', []) if self.state else []
                    self.status.set('；'.join(warnings) if warnings else '已完成。棋谱保存在 ' + str(self.data / 'games'))
                    if operation == 'load' and self.config and self.config.get('modelError'):
                        self.status.set('人人对弈可用。' + self.config['modelError'])
            except Exception as exc:
                logging.exception('Desktop operation failed: %s', operation)
                self.reply_pending = False
                self.status.set(str(exc))
                if operation == 'close':
                    self.closing = False
                if not self.closing:
                    messagebox.showerror('HexZero', str(exc), parent=self.root)
            self._control_states()
        if self.future is None:
            if self.closing:
                if self.service is None:
                    self.executor.shutdown(wait=False)
                    self.root.destroy()
                    return
                self._submit('close')
            elif self.pending_analysis and self.state:
                kind, self.pending_analysis = self.pending_analysis, None
                self.command('analyze', kind=kind)
            elif self.state and self.root.grab_current() is None:
                state = self.state
                live = not state['winner'] and not state['openingSetup'] and not state['replayMode']
                clock = state['clock']
                clock_ready = not clock['enabled'] or not clock['started'] or clock['active'] is not None
                if live and clock_ready and state['mode'] == 'human' and state['currentKind'] != 'human' and self.reply_pending:
                    self.reply_pending = False
                    self._submit('step', revision=state['revision'])
                elif time.monotonic() - self.last_clock_poll > .5 and live and clock['active'] is not None:
                    self.last_clock_poll = time.monotonic()
                    self._submit('state')
                elif time.monotonic() - self.last_catalog_poll > 5:
                    self.last_catalog_poll = time.monotonic()
                    self._submit('refresh')
        self._clock_display()
        self.root.after(50, self._poll)

    def _control_states(self):
        busy = self.future is not None or self.service is None or self.closing
        for widget in self.controls:
            widget.configure(state='disabled' if busy else 'normal')
        # 在模型计算时仍可请求暂停/分析，命令在这一手完成后串行执行。
        if self.future is not None and self.operation == 'step' and not self.closing:
            self.reply_button.configure(state='normal')
            self.analyze_button.configure(state='normal')
        self.reply_button.configure(text='暂停模型应手' if self.future is not None and self.operation == 'step' else '继续模型应手')
        if self.config and not self.config['modelAvailable']:
            for widget in self.ai_controls:
                widget.configure(state='disabled')
        if self.state and self.state['mode'] == 'pvp':
            self.reply_button.configure(state='disabled')
            self.step_button.configure(state='disabled')
        if self.competition:
            for widget in self.model_controls:
                widget.configure(state='disabled')

    def _options(self):
        seconds = float(self.move_seconds.get())
        if not math.isfinite(seconds):
            raise ValueError('单步秒数必须是有限数值')
        return dict(mode='pvp' if self.mode.get() == '人人对弈' else 'human',
                    mine=1 if self.seat.get() == '先手红方' else -1,
                    simulations=int(self.simulations.get()), seed=int(self.seed.get()),
                    move_time_ms=round(seconds * 1000), clock_minutes=int(self.clock_minutes.get()))

    def new_game(self, manual):
        try:
            self.command('new', manual_opening=manual, **self._options())
        except ValueError as exc:
            messagebox.showerror('设置无效', str(exc), parent=self.root)

    def toggle_reply(self):
        """人机对弈暂停/恢复当前模型回应，不再提供 AI 对 AI 自动对战。"""
        if not self.state or self.state['mode'] != 'human':
            return
        if self.future is not None:
            self.reply_pending = False
            self.status.set('正在计算的一手结束后暂停。')
        else:
            self.reply_pending = True
        self._control_states()

    def analyze(self):
        self.reply_pending = False
        if self.future is not None and self.operation == 'step':
            self.pending_analysis = 'trained'
            self.status.set('当前落子完成后暂停并分析。')
        else:
            self.command('analyze', kind='trained')

    def toggle_clock(self):
        if self.state:
            self.command('clock', running=self.state['clock']['active'] is None)

    def render(self):
        state, config = self.state, self.config
        if state['winner']:
            title = ('红方' if state['winner'] == 1 else '蓝方') + ('获胜（对方超时）' if state['winnerReason'] == 'timeout' else '获胜 · 已连接目标边')
            self.reply_pending = False
        elif state['replayMode']:
            title = f"打谱 · 第 {state['replayCursor']} / {state['replayTotal']} 手"
        else:
            title = f"{'手动开局 · ' if state['openingSetup'] else ''}轮到{'红方' if state['currentPlayer'] == 1 else '蓝方'} · {ACTORS[state['currentKind']]}"
        self.turn.set(title)
        self.threads.set(str(config['inference']['threads']))
        trained = config['models']['trained']
        self.mode_combo.configure(values=('人人对弈', '人机对弈') if trained else ('人人对弈',))
        if trained:
            iteration = trained['metadata'].get('iteration', '未知')
            self.model_info.set(f"{'比赛版锁定' if self.competition else '模型'}：第 {iteration} 轮 · FP32 · {self.snapshot['search_backend']} 搜索 · {config['inference']['threads']} 线程\n"
                                f"本局上限 {state['mctsSims']} 次 / 手")
        else:
            self.mode.set('人人对弈')
            self.model_info.set('未加载模型 · 当前可人人对弈、打谱和导出棋谱。\n放入模型并在设置中加载后，可启用人机对弈与分析。')
        catalog = config.get('modelLibrary') or {}
        self.model_entries = [entry for entry in catalog.get('entries', []) if entry['available']]
        labels = [entry['label'] for entry in self.model_entries]
        self.model_combo.configure(values=labels)
        if self.model_choice.get() not in labels:
            selected = next((entry['label'] for entry in self.model_entries if entry['id'] == catalog.get('selectedId')), '')
            self.model_choice.set(selected)
        # 时钟轮询不能重建树控件，否则会丢失用户选择和滚动位置。
        key = (state['id'], state['revision'], state['replayCursor'])
        if key != self.history_key:
            self.history_key = key
            self.history.delete(*self.history.get_children())
            for move in state['replayMoves'] if state['replayMode'] else state['moves']:
                self.history.insert('', 'end', iid=str(move['move']), values=(move['move'], '红' if move['player'] == 1 else '蓝',
                    coordinate(move['row'], move['col'], state['boardSize']), '手动' if move.get('source') == 'manual_opening' else ACTORS.get(move.get('actor'), '棋谱'), move.get('mark', '')))
            if state['moves']:
                self.history.selection_set(str(len(state['moves'])))
                self.history.see(str(len(state['moves'])))
            self.comment_text.set((state['moves'][-1].get('comment', '')[:500] if state['moves'] else '') or '双击棋谱中的一手查看局面。')
            self.clear_preview()
        analysis = state.get('analysis')
        analysis_key = (state['id'], analysis['revision'], analysis['engine']) if analysis else None
        if analysis_key != self.analysis_key:
            self.analysis_key = analysis_key
            self.candidates.delete(*self.candidates.get_children())
            self.preview_scale.configure(to=0)
            self.pv_text.set('选择候选可预览已搜索的后续变化。' if analysis else '点击“分析当前局面”获取三个候选。')
            if analysis:
                for index, candidate in enumerate(analysis['candidates']):
                    self.candidates.insert('', 'end', iid=str(index), values=(coordinate(candidate['row'], candidate['col'], state['boardSize']),
                        candidate['visits'], score_label(candidate['value'])))
        self.draw_board()

    def _clock_display(self):
        if not self.state:
            return
        clock = self.state['clock']
        if not clock['enabled']:
            self.clock_text.set('棋钟未启用')
            return
        values = [clock['redMs'], clock['blueMs']]
        for index, side in enumerate((1, -1)):
            if clock['active'] == side:
                values[index] -= (time.monotonic() - self.snapshot['sampled_at']) * 1000
        text = []
        for milliseconds in values:
            seconds = max(0, math.ceil(milliseconds / 1000))
            text.append(f'{seconds // 60:02d}:{seconds % 60:02d}')
        self.clock_text.set(f'红 {text[0]}     蓝 {text[1]}' + ('  暂停' if clock['started'] and clock['active'] is None else ''))

    def draw_board(self):
        """屏幕几何只影响显示，棋盘拓扑和模型输入由 HexGame 定义。"""
        canvas = self.canvas
        canvas.delete('all')
        self.centers.clear()
        if not self.state:
            canvas.create_text(250, 200, text='正在准备模型…', fill='#52665c')
            return
        state, n = self.state, self.state['boardSize']
        width, height = max(canvas.winfo_width(), 200), max(canvas.winfo_height(), 200)
        radius = min((width - 70) / (math.sqrt(3) * (1.5 * (n - 1) + 1)), (height - 80) / (1.5 * (n - 1) + 2))
        dx, dy = math.sqrt(3) * radius, 1.5 * radius
        x0 = (width - dx * 1.5 * (n - 1)) / 2
        y0 = (height - dy * (n - 1)) / 2
        for row in range(n):
            for col in range(n):
                self.centers[(row, col)] = (x0 + dx * (col + row / 2), y0 + dy * row)
        # 四条目标边：红色水平边、蓝色左右斜边；颜色不随模型角色交换。
        for cells, color, offset in [([(0, 0), (0, n-1)], RED, (0, -radius-6)),
                                     ([(n-1, 0), (n-1, n-1)], RED, (0, radius+6)),
                                     ([(0, 0), (n-1, 0)], BLUE, (-radius-8, 0)),
                                     ([(0, n-1), (n-1, n-1)], BLUE, (radius+8, 0))]:
            points = [v for cell in cells for v in (self.centers[cell][0]+offset[0], self.centers[cell][1]+offset[1])]
            canvas.create_line(*points, fill=color, width=5, capstyle='round')
        move_numbers = {(m['row'], m['col']): str(m['move']) for m in state['moves']}
        preview = {(m['row'], m['col']): (m['player'], '+'+str(i+1)) for i, m in enumerate(self.preview)}
        for (row, col), (x, y) in self.centers.items():
            player = state['board'][row][col]
            is_preview = (row, col) in preview
            label = move_numbers.get((row, col), '')
            if is_preview:
                player, label = preview[(row, col)]
            fill = RED if player == 1 else BLUE if player == -1 else '#ffffff'
            points = [v for index in range(6) for v in (x + (radius-1.5)*math.cos(math.radians(30+60*index)), y + (radius-1.5)*math.sin(math.radians(30+60*index)))]
            canvas.create_polygon(*points, fill=fill, outline='#2d5144' if is_preview else '#c5cec5', width=2 if is_preview else 1, dash=(3, 2) if is_preview else ())
            if label:
                canvas.create_text(x, y, text=label, fill='white', font=('sans-serif', max(8, round(radius*.40)), 'bold'))
        for col in range(n):
            x, y = self.centers[(n-1, col)]
            canvas.create_text(x, y+radius+22, text=chr(65+col), fill='#4c5b52')
        for row in range(n):
            x, y = self.centers[(row, 0)]
            canvas.create_text(x-radius-25, y, text=str(n-row), fill='#4c5b52')
        path = state.get('winningPath', [])
        if len(path) > 1:
            canvas.create_line(*[v for cell in path for v in self.centers[tuple(cell)]], fill='#f2d47e', width=4)

    def board_click(self, event):
        if not self.state or self.future is not None or self.preview:
            return
        state = self.state
        if state['winner'] or state['replayMode'] or (not state['openingSetup'] and state['currentKind'] != 'human'):
            return
        # 最近中心之外再限制命中半径，避免点击棋盘外或格子间隙也落子。
        if not self.centers:
            return
        cell = min(self.centers, key=lambda c: (self.centers[c][0]-event.x)**2 + (self.centers[c][1]-event.y)**2)
        x, y = self.centers[cell]
        spacing = abs(self.centers[(0, 1)][0] - self.centers[(0, 0)][0])
        if math.hypot(x-event.x, y-event.y) > spacing * .48 or state['board'][cell[0]][cell[1]]:
            return
        self.command('move', row=cell[0], col=cell[1])

    def selected_candidate(self):
        selection = self.candidates.selection()
        if not selection or not self.state or not self.state.get('analysis'):
            return None
        return self.state['analysis']['candidates'][int(selection[0])]

    def select_candidate(self, event=None):
        candidate = self.selected_candidate()
        if candidate:
            self.preview_scale.configure(to=len(candidate['pv']))
            self.preview_depth.set(len(candidate['pv']))
            self.update_preview()

    def update_preview(self):
        candidate = self.selected_candidate()
        self.preview = candidate['pv'][:self.preview_depth.get()] if candidate else []
        if candidate:
            path = ' → '.join(('红 ' if move['player'] == 1 else '蓝 ') + coordinate(move['row'], move['col'], self.state['boardSize']) for move in candidate['pv'])
            self.pv_text.set(path + '\n只显示已搜索分支；未搜索处不补造后续。')
        self.draw_board()

    def clear_preview(self):
        self.preview = []
        self.preview_depth.set(0)
        self.draw_board()

    def play_candidate(self):
        candidate = self.selected_candidate()
        if candidate:
            self.command('move', row=candidate['row'], col=candidate['col'])

    def history_seek(self, event=None):
        selection = self.history.selection()
        if selection:
            self.command('seek', ply=int(selection[0]))

    def seek_relative(self, change):
        if self.state:
            self.command('seek', ply=max(0, min(self.state['replayTotal'], self.state['replayCursor']+change)))

    def import_record(self):
        path = filedialog.askopenfilename(parent=self.root, filetypes=[('Hex 棋谱', '*.txt *.json')])
        if path:
            try:
                self.command('import', path=path, **self._options())
            except ValueError as exc:
                messagebox.showerror('设置无效', str(exc), parent=self.root)

    def export_record(self, kind):
        initial = 'hex-match.json'
        if kind == 'txt' and self.state:
            # 用只读快照生成比赛规定的默认文件名；实际导出仍在工作线程完成。
            from hex.record_format import export_txt
            try:
                initial, _ = export_txt(self.state['replayMoves'] if self.state['replayMode'] else self.state['moves'],
                                        self.state['recordMetadata'])
            except (ValueError, UnicodeError) as exc:
                messagebox.showerror('不能导出比赛 TXT', str(exc), parent=self.root)
                return
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension='.'+kind,
                                          initialfile=initial, filetypes=[('Hex 棋谱', '*.'+kind)])
        if path:
            self.command('export', path=path)

    def edit_metadata(self):
        if not self.state:
            return
        self.reply_pending = False
        dialog = tk.Toplevel(self.root)
        dialog.title('队名、赛果与赛事信息')
        dialog.transient(self.root)
        variables = {}
        for row, (key, title) in enumerate((('redTeam', '先手红方队名'), ('blueTeam', '后手蓝方队名'), ('result', '赛果'), ('timePlace', '时间地点'), ('event', '赛事'))):
            ttk.Label(dialog, text=title).grid(row=row, column=0, padx=10, pady=8)
            value = tk.StringVar(value=self.state['recordMetadata'].get(key, ''))
            variables[key] = value
            widget = ttk.Combobox(dialog, textvariable=value, values=('', '先手胜', '后手胜'), state='readonly') if key == 'result' else ttk.Entry(dialog, textvariable=value, width=42)
            widget.grid(row=row, column=1, padx=10, pady=8)
        def save():
            self.command('metadata', metadata={key: var.get() for key, var in variables.items()})
            dialog.destroy()
        ttk.Button(dialog, text='保存', command=save).grid(row=5, column=1, pady=12)
        dialog.grab_set()

    def edit_annotation(self):
        if not self.state or not self.state['moves']:
            return
        self.reply_pending = False
        move = self.state['moves'][-1]
        dialog = tk.Toplevel(self.root)
        dialog.title(f"第 {move['move']} 手评注")
        dialog.transient(self.root)
        mark = tk.StringVar(value=str(move.get('mark', '')))
        ttk.Label(dialog, text='MARK −2 大劣 / −1 小劣 / 0 一般 / +1 小好 / +2 大好').pack(padx=12, pady=8)
        ttk.Combobox(dialog, textvariable=mark, values=('', '-2', '-1', '0', '1', '2'), state='readonly').pack(fill='x', padx=12)
        comment = tk.Text(dialog, width=60, height=12, wrap='word')
        comment.pack(padx=12, pady=12)
        comment.insert('1.0', move.get('comment', ''))
        def save():
            self.command('annotate', mark=int(mark.get()) if mark.get() else None, comment=comment.get('1.0', 'end-1c'))
            dialog.destroy()
        ttk.Button(dialog, text='保存评注', command=save).pack(pady=10)
        dialog.grab_set()

    def change_model(self):
        entry = next((entry for entry in self.model_entries if entry['label'] == self.model_choice.get()), None)
        if entry:
            self.command('model', ident=entry['id'])

    def choose_models(self):
        if self.competition:
            self.status.set('比赛版已锁定 3250 模型。')
            return
        if self.future is not None:
            return
        path = filedialog.askdirectory(parent=self.root, initialdir=str(self.models))
        if path:
            self.models = Path(path)
            self.reply_pending = False
            self.requested_model = None
            # 在线程外读取 Tk 变量，下面的闭包只使用普通整数。
            threads = int(self.threads.get())
            def reload_safe():
                if self.service is not None:
                    self.service.perform('close')
                self.service = None
                from hex.desktop_service import DesktopService
                self.service = DesktopService(self.models, self.data, threads=threads)
                return self.service.snapshot()
            self.future = self.executor.submit(reload_safe)
            self.operation = 'load'
            self._control_states()

    def close(self):
        """关闭时等待正在执行的原生推理结束，保存棋局后再释放搜索树。"""
        self.reply_pending = False
        self.pending_analysis = None
        self.closing = True
        self.status.set('正在等待当前操作结束并保存棋局…')
        self._control_states()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_models = ROOT / 'models/local_play/onnx'
    parser.add_argument('--models', type=Path, default=default_models if default_models.is_dir() else ROOT / 'models')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'runs/desktop')
    parser.add_argument('--model', help='models 内的相对路径，例如 iteration_3250/trained.onnx')
    parser.add_argument('--threads', type=int, choices=range(4, 17))
    parser.add_argument('--competition', action='store_true', help='比赛模式：锁定指定 3250 权重并校验内容')
    args = parser.parse_args(argv)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.data_dir / 'desktop.log', level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s', encoding='utf-8')
    root = tk.Tk()
    DesktopWindow(root, args.models, args.data_dir, args.model, args.threads, args.competition)
    root.mainloop()


if __name__ == '__main__':
    main()

const $ = id => document.getElementById(id);
const searchPresets = [800, 1200, 2000, 3200, 6400, 12800, 51200];
function selectSearchPreset(value) {
  $('mctsSims').value = String(searchPresets.includes(Number(value)) ? Number(value) : 3200);
}
let state = null, config = null, busy = false, autoplay = false, timer = null;
let busyTask = null, analysisKey = null, previewIndex = null, previewDepth = 0;
let analysisPaused = false, pendingAnalysis = null;
let metadataFormKey = null, annotationFormKey = null;
let modelCatalogBusy = false, clockPollBusy = false;
const clockPlayable = () => !state?.clock?.enabled || !state.clock.started || state.clock.active !== null;
const stopReason = reason => ({time_budget:'达到时间预算',node_limit:'达到内存预算',simulation_limit:'达到次数上限'}[reason] || '搜索结束');
const names = {human: '你', trained: '训练模型', azalea: 'Azalea 参考模型', record: '棋谱'};
const svgNS = 'http://www.w3.org/2000/svg';
const coord = (row, col) => `${String.fromCharCode(65 + col)}${(state?.boardSize || 11) - row}`;
const moveName = move => move.source === 'manual_opening' ? '手动开局' : names[move.actor];
const valueText = value => value === null ? '未搜索' : `${value >= 0 ? '+' : ''}${value.toFixed(3)}`;
const markText = value => value === null ? '未评价' : value < -.6 ? '−2 大劣' : value < -.2 ? '−1 小劣' : value <= .2 ? '0 一般' : value <= .6 ? '+1 小好' : '+2 大好';
const previewMoves = () => state?.analysis?.candidates[previewIndex]?.pv.slice(0, previewDepth) || [];
function svg(tag, attrs, text) {
  const el = document.createElementNS(svgNS, tag);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
  if (text !== undefined) el.textContent = text;
  return el;
}
function drawBoard() {
  if (!state) return;
  const n = state.boardSize, r = 21, dx = Math.sqrt(3) * 23, dy = 34.5, pad = 52;
  const center = (row, col) => [pad + dx * (col + row / 2), pad + dy * row];
  const board = $('board');
  board.setAttribute('viewBox', `0 0 ${2 * pad + 1.5 * dx * (n - 1)} ${2 * pad + dy * (n - 1)}`);
  board.replaceChildren();
  const [x0, y0] = center(0, 0), [xt, yt] = center(0, n - 1);
  const [xb, yb] = center(n - 1, 0), [xe, ye] = center(n - 1, n - 1);
  for (const [color, a, b] of [
    ['red', [x0 - 8, y0 - 29], [xt + 8, yt - 29]], ['red', [xb - 8, yb + 29], [xe + 8, ye + 29]],
    ['blue', [x0 - 28, y0], [xb - 28, yb]], ['blue', [xt + 28, yt], [xe + 28, ye]],
  ]) board.append(svg('line', {class: `edge ${color}`, x1: a[0], y1: a[1], x2: b[0], y2: b[1]}));
  for (let i = 0; i < n; i++) {
    const [x, y] = center(n - 1, i), [rx, ry] = center(i, 0);
    board.append(svg('text', {class: 'coord-label', x, y: y + 42}, String.fromCharCode(65 + i)));
    board.append(svg('text', {class: 'coord-label', x: rx - 43, y: ry}, n - i));
  }
  const last = state.moves.at(-1), moveIds = new Map(state.moves.map(m => [m.action, m.move]));
  const preview = previewMoves(), ghosts = new Map(preview.map((m, i) => [m.action, {...m, index:i + 1}]));
  const recommendations = new Map((state.analysis?.candidates || []).map(c => [c.action, c.rank]));
  for (let row = 0; row < n; row++) for (let col = 0; col < n; col++) {
    const ghost = ghosts.get(row * n + col), rank = recommendations.get(row * n + col);
    const value = ghost ? ghost.player : state.board[row][col], [x, y] = center(row, col);
    const lastMove = last && last.row === row && last.col === col;
    const cls = value === 1 ? 'red' : value === -1 ? 'blue' : 'empty';
    const playable = (state.openingSetup || clockPlayable()) && !busy && !state.winner && !state.replayMode && !preview.length && (state.openingSetup || state.currentKind === 'human') && !value;
    const cell = svg('g', {class: `cell ${cls}${lastMove ? ' last' : ''}${ghost ? ' preview-stone' : ''}${!preview.length && rank ? ' recommended' : ''}`, 'data-row': row, 'data-col': col,
      role: 'button', tabindex: playable ? 0 : -1, 'aria-disabled': String(!playable),
      'aria-label': `${coord(row,col)} ${value === 1 ? '红棋' : value === -1 ? '蓝棋' : '空位'}`});
    const points = Array.from({length: 6}, (_, i) => `${x + r * Math.cos(Math.PI/6 + i*Math.PI/3)},${y + r * Math.sin(Math.PI/6 + i*Math.PI/3)}`).join(' ');
    cell.append(svg('polygon', {class: 'hex', points}));
    cell.append(svg('text', {x, y}, ghost ? `+${ghost.index}` : value ? ($('showNumbers').checked ? moveIds.get(row*n+col) : '•') : !preview.length && rank ? ['①','②','③'][rank - 1] : '·'));
    cell.addEventListener('click', () => playMove(row, col));
    cell.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); playMove(row,col); } });
    board.append(cell);
  }
  if (state.winningPath.length) board.append(svg('polyline', {class: 'winning-line', points: state.winningPath.map(([r,c]) => center(r,c).join(',')).join(' ')}));
}
function updateControls() {
  const stopped = !state || !!state.winner;
  $('cpuThreads').disabled = busy || autoplay;
  $('applyThreads').disabled = busy || autoplay;
  $('modelSelect').disabled = $('refreshModels').disabled = busy || autoplay || modelCatalogBusy;
  const candidate = config?.modelLibrary?.entries.find(e => e.id === $('modelSelect').value);
  $('applyModel').disabled = busy || autoplay || modelCatalogBusy || !candidate?.available ||
    (candidate.id === config.modelLibrary.selectedId && candidate.sha256 === config.modelLibrary.activeSha256);
  for (const id of ['importRecord','exportTxt','saveRecordMetadata','redTeam','blueTeam','recordResult','timePlace','eventName']) $(id).disabled = busy || autoplay;
  $('recordFile').disabled = busy || autoplay;
  const cursor = state?.replayCursor || 0, total = state?.replayTotal || 0;
  $('replayFirst').disabled = $('replayPrevious').disabled = busy || autoplay || cursor === 0;
  $('replayNext').disabled = $('replayLast').disabled = busy || autoplay || cursor === total;
  $('replaySlider').disabled = busy || autoplay || total === 0;
  $('continueRecord').hidden = !state?.replayMode;
  $('continueRecord').disabled = busy || stopped;
  for (const id of ['moveMark','moveComment','saveAnnotation']) $(id).disabled = busy || autoplay || !cursor;
  $('generateAnnotation').disabled = busy || autoplay || !cursor || !state?.analysis;
  for (const id of ['newGame','newManualGame','mode','myPlayer','mctsSims','seed','moveTimeMs','clockMinutes']) $(id).disabled = busy || autoplay;
  $('undoGame').disabled = busy || !state?.canUndo;
  $('saveGame').disabled = busy || !state;
  $('stepGame').disabled = busy || autoplay || stopped || !clockPlayable() || state?.openingSetup || state?.replayMode;
  $('autoGame').disabled = stopped || !clockPlayable() || state?.openingSetup || state?.replayMode || (busy && !autoplay);
  $('autoGame').textContent = autoplay ? '暂停对战' : '自动对战';
  $('watchControls').hidden = state?.mode !== 'azalea' || state?.openingSetup || state?.replayMode;
  $('openingControls').hidden = !state?.openingSetup;
  $('finishOpening').disabled = busy || stopped;
  $('analysisEngine').disabled = busyTask === '/api/analyze' || pendingAnalysis !== null;
  $('analyzeGame').disabled = stopped || busyTask === '/api/analyze' || pendingAnalysis !== null || (busy && busyTask !== '/api/step');
  $('analyzeGame').textContent = pendingAnalysis !== null ? '等待本手结束后分析…' :
    busyTask === '/api/analyze' ? '正在分析…' : autoplay || busyTask === '/api/step' ? '暂停并分析' : '分析当前局面';
  $('resumePlay').hidden = !analysisPaused || stopped || state?.openingSetup || state?.replayMode || state?.currentKind === 'human';
  $('resumePlay').disabled = busy;
  document.body.classList.toggle('thinking', busy);
}
function selectPreview(index, depth) {
  previewIndex = index; previewDepth = depth; render();
  if (window.matchMedia('(max-width:680px)').matches) {
    $('previewBanner').scrollIntoView({behavior:'smooth', block:'start'});
  }
}
function renderAnalysis() {
  const analysis = state.analysis;
  const key = analysis ? `${state.id}:${analysis.revision}:${analysis.engine}` : null;
  if (key !== analysisKey) { previewIndex = null; previewDepth = 0; analysisKey = key; }
  const preview = previewMoves();
  $('previewBanner').hidden = !preview.length;
  $('previewLabel').textContent = `候选 ${(previewIndex ?? 0) + 1} · 未来 ${preview.length} 手预览（尚未落子）`;
  $('analysisTime').textContent = analysis ? `${analysis.seconds.toFixed(2)}s` : '';
  $('analysisSummary').textContent = analysis ? `${names[analysis.engine]} · ${analysis.player === 1 ? '红方' : '蓝方'}视角 · 上限 ${analysis.simulations} 次 · 实际 ${analysis.completedSimulations ?? '—'} 次` :
    state.winner ? '本局已结束。悔棋后可继续分析。' : '自动对战时可点击“暂停并分析”；手动开局和人类回合也可使用。';
  $('candidates').replaceChildren();
  for (const [index, candidate] of (analysis?.candidates || []).entries()) {
    const li = document.createElement('li'); li.className = 'candidate';
    const head = document.createElement('button'); head.className = 'candidate-head';
    head.setAttribute('aria-pressed', String(previewIndex === index)); head.disabled = busy || autoplay;
    const label = document.createElement('strong'); label.textContent = `${candidate.rank}. ${coord(candidate.row, candidate.col)}`;
    const score = document.createElement('span'); score.className = 'candidate-score'; score.textContent = `Q ${valueText(candidate.value)}`;
    head.append(label, score); head.title = '点击预览这条主变化';
    head.addEventListener('click', () => selectPreview(index, candidate.pv.length));
    const stats = document.createElement('p'); stats.className = 'candidate-stats';
    stats.textContent = `${markText(candidate.value)} · ${candidate.visits} 次访问 · 先验 ${(candidate.prior * 100).toFixed(1)}%`;
    const line = document.createElement('div'); line.className = 'pv-line';
    for (const [ply, move] of candidate.pv.entries()) {
      const chip = document.createElement('button'); chip.className = `pv-move ${move.player === 1 ? 'red' : 'blue'}`;
      chip.textContent = `${ply + 1} ${coord(move.row,move.col)}`; chip.disabled = busy || autoplay;
      chip.title = `预览至第 ${ply + 1} 手 · 原执棋方视角 Q ${valueText(move.value)}`;
      chip.addEventListener('click', () => selectPreview(index, ply + 1));
      line.append(chip);
    }
    const end = document.createElement('small'); end.className = 'pv-end';
    end.textContent = candidate.stopReason === 'terminal' ? `此续着以${candidate.winner === 1 ? '红方' : '蓝方'}获胜结束` :
      candidate.stopReason === 'unsearched' ? '后续尚未搜索到；可增加搜索次数' : '显示至第 6 手';
    li.append(head, stats, line, end);
    if (!busy && !autoplay && !state.winner && !state.replayMode && (state.openingSetup || state.currentKind === 'human')) {
      const play = document.createElement('button'); play.className = 'use-candidate'; play.textContent = `落下 ${coord(candidate.row,candidate.col)}`;
      play.addEventListener('click', () => { previewIndex = null; previewDepth = 0; playMove(candidate.row,candidate.col); });
      li.append(play);
    }
    $('candidates').append(li);
  }
}
function render() {
  if (!state) return;
  $('title').textContent = `Hex ${state.boardSize} × ${state.boardSize}`;
  for (const [player, color] of [[1,'red'],[-1,'blue']]) {
    $(color+'Name').textContent = state.replayMode ? state.recordMetadata[player === 1 ? 'redTeam' : 'blueTeam'] : names[player === state.myPlayer ? state.mode : 'trained'];
    $(color+'Player').classList.toggle('active', !state.winner && player === state.currentPlayer);
  }
  $('turnBadge').textContent = state.replayMode ? `打谱 ${state.replayCursor} / ${state.replayTotal}` : state.winner ? '对局结束' : `${state.openingSetup ? '手动开局 · ' : ''}第 ${state.moves.length + 1} 手`;
  $('status').textContent = state.winnerReason === 'timeout' ? `${state.winner === 1 ? '蓝方' : '红方'}超时 · ${state.winner === 1 ? '红方' : '蓝方'}获胜` : state.winner ? `${state.replayMode ? '棋谱' : state.openingSetup ? '手动开局' : names[state.winnerKind]}获胜 · ${state.winner === 1 ? '红方连通上下' : '蓝方连通左右'}` :
    busyTask === '/api/analyze' ? '正在分析当前局面…' :
    state.replayMode ? `打谱查看 · 第 ${state.replayCursor} 手 · 下一手${state.currentPlayer === 1 ? '红方' : '蓝方'}` :
    state.openingSetup ? `手动开局 · 请放${state.currentPlayer === 1 ? '红棋' : '蓝棋'}` :
    state.currentKind === 'human' ? '轮到你了，点击空位落子' : `${names[state.currentKind]}${busy ? '正在思考…' : '待落子'}`;
  const last = state.moves.at(-1);
  $('detail').textContent = state.winnerReason === 'timeout' ? '按棋钟判负；棋盘未必已有连通路径。棋谱已保存。' : state.winner ? (state.replayMode ? '棋谱中的获胜路径已标出。' : '获胜路径已标出，棋谱已保存在本机。') : last ?
    `上一手：${moveName(last)} · ${coord(last.row,last.col)}${last.seconds !== undefined ? ` · 思考 ${last.seconds.toFixed(2)} 秒` : ''}${last.searchSimulations !== undefined ? ` · 本手搜索 ${last.searchSimulations} 次（${stopReason(last.searchStopReason)}）` : ''}` :
    `己方执${state.myPlayer === 1 ? '红棋，连接上下' : '蓝棋，连接左右'}。先连通目标两侧即获胜。`;
  $('moveCount').textContent = `${state.moves.length} 手`;
  $('emptyHistory').hidden = state.replayTotal > 0;
  $('openingHint').textContent = `已手动落下 ${state.openingLength} 手。红、蓝交替，步数不限；完成后由所选双方接着下。`;
  $('moves').replaceChildren();
  for (const m of (state.replayMode ? state.replayMoves : state.moves)) {
    const li = document.createElement('li'); li.className = m.player === 1 ? 'red' : 'blue';
    li.classList.toggle('replay-current', state.replayMode && m.move === state.replayCursor);
    li.classList.toggle('replay-future', state.replayMode && m.move > state.replayCursor);
    li.setAttribute('role','button'); li.tabIndex = busy || autoplay ? -1 : 0;
    li.title = `查看第 ${m.move} 手后的局面`;
    li.addEventListener('click', () => seekRecord(m.move));
    li.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); seekRecord(m.move); } });
    for (const [className,text] of [['move-index',m.move],['move-dot',''],['move-coord',coord(m.row,m.col)],['move-who',moveName(m)],['move-time',m.seconds === undefined ? '' : `${m.seconds.toFixed(2)}s`]]) {
      const span = document.createElement('span'); span.className = className; span.textContent = text; li.append(span);
    }
    if (m.mark !== undefined) { const mark=document.createElement('small'); mark.textContent=`${m.mark > 0 ? '+' : ''}${m.mark}`; li.append(mark); }
    $('moves').append(li);
  }
  if (!state.replayMode) $('moves').scrollTop = $('moves').scrollHeight;
  renderRecord(); renderAnalysis(); updateControls(); drawBoard(); renderClock();
}
function recordMetadata() {
  return {redTeam:$('redTeam').value,blueTeam:$('blueTeam').value,result:$('recordResult').value,timePlace:$('timePlace').value,event:$('eventName').value};
}
function renderRecord() {
  const key = `${state.id}:${JSON.stringify(state.recordMetadata)}`;
  if (metadataFormKey !== key) {
    metadataFormKey = key;
    const meta = state.recordMetadata;
    $('redTeam').value=meta.redTeam; $('blueTeam').value=meta.blueTeam; $('recordResult').value=meta.result;
    $('timePlace').value=meta.timePlace; $('eventName').value=meta.event;
  }
  if (state.winner && !$('recordResult').value) $('recordResult').value = state.winner === 1 ? '先手胜' : '后手胜';
  $('recordWarning').textContent = (state.recordWarnings || []).join('\n');
  $('replayPosition').textContent = `${state.replayCursor} / ${state.replayTotal} 手`;
  $('replaySlider').max=state.replayTotal; $('replaySlider').value=state.replayCursor;
  const move = state.moves.at(-1);
  const annotationKey = `${state.id}:${state.replayCursor}:${move?.mark}:${move?.comment}`;
  if (annotationFormKey !== annotationKey) {
    annotationFormKey=annotationKey; $('moveMark').value=move?.mark ?? ''; $('moveComment').value=move?.comment || '';
  }
  $('annotationTitle').textContent = move ? `第 ${move.move} 手 ${move.player === 1 ? 'R' : 'B'}(${coord(move.row,move.col)}) · 评注` : '逐手评注（先选择一手）';
}
async function request(path, body) {
  const res = await fetch(path, body === undefined ? {cache:'no-store'} : {method:'POST', cache:'no-store', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || '请求失败');
  if (data.clock) data.clockReceivedAt = performance.now();
  if (data.session?.clock) data.session.clockReceivedAt = performance.now();
  return data;
}
function schedule() {
  clearTimeout(timer);
  if (!busy && !analysisPaused && state && !state.replayMode && !state.winner && !state.openingSetup && clockPlayable() && state.currentKind !== 'human' && (state.mode === 'human' || autoplay)) {
    timer = setTimeout(() => action('/api/step'), 220);
  }
}
async function action(path, extra = {}) {
  if (busy) return;
  let succeeded = false;
  busy = true; busyTask = path; $('notice').textContent = ''; render();
  try {
    state = await request(path, {sessionId:state?.id, revision:state?.revision, ...extra});
    sessionStorage.setItem('hex-session', state.id);
    if (state.winner || state.openingSetup || state.replayMode) autoplay = false;
    succeeded = true;
  } catch (error) {
    autoplay = false;
    if (state) { try { state = await request(`/api/session?id=${state.id}`); } catch (_) {} }
    $('notice').textContent = error.message;
  } finally {
    busy = false; busyTask = null; render();
    if (pendingAnalysis !== null) {
      const engine = pendingAnalysis; pendingAnalysis = null;
      if (succeeded && state && !state.winner) await action('/api/analyze', {engine});
      else render();
    } else if (succeeded) {
      if (path === '/api/analyze') $('analysisCard').scrollIntoView({block:'start'});
      schedule();
    }
  }
  return succeeded;
}
function playMove(row,col) {
  if (!state || (!state.openingSetup && !clockPlayable()) || busy || state.winner || state.replayMode || previewMoves().length || (!state.openingSetup && clockPlayable() && state.currentKind !== 'human') || state.board[row][col]) return;
  analysisPaused = false;
  action('/api/move', {row,col});
}
function startNewGame(manualOpening) {
  const mctsSims = Number($('mctsSims').value), seed = Number($('seed').value);
  if (!searchPresets.includes(mctsSims) || !Number.isInteger(seed) || seed < 0 || seed > 4294967295) {
    $('notice').textContent = '请选择搜索次数档位，并输入有效的整数随机种子。'; return;
  }
  autoplay = false; analysisPaused = false; pendingAnalysis = null; clearTimeout(timer);
  action('/api/new', {mode:$('mode').value, myPlayer:Number($('myPlayer').value), mctsSims, seed, manualOpening, moveTimeMs:Number($('moveTimeMs').value),clockMinutes:Number($('clockMinutes').value)});
}
$('newGame').addEventListener('click', () => startNewGame(false));
$('newManualGame').addEventListener('click', () => startNewGame(true));
$('finishOpening').addEventListener('click', () => { analysisPaused = false; autoplay = state?.mode === 'azalea'; clearTimeout(timer); action('/api/start'); });
$('analyzeGame').addEventListener('click', () => {
  if (!state || state.winner || (busy && busyTask !== '/api/step')) return;
  autoplay = false; analysisPaused = true; clearTimeout(timer);
  const engine = $('analysisEngine').value;
  if (busy) { pendingAnalysis = engine; render(); }
  else action('/api/analyze', {engine});
});
$('resumePlay').addEventListener('click', () => { analysisPaused = false; autoplay = state?.mode === 'azalea'; render(); schedule(); });
$('exitPreview').addEventListener('click', () => { previewIndex = null; previewDepth = 0; render(); });
$('stepGame').addEventListener('click', () => { analysisPaused = false; action('/api/step'); });
$('autoGame').addEventListener('click', () => { analysisPaused = false; autoplay = !autoplay; render(); schedule(); });
$('undoGame').addEventListener('click', () => { autoplay = false; analysisPaused = false; clearTimeout(timer); action('/api/undo'); });
$('saveGame').addEventListener('click', async () => {
  if (!await action('/api/record-metadata', {metadata:recordMetadata()})) return;
  await action('/api/save');
  if (!state) return;
  const link = document.createElement('a'); link.href = `/api/export?id=${state.id}`;
  link.download = `hex-${state.id}.json`; link.click();
});
$('showNumbers').addEventListener('change', drawBoard);
function seekRecord(ply) {
  if (busy || autoplay || !state) return;
  analysisPaused=true;clearTimeout(timer);action('/api/seek',{ply});
}
$('replayFirst').addEventListener('click',()=>seekRecord(0));
$('replayPrevious').addEventListener('click',()=>seekRecord(Math.max(0,state.replayCursor-1)));
$('replayNext').addEventListener('click',()=>seekRecord(Math.min(state.replayTotal,state.replayCursor+1)));
$('replayLast').addEventListener('click',()=>seekRecord(state.replayTotal));
$('replaySlider').addEventListener('change',()=>seekRecord(Number($('replaySlider').value)));
$('continueRecord').addEventListener('click',()=>{analysisPaused=false;action('/api/continue-record');});
$('saveRecordMetadata').addEventListener('click',()=>action('/api/record-metadata',{metadata:recordMetadata()}));
$('saveAnnotation').addEventListener('click',()=>action('/api/annotate',{mark:$('moveMark').value===''?null:Number($('moveMark').value),comment:$('moveComment').value}));
$('generateAnnotation').addEventListener('click',async()=>{if(await action('/api/annotate',{generated:true})) $('annotationPanel').open=true;});
$('importRecord').addEventListener('click',()=>$('recordFile').click());
$('recordFile').addEventListener('change',async()=>{
  const file=$('recordFile').files[0];if(!file)return;
  try {
    const format=file.name.toLowerCase().endsWith('.json')?'json':'txt';
    const limit=format==='json'?8*1024*1024:262144;
    if(file.size>limit)throw new Error(format==='json'?'详细 JSON 棋谱不能超过 8 MB':'TXT 棋谱不能超过 256 KB');
    const bytes=new Uint8Array(await file.arrayBuffer());let binary='';
    for(let i=0;i<bytes.length;i+=8192)binary+=String.fromCharCode(...bytes.subarray(i,i+8192));
    autoplay=false;analysisPaused=true;pendingAnalysis=null;clearTimeout(timer);
    await action('/api/import',{format,data:btoa(binary),mode:$('mode').value,myPlayer:Number($('myPlayer').value),mctsSims:Number($('mctsSims').value),seed:Number($('seed').value),moveTimeMs:Number($('moveTimeMs').value),clockMinutes:Number($('clockMinutes').value)});
  }catch(error){$('notice').textContent=error.message;}
  finally{$('recordFile').value='';}
});
$('exportTxt').addEventListener('click',async()=>{
  if(!await action('/api/record-metadata',{metadata:recordMetadata()}))return;
  try {
    const response=await fetch(`/api/export-txt?id=${state.id}`,{cache:'no-store'});
    if(!response.ok)throw new Error((await response.json()).error);
    const match=/filename\*=UTF-8''([^;]+)/i.exec(response.headers.get('Content-Disposition')||'');
    const url=URL.createObjectURL(await response.blob()),a=document.createElement('a');
    a.href=url;a.download=match?decodeURIComponent(match[1]):'hex-match.txt';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(error){$('notice').textContent=error.message;}
});
function renderThreads() {
  const info = config?.inference;
  $('cpuControls').hidden = !info?.configurable;
  if (!info?.configurable) return;
  $('cpuThreads').replaceChildren();
  for (let n = info.minThreads; n <= info.maxThreads; n++) {
    const option = document.createElement('option'); option.value = n; option.textContent = `${n} 线程`;
    $('cpuThreads').append(option);
  }
  $('cpuThreads').value = Math.max(info.minThreads, info.threads);
  $('threadStatus').textContent = `当前 ${info.threads} 线程 · 优化 CPU 推理`;
}
function renderModelInfo() {
  $('device').textContent = config.device.toUpperCase();
  $('trainedName').textContent = config.models.trained.name;
  $('trainedName').title = config.models.trained.path;
  const meta = config.models.trained.metadata;
  const teacher = meta.run_config?.teacherFraction || 0;
  $('modelDetail').textContent = `${config.boardSize} × ${config.boardSize}${meta.iteration !== undefined ? ` · 第 ${meta.iteration} 轮后的权重` : ''}${teacher ? ` · 教师 ${(teacher*100).toFixed(0)}%` : ''} · 本机独立推理`;
  $('referenceName').textContent = config.models.azalea?.name || '未加载';
  $('referenceName').title = config.referenceError || config.models.azalea?.path || '';
  $('mode').querySelector('option[value=azalea]').disabled = !config.referenceAvailable;
  $('analysisEngine').querySelector('option[value=azalea]').disabled = !config.referenceAvailable;
}
function renderModelLibrary() {
  const library = config?.modelLibrary;
  $('modelControls').hidden = !library;
  if (!library) return;
  const previous = $('modelSelect').value || library.selectedId;
  $('modelSelect').replaceChildren();
  for (const entry of library.entries) {
    if (!entry.available) continue;
    const option = document.createElement('option');option.value=entry.id;option.textContent=entry.label;
    $('modelSelect').append(option);
  }
  $('modelSelect').value = library.entries.some(e=>e.available && e.id===previous) ? previous : library.selectedId;
  $('modelErrors').replaceChildren();
  for (const entry of library.entries.filter(e=>!e.available)) {
    const item=document.createElement('li');item.textContent=`${entry.id}：${entry.error}`;$('modelErrors').append(item);
  }
  $('modelProblems').hidden = !$('modelErrors').children.length;
  const active=library.entries.find(e=>e.id===library.selectedId);
  $('modelStatus').textContent = !active?.available ? '当前模型文件已移走；已加载的对局仍可继续，请选择有效模型。' :
    active.sha256!==library.activeSha256 ? '当前模型文件已更新；点击“切换并新开局”加载新权重。' :
    `已发现 ${library.entries.filter(e=>e.available).length} 个可用模型`;
  $('modelStatus').title=library.directory;
  updateControls();
}
async function refreshModelLibrary() {
  if (busy || autoplay || modelCatalogBusy || !config?.modelLibrary) return;
  modelCatalogBusy=true;updateControls();
  try {
    const library=await request('/api/models');
    if (library.selectedId!==config.modelLibrary.selectedId || library.activeSha256!==config.modelLibrary.activeSha256) {
      // Another window changed the global engine; its switch saved the old games.
      location.reload();return;
    }
    config.modelLibrary=library;renderModelLibrary();
  } catch(error) { $('modelStatus').textContent=error.message; }
  finally {modelCatalogBusy=false;updateControls();}
}
$('refreshModels').addEventListener('click',refreshModelLibrary);
$('modelSelect').addEventListener('change',updateControls);
$('applyModel').addEventListener('click',async()=>{
  if(busy || autoplay || modelCatalogBusy)return;
  clearTimeout(timer);analysisPaused=true;pendingAnalysis=null;busy=true;busyTask='/api/select-model';render();
  $('modelStatus').textContent='正在校验并加载模型…';
  try {
    const result=await request('/api/select-model',{modelId:$('modelSelect').value,mode:$('mode').value,
      myPlayer:Number($('myPlayer').value),mctsSims:Number($('mctsSims').value),seed:Number($('seed').value),moveTimeMs:Number($('moveTimeMs').value),clockMinutes:Number($('clockMinutes').value)});
    config=result.config;state=result.session;sessionStorage.setItem('hex-session',state.id);
    analysisPaused=false;autoplay=false;previewIndex=null;previewDepth=0;
    $('modelSelect').value=config.modelLibrary.selectedId;
    renderModelInfo();renderModelLibrary();renderThreads();$('notice').textContent='模型已切换；上一局棋谱已保存在 games 文件夹。';
  } catch(error) {$('modelStatus').textContent=error.message;}
  finally {busy=false;busyTask=null;render();schedule();}
});
setInterval(()=>{if(!document.hidden)refreshModelLibrary();},5000);
$('applyThreads').addEventListener('click', async () => {
  if (busy || autoplay) return;
  clearTimeout(timer); busy = true; busyTask = '/api/threads'; updateControls();
  $('threadStatus').textContent = '正在应用…';
  try {
    config = await request('/api/threads', {threads:Number($('cpuThreads').value)});
    renderThreads();
  } catch (error) { $('threadStatus').textContent = error.message; }
  finally { busy = false; busyTask = null; render(); schedule(); }
});
(async () => {
  try {
    config = await request('/api/config');
    renderThreads();
    renderModelInfo();renderModelLibrary();
    selectSearchPreset(config.defaultSimulations);
    const id = sessionStorage.getItem('hex-session');
    if (id) { try { state = await request(`/api/session?id=${id}`); } catch (_) {} }
    if (state) {
      $('mode').value = state.mode; $('myPlayer').value = state.myPlayer;
      // A legacy session can keep its exact budget; this control configures the next game.
      selectSearchPreset(state.mctsSims); $('seed').value = state.seed;
      $('moveTimeMs').value = state.moveTimeMs || 0;
      $('clockMinutes').value = state.clock?.minutes || 0;
      analysisPaused = !!state.analysis;
      render(); schedule();
    } else await action('/api/new', {mode:'human', myPlayer:1, mctsSims:Number($('mctsSims').value), seed:0});
  } catch (error) { $('status').textContent = '本地服务连接失败'; $('notice').textContent = error.message; }
})();

function renderClock() {
  const clock=state?.clock;
  $('clockPanel').hidden=!clock?.enabled;
  if(!clock?.enabled)return;
  const elapsed=performance.now()-(state.clockReceivedAt ?? performance.now());
  for(const [side,color] of [[1,'red'],[-1,'blue']]){
    const ms=Math.max(0,clock[color+'Ms']-(clock.active===side?elapsed:0));
    const seconds=Math.ceil(ms/1000);
    $(color+'Clock').textContent=`${Math.floor(seconds/60).toString().padStart(2,'0')}:${(seconds%60).toString().padStart(2,'0')}`;
    $(color+'Clock').classList.toggle('running',clock.active===side);
  }
  $('toggleClock').textContent=clock.active!==null?'暂停棋钟':clock.started?'继续计时':'开始计时';
  $('toggleClock').disabled=busy||!!state.winner||state.openingSetup||state.replayMode;
  $('clockHint').textContent=state.replayMode?'打谱不计时；续弈使用新棋钟。':state.openingSetup?'手动开局不计时。':clock.active!==null?'等待操作和分析均计入当前方时间；暂停自动对战不会暂停棋钟。':'棋钟未运行；开始计时或首次落子后启动。悔棋会恢复该手的时间并暂停。';
}
$('toggleClock').addEventListener('click',()=>{
  const running=state?.clock?.active===null;
  if(!running){autoplay=false;analysisPaused=true;clearTimeout(timer);}
  else analysisPaused=false;
  action('/api/clock',{running});
});
setInterval(renderClock,200);
setInterval(async()=>{
  if(busy||clockPollBusy||!state?.clock?.active||state.winner)return;
  clockPollBusy=true;
  const id=state.id, revision=state.revision;
  try{
    const fresh=await request(`/api/session?id=${id}`);
    if(!busy&&state?.id===id&&state.revision===revision){state=fresh;if(state.winner){autoplay=false;clearTimeout(timer);}render();}
  }catch(error){$('notice').textContent=error.message;}
  finally{clockPollBusy=false;}
},1000);

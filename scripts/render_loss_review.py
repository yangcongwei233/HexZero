#!/usr/bin/env python3
"""生成可独立查看的本地败局复盘页面及 CSV 汇总表。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hex 败局复盘 · Azalea</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f2;color:#20362f;font:15px/1.6 system-ui,sans-serif}
main{max-width:1350px;margin:24px auto;padding:0 24px}h1{font-size:28px;margin:0}h2{font-size:19px;margin:8px 0}
p{margin:8px 0}.muted{color:#64746f;font-size:13px}.metrics{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}.metric{background:white;padding:14px 24px;border-radius:12px}.metric b{display:block;font-size:26px}
.grid{display:grid;grid-template-columns:minmax(360px,1fr) minmax(400px,1.2fr);gap:20px}.grid>section{min-width:0}#source{overflow-wrap:anywhere}.card{background:white;padding:22px;border-radius:14px;box-shadow:0 1px 8px #18372a09;margin-bottom:16px}
select,button,input{font:inherit}select{width:100%;padding:8px;border:1px solid #cfdbd3;border-radius:6px;background:white;margin:6px 0}button{border:1px solid #bbcfc3;background:#f5f9f6;padding:6px 14px;border-radius:6px;cursor:pointer}.toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}input[type=range]{flex:1}
svg{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;border-bottom:1px solid #e7ece8;padding:9px 7px}tbody tr[data-branch]{cursor:pointer}tbody tr[data-branch]:hover{background:#eef6f0}.pill{display:inline-block;padding:2px 8px;background:#eff4f0;border-radius:12px;font-size:12px}.warn{color:#af512d}.good{color:#237244}a{color:#24734e}.scroll{overflow:auto}
@media(max-width:850px){.grid{grid-template-columns:1fr}main{padding:0 12px}.card{padding:14px}}
</style>
<main><h1>Azalea · 败局复盘</h1><p id="subtitle"></p>
<p class="muted">全部评分从你的模型视角出发：Q 越大越有利，范围 −1～+1；Q 不是经过校准的胜率，也不是必胜证明。这里只统计败局。</p>
<div id="metrics" class="metrics"></div>
<div class="card"><h2>复盘发现</h2><div id="findings"></div><div id="verified-cases" class="scroll"></div></div>
<div class="grid"><section><div class="card"><h2>选择棋局与落子</h2><select id="game"></select><div class="toolbar"><button id="prev">上一手</button><input id="ply" type="range" min="0"><button id="next">下一手</button></div><p id="position"></p><div id="board"></div><p class="muted">蓝方连上下，红方连左右。橙圈为实战着，绿圈为复核中评分最高的候选着。棋子内数字是实战手数。</p><button id="reset">回到落子前</button></div>
<div class="card"><h2>逐手局面评分</h2><div id="chart"></div><p class="muted">蓝线：落子前整体 Q；橙线：实战着后 Q。均为 800 参数的独立搜索，只用于筛查；可点击曲线切换落子。</p><p id="opening" class="muted"></p></div></section>
<section><div class="card"><h2 id="decision"></h2><div id="flags"></div><p id="evaluation"></p><div id="branches" class="scroll"></div><p id="line" class="muted"></p><div id="verification"></div></div>
<div class="card"><h2>本局加深复核</h2><p class="muted">每局最多选两处疑似转折；候选着及实战着落下后，分别从空树搜索 3200 参数。分差只比较这些已选候选，未穷举所有合法着法。</p><div id="keys"></div></div>
<div class="card"><h2>核验与范围</h2><ul><li>跳过预设的开局着法；原棋谱逐手重放并用 BFS 独立核验胜负。</li><li>一步获胜与唯一挡点逐格枚举核验；双杀表示对手有两个立即获胜点且己方没有一步获胜点。</li><li>Azalea 既是原对手也是复盘者，其判断不等于独立裁判的结论。</li><li>保留原版搜索批量 10；参数 800 / 3200 分别选择 810 / 3210 条路径。批内重复叶会合并，实际根访问数可能更少，已逐项记录。关闭噪声与温度采样，不复用不同局面的搜索树。</li><li>本机 CPU 执行，棋谱保持为评测数据，未加入训练池。</li></ul><a href="results.json">完整 JSON</a> · <a href="decisions.csv">逐手 CSV</a><p id="source" class="muted"></p></div></section></div></main>
<script type="application/json" id="data">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
let gi=0,ri=0;const coord=a=>String.fromCharCode(65+a%11)+(Math.floor(a/11)+1),q=v=>v===null||v===undefined?'—':(v>=0?'+':'')+v.toFixed(3);
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const summary=D.summary,all=D.games.flatMap(g=>g.positions);
$('subtitle').textContent=`模型第 ${[...new Set(D.games.map(g=>g.game.model_iteration))].join(' / ')} 轮 · ${summary.games} 局败局 · 本机 CPU 复盘`;
const ms=[['复盘落子',summary.trained_decisions],['加深复核',summary.deep_positions],['漏一步获胜',summary.exact_tactics.missed_immediate_win.positions],['漏唯一挡点',summary.exact_tactics.missed_unique_block.positions]];
$('metrics').innerHTML=ms.map(([s,n])=>`<div class="metric"><span>${s}</span><b>${n}</b></div>`).join('');
$('game').innerHTML=D.games.map((g,i)=>`<option value="${i}">${i+1}. ${escape(g.game.opening.id)} · ${g.game.opening.labels.join(' / ')} · ${g.game.trained_side===1?'己方先手蓝':'己方后手红'}</option>`).join('');
const proofs=D.games.flatMap(g=>g.positions.filter(r=>r.block_proof?.blocking_still_proven_loses));
const gaps=Object.values(summary.by_seat).reduce((n,s)=>n+s.games_with_deep_gap_at_least_03,0);
$('findings').innerHTML=`<p>${gaps} / ${summary.games} 局中，已检候选着比实战着高至少 0.3 Q；这是复盘筛查阈值，不是棋力评分标准。</p>`+(proofs.length?`<p>${proofs.length} 次漏唯一挡点均已核验：即使正确挡住，对手仍能制造双杀，因此这些晚期漏挡不是败势的起点。</p>`:'');
const verified=[];D.games.forEach((g,i)=>g.positions.forEach((r,j)=>{if(r.verification)verified.push({g,r,i,j})}));
if(verified.length){$('verified-cases').innerHTML='<p>以下案例进一步使用 12800 参数复核；点击可定位棋盘。续弈由 Azalea 接管双方。</p><table><thead><tr><th>开局 / 己方</th><th>手数</th><th>实战 → 替代</th><th>Q 变化</th><th>续弈结果</th></tr></thead><tbody>'+verified.map(({g,r,i,j})=>{const b=r.verification.branches;return`<tr><td>${g.game.opening.labels.join(' / ')} · ${r.player===1?'先手':'后手'}</td><td><button data-g="${i}" data-r="${j}">第 ${r.ply} 手</button></td><td>${coord(b[0].action)} → ${coord(b[1].action)}</td><td>${q(b[0].value)} → ${q(b[1].value)}</td><td>${b[0].rollout.trained_side_won?'胜':'负'} → ${b[1].rollout.trained_side_won?'胜':'负'}</td></tr>`}).join('')+'</tbody></table><p class="muted">这是选出的典型案例，不是随机胜率测试；单次续弈不构成必胜证明。</p>'; $('verified-cases').querySelectorAll('button').forEach(b=>b.onclick=()=>{gi=+b.dataset.g;ri=+b.dataset.r;$('game').value=gi;render();$('position').scrollIntoView({block:'center'})});}
function current(){return[D.games[gi],D.games[gi].positions[ri]]}
function board(preview=null){const[g,r]=current(),stones=new Map(g.moves.filter(m=>m.move<r.ply).map(m=>[m.action,{player:m.player,text:String(m.move)}]));
if(preview)preview.forEach((a,i)=>stones.set(a,{player:r.player*(i%2?-1:1),text:'+'+(i+1),ghost:true}));
const best=r.deep?r.deep.best_action:r.azalea_action;
let html='<svg viewBox="0 0 620 420" role="img" aria-label="当前棋盘">';
html+='<path d="M35 28 L415 28 M205 382 L585 382" stroke="#477eb5" stroke-width="4" fill="none"/><path d="M22 43 L195 363 M425 43 L600 363" stroke="#cc745d" stroke-width="4" fill="none"/>';
for(let c=0;c<11;c++)html+=`<text x="${50+c*34}" y="17" text-anchor="middle" font-size="11" fill="#667a70">${String.fromCharCode(65+c)}</text>`;
for(let row=0;row<11;row++){html+=`<text x="${27+row*17}" y="${55+row*30}" text-anchor="end" font-size="11" fill="#667a70">${row+1}</text>`;
for(let col=0;col<11;col++){const a=row*11+col,x=50+col*34+row*17,y=52+row*30,s=stones.get(a);const points=[[0,-19],[16.5,-9.5],[16.5,9.5],[0,19],[-16.5,9.5],[-16.5,-9.5]].map(([dx,dy])=>`${x+dx},${y+dy}`).join(' ');
html+=`<polygon points="${points}" fill="${s?(s.player===1?'#3d7dc5':'#d9755a'):'#f7faf5'}" stroke="#cad6cc" ${s?.ghost?'opacity=".65"':''}/>`;
if(s)html+=`<text x="${x}" y="${y+4}" text-anchor="middle" fill="white" font-size="11">${s.text}</text>`;
if(a===r.actual)html+=`<circle cx="${x}" cy="${y}" r="14" stroke="#dd9820" stroke-width="3" fill="none"/>`;
if(a===best)html+=`<circle cx="${x}" cy="${y}" r="17" stroke="#187648" stroke-width="3" stroke-dasharray="4 2" fill="none"/>`;
}}$('board').innerHTML=html+'</svg>';}
function chart(){const[g]=current(),rs=g.positions,w=540,h=155,x=i=>35+i*(w-60)/Math.max(1,rs.length-1),y=v=>70-55*v;
let s='<svg viewBox="0 0 540 155">';for(const v of[-1,0,1])s+=`<path d="M30 ${y(v)} H520" stroke="#d6ddd7" stroke-dasharray="3 3"/><text x="5" y="${y(v)+4}" font-size="10">${v}</text>`;
for(const[key,color]of[['azalea_root_value','#3d7dc5'],['actual_after_value','#d58a32']])s+=`<polyline points="${rs.map((r,i)=>`${x(i)},${y(r[key])}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2"/>`;
s+=`<path d="M${x(ri)} 10 V130" stroke="#183c2b"/>`;
rs.forEach((r,i)=>{s+=`<circle data-index="${i}" cx="${x(i)}" cy="${y(r.azalea_root_value)}" r="6" fill="#3d7dc5" opacity=".65" style="cursor:pointer"><title>第 ${r.ply} 手</title></circle>`;if(i%5===0)s+=`<text x="${x(i)}" y="149" text-anchor="middle" font-size="10">${r.ply}</text>`});
$('chart').innerHTML=s+'</svg>';$('chart').querySelectorAll('[data-index]').forEach(e=>e.onclick=()=>{ri=+e.dataset.index;render()});}
function render(){const[g,r]=current();$('ply').max=g.positions.length-1;$('ply').value=ri;$('prev').disabled=ri===0;$('next').disabled=ri===g.positions.length-1;
$('position').textContent=`实战第 ${r.ply} 手之前 · ${r.player===1?'蓝方':'红方'}轮到落子`;
$('decision').textContent=`实战 ${coord(r.actual)} · Azalea 800 建议 ${coord(r.azalea_action)}`;
const flags=[];if(r.missed_immediate_win)flags.push(`漏一步获胜：${r.own_winning_moves.map(coord).join(' / ')}`);if(r.missed_unique_block)flags.push(`漏唯一挡点：${r.opponent_winning_moves.map(coord).join(' / ')}`);if(r.already_lost_double_threat)flags.push(`已经面对双杀：${r.opponent_winning_moves.map(coord).join(' / ')}`);
$('flags').innerHTML=flags.map(s=>`<p class="warn">${s}</p>`).join('')||'<span class="pill">未检出一步漏胜、漏唯一挡点或对手双杀</span>';
if(r.block_proof?.blocking_still_proven_loses){const p=r.block_proof,f=p.opponent_winning_forks_after_block[0];$('flags').innerHTML+=`<p>规则核验：即使挡在 ${coord(p.unique_block)}，对手下 ${coord(f.action)} 后仍有 ${f.winning_cells.map(coord).join(' / ')} 两个以上获胜点。这一局面在漏挡之前已可证明败势。</p>`;}
if(r.verification){const v=r.verification,bs=v.branches;$('verification').innerHTML=`<h2>12800 再复核与续弈</h2><p>实战 ${coord(bs[0].action)}：Q ${q(bs[0].value)}；替代 ${coord(bs[1].action)}：Q ${q(bs[1].value)}。</p><p>Azalea 接管双方续弈：实战着分支${bs[0].rollout.trained_side_won?'胜':'负'}，替代着分支${bs[1].rollout.trained_side_won?'胜':'负'}。</p><p class="muted">续弈每步使用 800 参数，各分支仅一条线路。它说明该引擎如何兑现局面，不是你的模型改下后的胜率，也不构成必胜证明。</p>`;}else{$('verification').innerHTML='';}
let branches;if(r.deep){branches=r.deep.branches;$('evaluation').textContent=`3200 等预算复核：实战着 Q ${q(r.deep.actual_value)}；已检候选最高 Q ${q(r.deep.best_value)}，差 ${r.deep.gap.toFixed(3)}。`;}
else{branches=r.candidates.map(c=>({...c,pv:c.pv.map(m=>m.action)}));$('evaluation').textContent=`800 筛查：落子前整体 Q ${q(r.azalea_root_value)}；实战着后 Q ${q(r.actual_after_value)}。下表为根边估值，访问次数不同，不能当作等预算复核。`;}
$('branches').innerHTML='<table><thead><tr><th>着法</th><th>Q</th><th>访问数</th><th>搜索主变</th></tr></thead><tbody>'+branches.map((b,i)=>`<tr data-branch="${i}"><td>${coord(b.action)} ${b.action===r.actual?'（实战）':''}</td><td>${q(b.value)}</td><td>${b.root_visits??b.visits}</td><td>${b.pv.map(coord).join(' → ')}</td></tr>`).join('')+'</tbody></table>';
$('branches').querySelectorAll('[data-branch]').forEach(e=>e.onclick=()=>{const b=branches[+e.dataset.branch];board(b.pv);$('line').textContent='预览主变：'+b.pv.map(coord).join(' → ')});
$('line').textContent='点击表格行可预览搜索主变；主变仅为模型搜索到的线路。';
$('keys').innerHTML=g.positions.map((p,i)=>p.deep?`<p><button data-key="${i}">第 ${p.ply} 手</button> ${coord(p.actual)} → ${coord(p.deep.best_action)}，Q ${q(p.deep.actual_value)} → ${q(p.deep.best_value)}</p>`:'').join('')||'<p>未选出需要加深的不同着法。</p>';
$('keys').querySelectorAll('[data-key]').forEach(e=>e.onclick=()=>{ri=+e.dataset.key;render()});
$('opening').textContent=`预设开局结束后的己方 Q：${q(g.opening_value)}；此时开局已给定，不归责于模型。`;
$('source').textContent='原棋谱：'+g.game.record;board();chart();}
$('game').onchange=()=>{gi=+$('game').value;ri=0;render()};$('ply').oninput=()=>{ri=+$('ply').value;render()};$('prev').onclick=()=>{ri--;render()};$('next').onclick=()=>{ri++;render()};$('reset').onclick=()=>{board();$('line').textContent='已回到落子前。'};render();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    assert data['status'] == 'complete'
    proofs_path = args.results.with_name('missed_block_certificates.json')
    proofs = json.loads(proofs_path.read_text())['cases'] if proofs_path.exists() else []
    proofs = {(p['record'], p['ply']): p for p in proofs}
    verification_path = args.results.with_name('verification.json')
    verifications = {}
    if verification_path.exists():
        verified = json.loads(verification_path.read_text())
        assert verified['source_review_sha256'] == hashlib.sha256(args.results.read_bytes()).hexdigest()
        verifications = {(v['game']['record'], v['ply']): v for v in verified['cases']}
    fields = ['opening', 'trained_side', 'ply', 'actual', 'azalea_800', 'root_q_800',
              'actual_after_q_800', 'deep_best', 'deep_actual_q', 'deep_best_q', 'deep_gap',
              'missed_immediate_win', 'missed_unique_block', 'already_lost_double_threat', 'record']
    def coord(a):
        return chr(65 + a % 11) + str(a // 11 + 1)
    with args.results.with_name('decisions.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for game in data['games']:
            record = json.loads(Path(game['game']['record']).read_text())
            game['moves'] = [{k: m[k] for k in ('move', 'action', 'player')} for m in record['moves']]
            for row in game['positions']:
                key = (game['game']['record'], row['ply'])
                if key in proofs:
                    row['block_proof'] = proofs[key]
                if key in verifications:
                    row['verification'] = verifications[key]
                deep = row.get('deep', {})
                writer.writerow(dict(opening=game['game']['opening']['id'],
                                     trained_side=row['player'], ply=row['ply'], actual=coord(row['actual']),
                                     azalea_800=coord(row['azalea_action']), root_q_800=row['azalea_root_value'],
                                     actual_after_q_800=row['actual_after_value'],
                                     deep_best=coord(deep['best_action']) if deep else '',
                                     deep_actual_q=deep.get('actual_value', ''), deep_best_q=deep.get('best_value', ''),
                                     deep_gap=deep.get('gap', ''),
                                     **{k: row[k] for k in fields[-4:-1]}, record=game['game']['record']))
    content = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    destination = args.results.with_name('report.html')
    destination.write_text(TEMPLATE.replace('__DATA__', content), encoding='utf-8')
    print(destination.resolve())


if __name__ == '__main__':
    main()

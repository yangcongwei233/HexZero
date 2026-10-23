#!/usr/bin/env python3
"""Build model-independent, local central openings distant from recorded training states."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hex.HexGame import HexGame


def position_key(moves):
    own, other = tuple(sorted(moves[::2])), tuple(sorted(moves[1::2]))
    rotated = tuple(sorted(120-a for a in own)), tuple(sorted(120-a for a in other))
    return min((own, other), rotated)


def distance(a, b):
    ar, ac = divmod(a, 11); br, bc = divmod(b, 11)
    dr, dc = ar-br, ac-bc
    return max(abs(dr), abs(dc), abs(dr+dc))


class PositionIndex:
    def __init__(self, games, plies):
        keys = sorted({position_key(moves[:plies]) for moves in games})
        self.positions = []
        for red, blue in keys:
            self.positions.append((red, blue))
            rotated = tuple(sorted(120-a for a in red)), tuple(sorted(120-a for a in blue))
            if rotated != (red, blue):
                self.positions.append(rotated)
        self.matrix = np.zeros((242, len(self.positions)), dtype=np.uint8)
        for i, (red, blue) in enumerate(self.positions):
            self.matrix[list(red), i] = 1
            self.matrix[[a+121 for a in blue], i] = 1
        self.plies = plies
        self.unique_positions_up_to_rotation = len(keys)

    def nearest(self, moves):
        indices = moves[::2]+[a+121 for a in moves[1::2]]
        overlap = self.matrix[indices].sum(axis=0, dtype=np.uint8)
        index = int(overlap.argmax())
        return self.plies-int(overlap[index]), self.positions[index]


def sample_opening(rng, plies):
    # Common central starts, nearby replies, and local continuations for BOTH colors.
    first = int(rng.choice([r*11+c for r in range(3,8) for c in range(3,8)]))
    second = int(rng.choice([a for a in range(121) if 1<=distance(a,first)<=2
                            and 1<=a//11<=9 and 1<=a%11<=9]))
    moves = [first, second]
    while len(moves)<plies:
        own = moves[len(moves)%2::2]
        options = [a for a in range(121) if a not in moves
                   and 1<=a//11<=9 and 1<=a%11<=9
                   and min(distance(a,b) for b in own)<=2]
        if not options:
            return None
        moves.append(int(rng.choice(options)))
    return moves


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefixes',type=Path,required=True)
    parser.add_argument('--previous-openings',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=2026101503)
    parser.add_argument('--per-depth',type=int,default=20)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    data=json.loads(gzip.decompress(args.prefixes.read_bytes()))
    training=[r[3] for r in data['selfplay']]
    teachers=[r['moves'] for r in data['teacher']]
    assert len(training)==880*128 and len(teachers)==512
    old=json.loads(args.previous_openings.read_text())
    old_keys={position_key(r['actions'][:2]) for r in old}
    first2=Counter(position_key(m[:2]) for m in training)
    teacher2={position_key(m[:2]) for m in teachers}
    rng=np.random.default_rng(args.seed)
    selected=[];groups=[]
    for plies in (4,6,8):
        self_index=PositionIndex(training,plies)
        teacher_index=PositionIndex(teachers,plies)
        threshold=plies//2
        pool=[];keys=set();attempts=0
        # Select from a fixed-size pool, using position distances alone, never model evaluations.
        for _ in range(2400):
            attempts+=1
            moves=sample_opening(rng,plies)
            if moves is None or position_key(moves[:2]) in old_keys:continue
            key=position_key(moves)
            if key in keys:continue
            keys.add(key)
            self_distance,near_self=self_index.nearest(moves)
            if self_distance<threshold:continue
            teacher_distance,near_teacher=teacher_index.nearest(moves)
            if teacher_distance<threshold:continue
            pool.append(dict(actions=moves,training_distance=self_distance,teacher_distance=teacher_distance,
                nearest_selfplay_position=near_self,nearest_teacher_position=near_teacher,
                first_two_seen_in_selfplay=first2[position_key(moves[:2])],
                first_two_seen_in_teacher=position_key(moves[:2]) in teacher2))
        # Stable shuffle breaks ties reproducibly without favoring coordinates or any model.
        rng.shuffle(pool)
        pool.sort(key=lambda p:(p['training_distance'],p['teacher_distance']),reverse=True)
        taken=[];first_counts=Counter();pair_counts=Counter()
        for item in pool:
            first=item['actions'][0];pair=position_key(item['actions'][:2])
            if first_counts[first]>=1 or pair_counts[pair]>=1:continue
            # Avoid selecting almost the same test position more than once in this depth stratum.
            if taken and min(plies-max(len(set(item['actions'][::2])&set(v[0]))+
                                      len(set(item['actions'][1::2])&set(v[1]))
                                      for v in (position_key(other['actions']),
                                        (tuple(120-a for a in position_key(other['actions'])[0]),
                                         tuple(120-a for a in position_key(other['actions'])[1]))))
                             for other in taken)<2:continue
            taken.append(item);first_counts[first]+=1;pair_counts[pair]+=1
            if len(taken)==args.per_depth:break
        if len(taken)!=args.per_depth:
            raise RuntimeError(f'{plies} plies: only {len(taken)} sufficiently different central starts from {len(pool)} candidates')
        game=HexGame(11)
        for i,item in enumerate(taken):
            board,player=game.getInitBoard(),1
            for action in item['actions']:
                board,player=game.getNextState(board,player,action)
                assert not game.getGameEnded(board,player)
            # Independent set calculation checks the reported nearest-position distance.
            for name,near in [('training',item['nearest_selfplay_position']),('teacher',item['nearest_teacher_position'])]:
                shared=len(set(item['actions'][::2])&set(near[0]))+len(set(item['actions'][1::2])&set(near[1]))
                assert plies-shared==item[name+'_distance']
            item.update(id=f'central_novel_{plies}ply_{i+1:02d}',plies=plies,
                labels=[f'{chr(65+a%11)}{11-a//11}' for a in item['actions']])
            selected.append(item)
        groups.append(dict(plies=plies,attempted=attempts,qualifying_pool=len(pool),selected=len(taken),
            minimum_required_relocated_stones=threshold,
            selfplay_unique_positions=self_index.unique_positions_up_to_rotation,
            teacher_unique_positions=teacher_index.unique_positions_up_to_rotation,
            selected_min_selfplay_distance=min(r['training_distance'] for r in taken),
            selected_max_selfplay_distance=max(r['training_distance'] for r in taken),
            selected_min_teacher_distance=min(r['teacher_distance'] for r in taken)))
        print(json.dumps(groups[-1]),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),seed=args.seed,openings=len(selected),groups=groups,
        source_sha256=hashlib.sha256(args.prefixes.read_bytes()).hexdigest(),
        openings_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        complete_selfplay_iterations=[1,880],selfplay_games=len(training),teacher_games=len(teachers),
        distance='For the same number of stones, plies minus maximum same-color overlap with ANY recorded position or its 180-degree rotation. Equivalently, the number of stones needing relocation.',
        structure='First move in central 5x5, second at Hex distance 1 or 2, subsequent moves within distance 2 of own stones; every stone in inner 9x9.',
        selection='No model values, policy priors, game results, or outcomes were used. Fixed candidate pool and seed; sort by novelty within geometric constraints.',
        old_suite=dict(count=len(old),first_two_seen_in_selfplay=sum(position_key(r['actions'][:2]) in first2 for r in old),
                       first_two_seen_in_teacher=sum(position_key(r['actions'][:2]) in teacher2 for r in old)),
        new_suite=dict(exact_training_position_matches=0,exact_teacher_position_matches=0,
            first_two_seen_in_selfplay=sum(bool(r['first_two_seen_in_selfplay']) for r in selected),
            first_two_seen_in_teacher=sum(r['first_two_seen_in_teacher'] for r in selected)),
        limitations=['Geometric constraints preserve local play but do not prove optimal moves or balanced positions.',
                     'Distance is a board-position metric, not a semantic or strategic similarity guarantee.',
                     'Coverage is for recorded training through iteration 880. Future checkpoints require another overlap audit.',
                     'Common first two moves are allowed; the final preset position is the unit of exclusion.'])
    args.output.with_suffix('.selection.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output),summary=report['new_suite']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()

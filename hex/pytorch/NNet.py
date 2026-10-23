import os
import sys
import time
import logging
import copy
import random
import tempfile

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from NeuralNet import NeuralNet
from utils import AverageMeter, dotdict

import torch
import torch.nn.functional as F
import torch.optim as optim

from .HexNNet import HexNNet
from ..value_diagnostics import format_numeric_stats, value_stats


log = logging.getLogger(__name__)


def use_cuda():
    if os.environ.get('HEX_FORCE_CPU') == '1':
        return False
    return torch.cuda.is_available()


args = dotdict({
    'lr': 0.001,
    'epochs': 3,
    'batch_size': 32,
    'cuda': use_cuda(),
    'num_channels': 64,
    'num_res_blocks': 3,
    'policy_channels': 4,
    'value_channels': 8,
    'value_hidden': 128,
    'disable_value_bn': False,
    'debug_value_diagnostics': False,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game, config=None):
        self.game = game
        self.args = dotdict(dict(args))
        self.args.update(config or {})
        self.writer = None
        self.train_step = 0
        self.last_train_metrics = {}
        self.checkpoint_metadata = {}
        self.nnet = HexNNet(game, self.args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if self.args.cuda:
            self.nnet.cuda()
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)

    def train(self, examples, *, teacher=None, teacher_fraction=0.):
        if not examples:
            raise ValueError("Cannot train on an empty replay buffer")
        if not 0 <= teacher_fraction < 1 or (teacher_fraction and teacher is None):
            raise ValueError('Teacher mixing requires a dataset and a fraction in [0, 1)')
        optimizer = self.optimizer
        logged_shape_diagnostic = False
        logged_feature_diagnostic = False
        debug_value_diagnostics = bool(getattr(self.args, 'debug_value_diagnostics', False))
        if debug_value_diagnostics:
            self._log_value_head_optimizer_diagnostic(optimizer)
        teacher_credit, teacher_seen, total_seen = 0., 0, 0

        for epoch in range(self.args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            source_losses = {source: {'policy': 0., 'value': 0., 'count': 0} for source in ('selfplay', 'teacher')}
            batch_count = max(1, int(np.ceil(len(examples) / self.args.batch_size)))

            t = tqdm(range(batch_count), desc='Training Net')
            for batch_idx in t:
                batch_size = min(self.args.batch_size, len(examples))
                teacher_credit += batch_size * teacher_fraction
                teacher_count = int(teacher_credit + 1e-12)
                teacher_credit -= teacher_count
                sample_ids = np.random.randint(len(examples), size=batch_size - teacher_count)
                batch = [examples[i] for i in sample_ids]
                if teacher_count:
                    batch.extend(teacher.sample(teacher_count))
                boards, pis, vs = list(zip(*batch))
                boards = torch.FloatTensor(np.array(boards).astype(np.float32))
                target_pis = torch.FloatTensor(np.array(pis).astype(np.float32))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float32))

                if self.args.cuda:
                    boards = boards.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                if debug_value_diagnostics and not logged_feature_diagnostic:
                    out_pi, out_v, feature_debug = self.nnet.forward_debug(boards)
                    self._log_value_feature_diagnostic(feature_debug)
                    logged_feature_diagnostic = True
                else:
                    out_pi, out_v = self.nnet(boards)
                out_v_flat = out_v.float().view(-1)
                target_vs_flat = target_vs.float().view(-1)
                if debug_value_diagnostics and not logged_shape_diagnostic:
                    log.info(
                        "Training tensor shapes: out_v=%s target_v=%s flattened_out_v=%s flattened_target_v=%s out_pi=%s target_pi=%s",
                        tuple(out_v.shape),
                        tuple(target_vs.shape),
                        tuple(out_v_flat.shape),
                        tuple(target_vs_flat.shape),
                        tuple(out_pi.shape),
                        tuple(target_pis.shape),
                    )
                    logged_shape_diagnostic = True
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs_flat, out_v_flat)
                total_loss = l_pi + l_v
                teacher_seen += teacher_count
                total_seen += batch_size
                if teacher_fraction:
                    # 分别统计各来源损失；总损失仍按原批次均值计算，
                    # 不增加优化器更新次数。
                    per_policy = -(target_pis * out_pi).sum(dim=1).detach()
                    per_value = (target_vs_flat - out_v_flat).square().detach()
                    split = batch_size - teacher_count
                    for source, section in (('selfplay', slice(0, split)), ('teacher', slice(split, None))):
                        count = len(per_policy[section])
                        if count:
                            source_losses[source]['policy'] += float(per_policy[section].sum().item())
                            source_losses[source]['value'] += float(per_value[section].sum().item())
                            source_losses[source]['count'] += count

                if debug_value_diagnostics and batch_idx == 0:
                    pred_vs = out_v_flat.detach().cpu().numpy()
                    target_vs_np = target_vs_flat.detach().cpu().numpy()
                    log.info(
                        "Epoch %s first batch value diagnostic: target_v %s ; pred_v %s ; value_loss=%.6f policy_loss=%.6f",
                        epoch + 1,
                        format_numeric_stats(value_stats(target_vs_np)),
                        format_numeric_stats(value_stats(pred_vs)),
                        float(l_v.item()),
                        float(l_pi.item()),
                    )

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                self.train_step += 1
                if self.writer is not None and self.train_step % self.args.get('tensorboard_interval', 10) == 0:
                    self.writer.add_scalar('train/policy_loss', l_pi.item(), self.train_step)
                    self.writer.add_scalar('train/value_loss', l_v.item(), self.train_step)
                    self.writer.add_scalar('train/total_loss', total_loss.item(), self.train_step)
                    self.writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], self.train_step)
                    if teacher_fraction:
                        self.writer.add_scalar('teacher/batch_fraction', teacher_count / batch_size, self.train_step)
            self.last_train_metrics = {'policy_loss': pi_losses.avg, 'value_loss': v_losses.avg}
        if teacher_fraction:
            self.last_train_metrics.update(teacher_fraction=teacher_seen / total_seen, teacher_samples=teacher_seen)
            for source, totals in source_losses.items():
                if totals['count']:
                    for loss in ('policy', 'value'):
                        self.last_train_metrics[source + '_' + loss + '_loss'] = totals[loss] / totals['count']
            log.info('Teacher mixture: %s/%s samples (%.4f), balanced teacher colors',
                     teacher_seen, total_seen, teacher_seen / total_seen)
        return self.last_train_metrics

    def predict(self, board):
        pis, vs = self.predict_batch([board])
        return pis[0], vs[0]

    def predict_batch(self, boards):
        boards = torch.FloatTensor(np.array(boards).astype(np.float32))
        if self.args.cuda:
            boards = boards.contiguous().cuda()
        boards = boards.view(-1, self.board_x, self.board_y)

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(boards)

        return torch.exp(pi).data.cpu().numpy(), v.data.cpu().numpy()

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        targets = targets.float().view(-1)
        outputs = outputs.float().view(-1)
        if targets.shape != outputs.shape:
            raise ValueError("value loss shape mismatch: targets={} outputs={}".format(
                tuple(targets.shape),
                tuple(outputs.shape),
            ))
        return F.mse_loss(outputs, targets)

    def _log_value_head_optimizer_diagnostic(self, optimizer):
        optimizer_param_ids = {
            id(param)
            for group in optimizer.param_groups
            for param in group['params']
        }
        value_params = [
            (name, param)
            for name, param in self.nnet.named_parameters()
            if name.startswith('value_')
        ]
        for name, param in value_params:
            log.info(
                "Value head optimizer diagnostic: %s shape=%s requires_grad=%s in_optimizer=%s",
                name,
                tuple(param.shape),
                bool(param.requires_grad),
                id(param) in optimizer_param_ids,
            )

    def _log_value_feature_diagnostic(self, debug):
        for label in (
            'trunk',
            'value_conv',
            'value_bn',
            'value_fc1',
            'value_fc2_tanh',
        ):
            values = debug[label].detach().cpu().numpy()
            log.info(
                "Value head feature diagnostic: %s %s",
                label,
                format_numeric_stats(value_stats(values)),
            )

        policy_logits = debug['policy_logits'].detach().cpu().numpy()
        log.info(
            "Value head feature diagnostic: policy_logits std=%.6f",
            float(policy_logits.std()),
        )
        log.info(
            "Value head feature diagnostic: disable_value_bn=%s value_bn_module=%s",
            bool(getattr(self.args, 'disable_value_bn', False)),
            self.nnet.value_bn.__class__.__name__,
        )

    ARCH_KEYS = ('num_channels', 'num_res_blocks', 'policy_channels', 'value_channels',
                 'value_hidden', 'disable_value_bn')

    def get_inference_state(self):
        return {
            'format_version': 2,
            'board_size': [self.board_x, self.board_y],
            'canonical_version': 1,
            'network_config': {k: self.args[k] for k in self.ARCH_KEYS},
            'state_dict': {k: v.detach().cpu().clone() for k, v in self.nnet.state_dict().items()},
        }

    def get_training_state(self):
        np_state = np.random.get_state()
        return {
            **self.get_inference_state(),
            'training_config': {k: self.args[k] for k in ('lr', 'epochs', 'batch_size')},
            'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
            'train_step': self.train_step,
            'metadata': dict(self.checkpoint_metadata),
            'rng_state': {
                'python': random.getstate(),
                'numpy': [np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]],
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if self.args.cuda else [],
            },
        }

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', include_training=True):
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        with tempfile.NamedTemporaryFile(dir=folder, prefix='.checkpoint-', delete=False) as f:
            temporary = f.name
        try:
            state = self.get_training_state() if include_training else self.get_inference_state()
            with open(temporary, 'wb') as output:
                torch.save(state, output)
            os.replace(temporary, filepath)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load_training_state(self, checkpoint, restore_rng=False):
        if 'board_size' in checkpoint and tuple(checkpoint['board_size']) != (self.board_x, self.board_y):
            raise ValueError('Checkpoint board size does not match the game')
        if checkpoint.get('canonical_version', 1) != 1:
            raise ValueError('Unsupported checkpoint canonical representation')
        state = checkpoint['state_dict']
        config = checkpoint.get('network_config')
        if config is None:
            # 兼容仅含权重的旧 HexZero 断点。
            config = {
                'num_channels': int(state['input_conv.weight'].shape[0]),
                'policy_channels': int(state['policy_conv.weight'].shape[0]),
                'value_channels': int(state['value_conv.weight'].shape[0]),
                'value_hidden': int(state['value_fc1.weight'].shape[0]),
                'disable_value_bn': 'value_bn.weight' not in state,
                'num_res_blocks': len({key.split('.')[1] for key in state if key.startswith('res_blocks.')}),
            }
            log.warning('Loading legacy weights without representation metadata or complete training state')
        partial = bool(self.args.get('allow_partial_checkpoint', False))
        if not partial:
            if any(self.args.get(k) != config[k] for k in self.ARCH_KEYS):
                self.args.update(config)
                self.nnet = HexNNet(self.game, self.args)
                if self.args.cuda:
                    self.nnet.cuda()
            self.args.update(checkpoint.get('training_config', {}))
            self.nnet.load_state_dict(state, strict=True)
        else:
            current = self.nnet.state_dict()
            compatible = {k: v for k, v in state.items() if k in current and v.shape == current[k].shape}
            missing, unexpected = self.nnet.load_state_dict(compatible, strict=False)
            log.warning('Partial checkpoint migration: missing=%s skipped=%s', missing, sorted(set(state) - set(compatible)))
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)
        if not partial and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_step = int(checkpoint.get('train_step', 0)) if not partial else 0
        self.checkpoint_metadata = dict(checkpoint.get('metadata', {}))
        if restore_rng and not partial and 'rng_state' in checkpoint:
            rng = checkpoint['rng_state']
            random.setstate(rng['python'])
            ns = rng['numpy']
            np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
            torch.set_rng_state(rng['torch'].cpu())
            if self.args.cuda and len(rng['cuda']) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar', restore_rng=False):
        filepath = os.path.join(folder, filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError("No model in path {}".format(filepath))
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)
        self.load_training_state(checkpoint, restore_rng=restore_rng)

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../..')


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        """建立两个同宽卷积及批归一化层，保持空间尺寸以便与输入残差相加。"""
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """执行卷积、归一化和激活，再与输入相加，返回同形状残差特征。"""
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class HexNNet(nn.Module):
    def __init__(self, game, args):
        """按配置建立三通道输入、残差主干及独立策略/价值分支；分支宽度和价值归一化可配置。"""
        super(HexNNet, self).__init__()
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        channels = args.num_channels
        self.input_conv = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(args.num_res_blocks)])

        policy_channels = int(getattr(args, 'policy_channels', 4))
        value_channels = int(getattr(args, 'value_channels', 2))

        self.policy_conv = nn.Conv2d(channels, policy_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_fc = nn.Linear(policy_channels * self.board_x * self.board_y, self.action_size)

        self.value_conv = nn.Conv2d(channels, value_channels, 1, bias=False)
        self.disable_value_bn = bool(getattr(args, 'disable_value_bn', False))
        self.value_bn = nn.Identity() if self.disable_value_bn else nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * self.board_x * self.board_y, args.value_hidden)
        self.value_fc2 = nn.Linear(args.value_hidden, 1)

    def _forward_impl(self, s, return_debug=False):
        """将规范棋盘编码为己方、对方和常数通道，返回策略对数概率及 tanh 价值；调试模式额外返回价值分支中间特征。"""
        s = s.view(-1, self.board_x, self.board_y)
        own = (s == 1).float()
        opp = (s == -1).float()
        ones = torch.ones_like(own)
        x = torch.stack((own, opp, ones), dim=1)

        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            x = block(x)
        trunk = x

        p = F.relu(self.policy_bn(self.policy_conv(trunk)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        value_conv = self.value_conv(trunk)
        value_bn = self.value_bn(value_conv)
        v = F.relu(value_bn)
        v = v.view(v.size(0), -1)
        value_fc1 = self.value_fc1(v)
        v = F.relu(value_fc1)
        value_fc2 = self.value_fc2(v)
        value_tanh = torch.tanh(value_fc2)

        out = (F.log_softmax(p, dim=1), value_tanh)
        if not return_debug:
            return out

        debug = {
            'trunk': trunk,
            'value_conv': value_conv,
            'value_bn': value_bn,
            'value_fc1': value_fc1,
            'value_fc2_tanh': value_tanh,
            'policy_logits': p,
        }
        return out[0], out[1], debug

    def forward(self, s):
        """执行常规网络前向计算，只返回训练或搜索需要的策略和价值。"""
        return self._forward_impl(s, return_debug=False)

    def forward_debug(self, s):
        """执行同一前向路径并附带中间特征，供价值饱和与梯度诊断使用。"""
        return self._forward_impl(s, return_debug=True)

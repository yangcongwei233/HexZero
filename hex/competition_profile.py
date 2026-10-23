"""比赛发行包固定使用经过选择的 3250 FP32 权重。

同时核对轮次和内容 SHA-256，不能只靠文件名/清单中的 iteration 判断模型。
此限制用于防止误换模型；Python 源码可修改，不将其描述为防篡改或加密保护。
公开源码只包含标识和校验值，不包含权重；普通版也不受此锁定限制。
"""
MODEL_ID = 'iteration_3250/trained.onnx'
MODEL_SHA256 = '828692134042a3a8ce827790a8191618ea0979cbe118bfbc45d690989de25f52'


def validate_competition_model(net):
    """OnnxNetwork 加载时已对实际文件重新计算哈希，此处再与固定版本比对。"""
    if (net.manifest['onnx_sha256'] != MODEL_SHA256
            or net.checkpoint_metadata.get('iteration') != 3250):
        raise ValueError('比赛版只允许指定的 3250 模型；权重或清单与固定版本不符')

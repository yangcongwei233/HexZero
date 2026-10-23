"""比赛版按轮次和 SHA-256 锁定 3250；仅防误换，不含权重或源码防篡改机制。"""
MODEL_ID = 'iteration_3250/trained.onnx'
MODEL_SHA256 = '828692134042a3a8ce827790a8191618ea0979cbe118bfbc45d690989de25f52'


def validate_competition_model(net):
    """OnnxNetwork 加载时已对实际文件重新计算哈希，此处再与固定版本比对。"""
    if (net.manifest['onnx_sha256'] != MODEL_SHA256
            or net.checkpoint_metadata.get('iteration') != 3250):
        raise ValueError('比赛版只允许指定的 3250 模型；权重或清单与固定版本不符')

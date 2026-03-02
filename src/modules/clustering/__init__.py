# 使用延迟导入以避免循环导入问题
# 先导入不依赖其他模块的类
from src.modules.clustering.vector_quantization import VectorQuantization

# 然后导入其他模块
from src.modules.clustering.residual_quantization import ResidualQuantization

# 最后导入 VQVAE（它依赖 VectorQuantization）
from src.modules.clustering.vq_vae import VQVAE

__all__ = [
    "ResidualQuantization",
    "VectorQuantization",
    "VQVAE",
]


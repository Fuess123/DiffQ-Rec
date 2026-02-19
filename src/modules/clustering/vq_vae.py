"""
VQ-VAE (Vector Quantized Variational AutoEncoder) 模块

该模块实现了可微分的语义量化，将连续的嵌入向量量化为离散的语义ID。
与 RQ-VAE 不同，VQ-VAE 使用单层量化而不是多层残差量化。
"""

from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningModule
from lightning.pytorch.trainer.states import TrainerFn
from torch import nn
from torchmetrics import MeanMetric


class VQVAE(LightningModule):
    """
    VQ-VAE (Vector Quantized Variational AutoEncoder) 模块
    
    该模块实现了基于向量量化的自编码器，能够将连续的嵌入向量量化为离散的语义ID，
    同时保持可微分性以支持端到端训练。
    
    与 ResidualQuantization 的主要区别：
    - VQ-VAE 使用单层量化，而 RQ-VAE 使用多层残差量化
    - VQ-VAE 的架构更简单，适合作为基础量化模块
    """
    
    def __init__(
        self,
        normalization_layer: nn.Module = nn.Identity(),
        encoder: nn.Module = nn.Identity(),
        decoder: nn.Module = nn.Identity(),
        vector_quantization_layer: Optional["VectorQuantization"] = None,
        init_buffer_size: int = 1000,
        training_loop_function: callable = None,
        quantization_loss_weight: float = 1.0,
        reconstruction_loss_function: Optional[nn.Module] = None,
        reconstruction_loss_weight: float = 1.0,
        commitment_loss_weight: float = 0.25,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        初始化 VQ-VAE 模块
        
        Args:
            normalization_layer: 输入归一化层，用于归一化输入嵌入
            encoder: 编码器网络，将输入嵌入映射到潜在空间
            decoder: 解码器网络，将量化后的潜在表示映射回原始嵌入空间
            vector_quantization_layer: 向量量化层，执行量化操作
            init_buffer_size: 用于初始化 codebook 的缓冲区大小
            training_loop_function: 自定义训练循环函数（可选）
            quantization_loss_weight: 量化损失的权重
            reconstruction_loss_function: 重构损失函数
            reconstruction_loss_weight: 重构损失的权重
            commitment_loss_weight: commitment loss 的权重（用于稳定训练）
            optimizer: 优化器
            scheduler: 学习率调度器
            verbose: 是否输出详细日志
        """
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "optimizer",
                "scheduler",
                "training_loop_function",
                "normalization_layer",
                "encoder",
                "decoder",
                "vector_quantization_layer",
                "reconstruction_loss_function",
            ],
        )
        
        # 保存优化器和调度器
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # 保存组件
        self.normalization_layer = normalization_layer
        self.encoder = encoder
        self.decoder = decoder
        
        # 验证量化层是否存在
        if vector_quantization_layer is None:
            raise ValueError(
                "vector_quantization_layer must be provided. "
                "It should be an instance of VectorQuantization."
            )
        # 延迟导入 VectorQuantization 以避免循环导入
        from src.modules.clustering.vector_quantization import VectorQuantization
        if not isinstance(vector_quantization_layer, VectorQuantization):
            raise TypeError(
                f"vector_quantization_layer must be an instance of VectorQuantization, "
                f"got {type(vector_quantization_layer)}"
            )
        self.vector_quantization_layer = vector_quantization_layer
        
        # 保存训练相关参数
        self.training_loop_function = training_loop_function
        if self.training_loop_function is not None:
            self.automatic_optimization = False
        
        # 保存损失权重
        self.quantization_loss_weight = quantization_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        
        # 保存损失函数
        self.reconstruction_loss_function = reconstruction_loss_function
        
        # 初始化指标
        self.train_loss = MeanMetric()
        self.train_quantization_loss = MeanMetric()
        self.train_reconstruction_loss = MeanMetric()
        self.train_commitment_loss = MeanMetric()
        
        self.val_loss = MeanMetric()
        self.val_quantization_loss = MeanMetric()
        self.val_reconstruction_loss = MeanMetric()
        self.val_commitment_loss = MeanMetric()
        
        self.test_loss = MeanMetric()
        self.test_quantization_loss = MeanMetric()
        self.test_reconstruction_loss = MeanMetric()
        self.test_commitment_loss = MeanMetric()
        
        # 日志设置
        self.verbose = verbose
        
        # 初始化缓冲区大小（传递给量化层）
        self.init_buffer_size = init_buffer_size
        if hasattr(self.vector_quantization_layer, 'init_buffer_size'):
            self.vector_quantization_layer.init_buffer_size = init_buffer_size
    
    def forward(
        self, 
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            embeddings: 编码后的嵌入，形状为 (batch_size, latent_dim)
            
        Returns:
            cluster_ids: 量化索引，形状为 (batch_size,)
            quantized_embeddings: 量化后的嵌入，形状为 (batch_size, latent_dim)
            reconstruction_loss_embeddings: 用于重构损失的可微分嵌入，形状为 (batch_size, latent_dim)
            quantization_loss: 量化损失值
        """
        # 确保嵌入在正确的设备上
        if embeddings.device != self.device:
            embeddings = embeddings.to(self.device)
        
        # 通过量化层进行量化
        # VectorQuantization.forward 返回: (ids, embeddings, reconstruction_loss_embeddings)
        cluster_ids, quantized_embeddings, reconstruction_loss_embeddings = (
            self.vector_quantization_layer.forward(embeddings)
        )
        
        # 计算量化损失（如果量化层已初始化）
        if self.vector_quantization_layer.is_initialized:
            # 使用 model_step 获取量化损失
            _, _, quantization_loss = self.vector_quantization_layer.model_step(embeddings)
        else:
            # 初始化阶段量化损失为0
            quantization_loss = torch.tensor(0.0).to(self.device)
        
        return cluster_ids, quantized_embeddings, reconstruction_loss_embeddings, quantization_loss
    
    def model_step(
        self, 
        model_input: "ItemData"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行一个模型步骤（前向传播 + 损失计算）
        
        Args:
            model_input: ItemData 对象，包含输入数据
            
        Returns:
            cluster_ids: 量化索引，形状为 (batch_size,)
            quantized_embeddings: 量化后的嵌入，形状为 (batch_size, latent_dim)
            total_loss: 总损失值
            quantization_loss: 量化损失值
            reconstruction_loss: 重构损失值
            commitment_loss: commitment损失值
        """
        # 延迟导入以避免循环导入
        from src.data.loading.components.interfaces import ItemData
        
        # 从 ItemData 中提取输入嵌入
        input_embeddings = model_input.transformed_features["input_embedding"].to(
            self.device
        )
        
        # 归一化输入嵌入
        normalized_input_embeddings = self.normalization_layer(input_embeddings)
        
        # 编码：将输入嵌入映射到潜在空间
        encoded_embeddings = self.encoder(normalized_input_embeddings)
        
        # 量化：通过量化层进行量化
        (
            cluster_ids,
            quantized_embeddings,
            reconstruction_loss_embeddings,
            quantization_loss,
        ) = self.forward(encoded_embeddings)
        
        # 检查是否在预测模式（如果没有 trainer，假设不在预测模式）
        try:
            is_predicting = (
                self.trainer is not None 
                and self.trainer.state.fn == TrainerFn.PREDICTING
            )
        except RuntimeError:
            # 如果没有 trainer，假设不在预测模式
            is_predicting = False
        
        # 计算重构损失（如果量化层已初始化且不在预测模式）
        if (
            not is_predicting
            and self.reconstruction_loss_function is not None
            and self.vector_quantization_layer.is_initialized
        ):
            # 解码：将量化后的嵌入映射回原始嵌入空间
            reconstructed_embeddings = self.decoder(reconstruction_loss_embeddings)
            reconstruction_loss = self.reconstruction_loss_function(
                reconstructed_embeddings, normalized_input_embeddings
            )
        else:
            reconstruction_loss = torch.tensor(0.0).to(self.device)
        
        # 计算 commitment loss（用于稳定训练）
        # commitment loss = ||sg(quantized) - encoded||^2
        # 其中 sg 表示 stop gradient
        if (
            not is_predicting
            and self.vector_quantization_layer.is_initialized
            and self.commitment_loss_weight > 0.0
        ):
            commitment_loss = torch.nn.functional.mse_loss(
                quantized_embeddings.detach(), encoded_embeddings
            )
        else:
            commitment_loss = torch.tensor(0.0).to(self.device)
        
        # 计算总损失
        total_loss = (
            self.quantization_loss_weight * quantization_loss
            + self.reconstruction_loss_weight * reconstruction_loss
            + self.commitment_loss_weight * commitment_loss
        )
        
        return cluster_ids, quantized_embeddings, total_loss, quantization_loss, reconstruction_loss, commitment_loss
    
    def training_step(
        self, 
        batch: Tuple["ItemData"], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        训练步骤
        
        Args:
            batch: 训练批次数据（Lightning 会将其包装在元组中，实际数据在 batch[0]）
            batch_idx: 批次索引
            
        Returns:
            loss: 损失值
        """
        # 延迟导入以避免循环导入
        from src.data.loading.components.interfaces import ItemData
        
        # Lightning 在训练时会将批次包装在元组中，我们从位置0获取实际数据
        model_input: ItemData = batch[0]
        
        # 执行模型步骤
        (
            cluster_ids,
            quantized_embeddings,
            total_loss,
            quantization_loss,
            reconstruction_loss,
            commitment_loss,
        ) = self.model_step(model_input)
        
        # 记录训练指标
        self.train_loss(total_loss)
        self.train_quantization_loss(quantization_loss)
        self.train_reconstruction_loss(reconstruction_loss)
        self.train_commitment_loss(commitment_loss)
        
        # 准备日志字典
        train_dict_to_log = {
            "train/loss": self.train_loss,
            "train/quantization_loss": self.train_quantization_loss,
            "train/reconstruction_loss": self.train_reconstruction_loss,
            "train/commitment_loss": self.train_commitment_loss,
        }
        
        # 记录指标
        self.log_dict(
            train_dict_to_log,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        
        return total_loss
    
    def validation_step(
        self, 
        batch: "ItemData", 
        batch_idx: int
    ) -> None:
        """
        验证步骤
        
        Args:
            batch: 验证批次数据（不是元组）
            batch_idx: 批次索引
        """
        # 执行模型步骤
        (
            cluster_ids,
            quantized_embeddings,
            total_loss,
            quantization_loss,
            reconstruction_loss,
            commitment_loss,
        ) = self.model_step(batch)
        
        # 记录验证指标
        self.val_loss(total_loss)
        self.val_quantization_loss(quantization_loss)
        self.val_reconstruction_loss(reconstruction_loss)
        self.val_commitment_loss(commitment_loss)
        
        # 准备日志字典
        val_dict_to_log = {
            "val/loss": self.val_loss,
            "val/quantization_loss": self.val_quantization_loss,
            "val/reconstruction_loss": self.val_reconstruction_loss,
            "val/commitment_loss": self.val_commitment_loss,
        }
        
        # 记录指标
        self.log_dict(
            val_dict_to_log,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
    
    def test_step(
        self, 
        batch: "ItemData", 
        batch_idx: int
    ) -> None:
        """
        测试步骤
        
        Args:
            batch: 测试批次数据（不是元组）
            batch_idx: 批次索引
        """
        # 执行模型步骤
        (
            cluster_ids,
            quantized_embeddings,
            total_loss,
            quantization_loss,
            reconstruction_loss,
            commitment_loss,
        ) = self.model_step(batch)
        
        # 记录测试指标
        self.test_loss(total_loss)
        self.test_quantization_loss(quantization_loss)
        self.test_reconstruction_loss(reconstruction_loss)
        self.test_commitment_loss(commitment_loss)
        
        # 准备日志字典
        test_dict_to_log = {
            "test/loss": self.test_loss,
            "test/quantization_loss": self.test_quantization_loss,
            "test/reconstruction_loss": self.test_reconstruction_loss,
            "test/commitment_loss": self.test_commitment_loss,
        }
        
        # 记录指标
        self.log_dict(
            test_dict_to_log,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
    
    def predict_step(
        self, 
        batch: "ItemData"
    ) -> "OneKeyPerPredictionOutput":
        """
        预测步骤，为输入项生成语义ID
        
        Args:
            batch: 预测批次数据
            
        Returns:
            OneKeyPerPredictionOutput 对象，包含 item_ids 和语义ID
        """
        # 延迟导入以避免循环导入
        from src.data.loading.components.interfaces import ItemData
        from src.models.components.interfaces import OneKeyPerPredictionOutput
        
        # 执行模型步骤，获取 cluster_ids（语义ID）
        (
            cluster_ids,
            quantized_embeddings,
            total_loss,
            quantization_loss,
            reconstruction_loss,
            commitment_loss,
        ) = self.model_step(batch)
        
        # 提取 item_ids，处理可能的 Tensor 类型
        item_ids = [
            item_id.item() if isinstance(item_id, torch.Tensor) else item_id
            for item_id in batch.item_ids
        ]
        
        # 创建 OneKeyPerPredictionOutput 对象
        model_output = OneKeyPerPredictionOutput(
            keys=item_ids,
            predictions=cluster_ids,
            key_name="item_id",
            prediction_name="cluster_ids",
        )
        
        return model_output
    
    def configure_optimizers(
        self
    ) -> Dict[str, Any]:
        """
        配置优化器和学习率调度器
        
        Returns:
            包含优化器和调度器的字典
        """
        if self.optimizer is not None:
            optimizer = self.optimizer(params=self.parameters())
            if self.scheduler is not None:
                scheduler = self.scheduler(optimizer=optimizer)
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val/loss",
                        "interval": "step",
                        "frequency": 1,
                    },
                }
            return {"optimizer": optimizer}
        return {}


import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from src.models.components.interfaces import OneKeyPerPredictionOutput

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        flat_inputs = inputs.view(-1, self.embedding_dim)
        
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_inputs, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        return quantized, vq_loss, encoding_indices.view(inputs.shape[:-1])


class VQVAEQuantization(pl.LightningModule):
    # 核心改动 1：增加 **kwargs 吸收框架自动塞进来的多余参数
    def __init__(self, input_dim=2048, codebook_width=256, commitment_cost=0.25, lr=1e-3, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU()
        )
        self.quantizer = VectorQuantizer(codebook_width, input_dim, commitment_cost)
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, semantic_ids = self.quantizer(z)
        reconstructed = self.decoder(quantized)
        return reconstructed, vq_loss, semantic_ids

    # 把特征提取逻辑单独抽出来，方便训练和预测共用
    def _extract_features(self, batch):
        x = None
        data_obj = batch[0] if isinstance(batch, (list, tuple)) else batch
        
        if hasattr(data_obj, 'transformed_features') and isinstance(data_obj.transformed_features, dict):
            x = data_obj.transformed_features.get('input_embedding')
        
        if x is None:
            for k in dir(data_obj):
                if not k.startswith('_'):
                    v = getattr(data_obj, k)
                    if isinstance(v, torch.Tensor) and v.is_floating_point() and self.hparams.input_dim in v.shape:
                        x = v
                        break
                        
        if x is None:
            raise ValueError("无法提取特征！")
            
        x = x.to(self.device)
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])
        return x.float()

    def training_step(self, batch, batch_idx):
        x = self._extract_features(batch)
        
        reconstructed, vq_loss, _ = self(x)
        recon_loss = F.mse_loss(reconstructed, x)
        loss = recon_loss + vq_loss
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/recon_loss', recon_loss, prog_bar=True)
        self.log('train/vq_loss', vq_loss, prog_bar=True)
        
        return loss

    # 核心改动 2：专为 Inference 编写的预测步骤
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # 1. 提取特征并量化
        x = self._extract_features(batch)
        z = self.encoder(x)
        _, _, semantic_ids = self.quantizer(z)
        
        # 2. 终极自适应商品 ID 提取
        data_obj = batch[0] if isinstance(batch, (list, tuple)) else batch
        item_ids = None
        
        if hasattr(data_obj, 'item_ids') and data_obj.item_ids is not None:
            item_ids = data_obj.item_ids
        elif hasattr(data_obj, 'transformed_features') and isinstance(data_obj.transformed_features, dict):
            for possible_key in ['item_ids', 'id', 'item_id']:
                if possible_key in data_obj.transformed_features:
                    item_ids = data_obj.transformed_features[possible_key]
                    if item_ids is not None:
                        break
        
        if item_ids is None:
            for attr_name in dir(data_obj):
                if not attr_name.startswith('_'):
                    val = getattr(data_obj, attr_name)
                    if isinstance(val, torch.Tensor) and not val.is_floating_point() and val.shape[0] == x.shape[0]:
                        item_ids = val
                        break
        
        if item_ids is None:
            raise ValueError(f"彻底找不到 item_ids！Batch 结构: {dir(data_obj)}")
            
        # ==========================================
        # 核心改动：把 to(torch.int32) 改为 float()，迎合底层框架的 Float 合并机制！
        cluster_ids = semantic_ids.view(-1, 1).float()
        # ==========================================
        
        # 3. 使用框架的标准化输出结构
        return OneKeyPerPredictionOutput(
            keys=item_ids,
            predictions=cluster_ids,
            key_name="item_id",
            prediction_name="cluster_ids"
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
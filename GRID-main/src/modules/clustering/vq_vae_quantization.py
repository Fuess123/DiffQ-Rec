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
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        flat_inputs = inputs.view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_inputs**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_inputs, self.embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        return quantized, vq_loss, encoding_indices.view(inputs.shape[:-1])


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_hierarchies, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        if num_hierarchies < 1:
            raise ValueError("num_hierarchies must be >= 1")

        self.num_hierarchies = num_hierarchies
        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    commitment_cost=commitment_cost,
                )
                for _ in range(num_hierarchies)
            ]
        )

    def forward(self, inputs):
        residual = inputs
        quantized_sum = torch.zeros_like(inputs)
        semantic_ids = []
        total_vq_loss = inputs.new_tensor(0.0)

        for quantizer in self.quantizers:
            quantized, vq_loss, encoding_indices = quantizer(residual)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
            semantic_ids.append(encoding_indices)
            total_vq_loss = total_vq_loss + vq_loss

        semantic_ids = torch.stack(semantic_ids, dim=-1)
        return quantized_sum, total_vq_loss, semantic_ids


class VQVAEQuantization(pl.LightningModule):
    def __init__(
        self,
        input_dim=2048,
        codebook_width=256,
        num_hierarchies=1,
        commitment_cost=0.25,
        lr=1e-3,
        morph_loss_weight=0.0,
        morph_temperature=1.0,
        modal_teacher_embedding_path=None,
        modal_loss_weight=0.0,
        modal_temperature=1.0,
        **kwargs,
    ):
        super().__init__()
        if morph_temperature <= 0:
            raise ValueError("morph_temperature must be > 0")
        if modal_temperature <= 0:
            raise ValueError("modal_temperature must be > 0")

        modal_teacher_embeddings = self._load_modal_teacher_embeddings(modal_teacher_embedding_path)
        modal_teacher_dim = None if modal_teacher_embeddings is None else modal_teacher_embeddings.shape[-1]
        self.save_hyperparameters(ignore=["modal_teacher_embeddings"])

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
        )
        self.quantizer = ResidualVectorQuantizer(
            num_hierarchies=num_hierarchies,
            num_embeddings=codebook_width,
            embedding_dim=input_dim,
            commitment_cost=commitment_cost,
        )

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
        )
        self.modal_projector = nn.Identity() if modal_teacher_dim in (None, input_dim) else nn.Linear(input_dim, modal_teacher_dim)
        self.register_buffer("modal_teacher_embeddings", modal_teacher_embeddings, persistent=False)

    @staticmethod
    def _load_modal_teacher_embeddings(path):
        if path is None or path == "" or str(path).lower() == "none":
            return None

        teacher_embeddings = torch.load(path, map_location="cpu").float()
        if teacher_embeddings.dim() != 2:
            raise ValueError(f"modal teacher embeddings must be 2-D, got {teacher_embeddings.shape}")
        return teacher_embeddings.contiguous()

    def encode_quantize_decode(self, x):
        z = self.encoder(x)
        quantized, vq_loss, semantic_ids = self.quantizer(z)
        reconstructed = self.decoder(quantized)
        return reconstructed, vq_loss, semantic_ids, z, quantized

    def forward(self, x):
        reconstructed, vq_loss, semantic_ids, _, _ = self.encode_quantize_decode(x)
        return reconstructed, vq_loss, semantic_ids

    def _extract_features(self, batch):
        x = None
        data_obj = batch[0] if isinstance(batch, (list, tuple)) else batch

        if hasattr(data_obj, "transformed_features") and isinstance(data_obj.transformed_features, dict):
            x = data_obj.transformed_features.get("input_embedding")

        if x is None:
            for k in dir(data_obj):
                if not k.startswith("_"):
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

    def _extract_item_ids(self, batch, batch_size=None):
        data_obj = batch[0] if isinstance(batch, (list, tuple)) else batch
        item_ids = None

        if hasattr(data_obj, "item_ids") and data_obj.item_ids is not None:
            item_ids = data_obj.item_ids
        elif hasattr(data_obj, "transformed_features") and isinstance(data_obj.transformed_features, dict):
            for possible_key in ["item_ids", "id", "item_id"]:
                if possible_key in data_obj.transformed_features:
                    item_ids = data_obj.transformed_features[possible_key]
                    if item_ids is not None:
                        break

        if item_ids is None:
            for attr_name in dir(data_obj):
                if not attr_name.startswith("_"):
                    val = getattr(data_obj, attr_name)
                    if isinstance(val, torch.Tensor) and not val.is_floating_point():
                        if batch_size is None or val.numel() == batch_size or val.shape[0] == batch_size:
                            item_ids = val
                            break

        if item_ids is None:
            raise ValueError(f"彻底找不到 item_ids！Batch 结构: {dir(data_obj)}")

        item_ids = self._coerce_item_ids_to_tensor(item_ids)
        item_ids = item_ids.to(self.device).long().view(-1)
        if batch_size is not None and item_ids.shape[0] != batch_size:
            item_ids = item_ids[:batch_size]
        return item_ids

    @staticmethod
    def _coerce_item_ids_to_tensor(item_ids):
        if isinstance(item_ids, torch.Tensor):
            return item_ids

        if isinstance(item_ids, (list, tuple)):
            if len(item_ids) == 1 and isinstance(item_ids[0], torch.Tensor):
                return item_ids[0]
            if all(isinstance(v, torch.Tensor) for v in item_ids):
                return torch.cat([v.view(-1) for v in item_ids], dim=0)
            return torch.as_tensor(item_ids)

        return torch.as_tensor(item_ids)

    def _pairwise_similarity(self, embeddings):
        normalized = F.normalize(embeddings, p=2, dim=-1, eps=1e-8)
        return torch.matmul(normalized, normalized.transpose(0, 1)) / self.hparams.morph_temperature

    def _morphology_distillation_loss(self, teacher_embeddings, student_embeddings):
        if self.hparams.morph_loss_weight <= 0 or teacher_embeddings.shape[0] < 2:
            return teacher_embeddings.new_tensor(0.0)

        with torch.no_grad():
            teacher_similarity = self._pairwise_similarity(teacher_embeddings)

        student_similarity = self._pairwise_similarity(student_embeddings)
        return F.mse_loss(student_similarity, teacher_similarity)

    def _modal_distillation_loss(self, item_ids, student_embeddings):
        if self.hparams.modal_loss_weight <= 0 or self.modal_teacher_embeddings is None:
            return student_embeddings.new_tensor(0.0)

        if item_ids.max() >= self.modal_teacher_embeddings.shape[0] or item_ids.min() < 0:
            raise IndexError(
                f"item_ids out of modal teacher embedding range: "
                f"min={item_ids.min().item()}, max={item_ids.max().item()}, "
                f"num_embeddings={self.modal_teacher_embeddings.shape[0]}"
            )

        with torch.no_grad():
            teacher_embeddings = self.modal_teacher_embeddings[item_ids].to(student_embeddings.device)
            teacher_embeddings = F.normalize(teacher_embeddings, p=2, dim=-1, eps=1e-8)

        projected_student = self.modal_projector(student_embeddings)
        projected_student = F.normalize(projected_student, p=2, dim=-1, eps=1e-8)
        cosine_similarity = torch.sum(projected_student * teacher_embeddings, dim=-1)
        return (1.0 - cosine_similarity).mean() / self.hparams.modal_temperature

    def training_step(self, batch, batch_idx):
        x = self._extract_features(batch)
        item_ids = self._extract_item_ids(batch, batch_size=x.shape[0])

        reconstructed, vq_loss, semantic_ids, z, quantized = self.encode_quantize_decode(x)
        recon_loss = F.mse_loss(reconstructed, x)
        morph_loss = self._morphology_distillation_loss(x, z)
        modal_loss = self._modal_distillation_loss(item_ids, z)
        loss = (
            recon_loss
            + vq_loss
            + self.hparams.morph_loss_weight * morph_loss
            + self.hparams.modal_loss_weight * modal_loss
        )

        with torch.no_grad():
            unique_sid_ratio = torch.unique(semantic_ids, dim=0).shape[0] / semantic_ids.shape[0]
            quantized_morph_loss = self._morphology_distillation_loss(x, quantized)
            quantized_modal_loss = self._modal_distillation_loss(item_ids, quantized)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/recon_loss", recon_loss, prog_bar=True)
        self.log("train/vq_loss", vq_loss, prog_bar=True)
        self.log("train/morph_loss", morph_loss, prog_bar=True)
        self.log("train/modal_loss", modal_loss, prog_bar=True)
        self.log("train/quantized_morph_loss", quantized_morph_loss, prog_bar=False)
        self.log("train/quantized_modal_loss", quantized_modal_loss, prog_bar=False)
        self.log("train/unique_sid_ratio", unique_sid_ratio, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = self._extract_features(batch)
        z = self.encoder(x)
        _, _, semantic_ids = self.quantizer(z)
        item_ids = self._extract_item_ids(batch, batch_size=x.shape[0])
        cluster_ids = semantic_ids.long()

        return OneKeyPerPredictionOutput(
            keys=item_ids,
            predictions=cluster_ids,
            key_name="item_id",
            prediction_name="cluster_ids",
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

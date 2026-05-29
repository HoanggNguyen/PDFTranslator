from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import DynamicCache, QuantizedCache


class ContinuousBatchingMixin:
    def pad_left(
        self, key_states: torch.Tensor, value_states: torch.Tensor, padding_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Size is assumed to be (batch_size, num_kv_heads, seq_length, head_dim) - To match huggingface
        key_states_padded = F.pad(
            key_states,
            pad=(0, 0, padding_size, 0),  # (left, right, top, bottom)
            mode="constant",
            value=0,
        )

        value_states_padded = F.pad(
            value_states,
            pad=(0, 0, padding_size, 0),
            mode="constant",
            value=0,
        )

        return key_states_padded, value_states_padded

    # Trim the cache from the left - Useful when longer sequences are evicted and we have long padding on the left
    def trim_left(self, trim_length: int):
        for layer_idx in range(len(self)):
            layer = self.layers[layer_idx]
            
            # Cập nhật số token đã thấy của layer theo chuẩn HF mới
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length -= trim_length
                
            # Cắt tỉa cache (tensors)
            if layer.keys is not None:
                layer.keys = layer.keys[:, :, trim_length:, :]
            if layer.values is not None:
                layer.values = layer.values[:, :, trim_length:, :]

    def get_full_cache(self, layer_idx: int):
        layer = self.layers[layer_idx]
        return layer.keys, layer.values

    def set_full_cache(
        self, layer_idx: int, key_cache: torch.Tensor, value_cache: torch.Tensor
    ):
        layer = self.layers[layer_idx]
        layer.keys = key_cache
        layer.values = value_cache

    def merge(
        self,
        new_cache: "ContinuousBatchingCache",
        merge_idxs: List[int],
        device: torch.device,
    ) -> int:
        assert len(new_cache) == len(self), "The two caches should have the same number of layers"

        # Lấy độ dài chuỗi hiện tại của cache dựa trên hàm chuẩn của Hugging Face
        current_seq_length = self.get_seq_length()
        new_cache_seq_length = new_cache.get_seq_length()
        offset = current_seq_length - new_cache_seq_length  # Thường là dương, nhưng âm vẫn xử lý được

        # Cập nhật số lượng token (cumulative_length) cho từng layer con thay vì dùng _seen_tokens
        for layer_idx in range(len(self)):
            if offset > 0:
                layer_new = new_cache.layers[layer_idx]
                if hasattr(layer_new, "cumulative_length"):
                    layer_new.cumulative_length += offset
            elif offset < 0:
                layer_self = self.layers[layer_idx]
                if hasattr(layer_self, "cumulative_length"):
                    layer_self.cumulative_length += abs(offset)

        merge_idxs_tensor = torch.tensor(merge_idxs, dtype=torch.long, device=device)

        with torch.inference_mode():
            # Như trước, chỉ cần attention mask và position ids đúng thì giá trị padding là gì không quan trọng
            for layer_idx in range(len(self)):
                new_k, new_v = new_cache.get_full_cache(layer_idx)
                if offset > 0:
                    new_k, new_v = self.pad_left(new_k, new_v, offset)

                old_k, old_v = self.get_full_cache(layer_idx)
                if offset < 0:
                    adjusted_key_cache, adjusted_value_cache = self.pad_left(
                        old_k,
                        old_v,
                        abs(offset),
                    )
                else:
                    adjusted_key_cache, adjusted_value_cache = (
                        old_k,
                        old_v,
                    )

                adjusted_key_cache.index_put_((merge_idxs_tensor,), new_k)
                adjusted_value_cache.index_put_((merge_idxs_tensor,), new_v)

                self.set_full_cache(layer_idx, adjusted_key_cache, adjusted_value_cache)

        return offset


class ContinuousBatchingCache(ContinuousBatchingMixin, DynamicCache):
    pass


class ContinuousBatchingQuantizedCache(ContinuousBatchingMixin, QuantizedCache):
    def __init__(self, config, nbits=8, axis_key=1, axis_value=1, q_group_size=64, residual_length=128):
        # Khởi tạo lượng tử hóa phẳng thông qua lớp cha QuantizedCache
        super().__init__(
            backend="hqq",
            config=config,
            nbits=nbits,
            axis_key=axis_key,
            axis_value=axis_value,
            q_group_size=q_group_size,
            residual_length=residual_length
        )
        self.residual_length = residual_length
        self.axis_key = axis_key
        self.axis_value = axis_value
        self.q_group_size = q_group_size

    def get_full_cache(self, layer_idx: int):
        layer = self.layers[layer_idx]
        
        unquant_key_cache = layer.keys
        unquant_value_cache = layer.values
        
        # Gọi hàm _dequantize trực tiếp từ đối tượng layer bên trong
        quant_key_cache = layer._dequantize(layer._quantized_keys)
        quant_value_cache = layer._dequantize(layer._quantized_values)

        full_key_cache = torch.cat([quant_key_cache, unquant_key_cache], dim=-2)
        full_value_cache = torch.cat([quant_value_cache, unquant_value_cache], dim=-2)

        return full_key_cache, full_value_cache

    def set_full_cache(
        self, layer_idx: int, key_cache: torch.Tensor, value_cache: torch.Tensor
    ):
        layer = self.layers[layer_idx]
        
        if key_cache.shape[-2] < self.residual_length:
            layer.keys = torch.tensor([], dtype=key_cache.dtype, device=key_cache.device)
            layer.values = torch.tensor([], dtype=value_cache.dtype, device=value_cache.device)
            
            layer._quantized_keys = layer._quantize(key_cache.contiguous(), axis=self.axis_key)
            layer._quantized_values = layer._quantize(value_cache.contiguous(), axis=self.axis_value)
        else:
            layer.keys = key_cache[:, :, self.residual_length :, :]
            layer.values = value_cache[:, :, self.residual_length :, :]

            quant_key_cache = key_cache[:, :, : self.residual_length, :]
            quant_value_cache = value_cache[:, :, : self.residual_length, :]
            
            layer._quantized_keys = layer._quantize(quant_key_cache, axis=self.axis_key)
            layer._quantized_values = layer._quantize(quant_value_cache, axis=self.axis_value)

    def trim_left(self, trim_length: int):
        if trim_length == 0 or len(self.layers) == 0:
            return

        # Lấy số lượng token hiện tại từ layer đầu tiên trước khi cắt
        current_seen_tokens = getattr(self.layers[0], "cumulative_length", 0)
        new_seen_tokens = current_seen_tokens - trim_length
        
        to_keep = new_seen_tokens - trim_length
        quantized_to_keep = to_keep - self.residual_length

        for layer_idx in range(len(self.layers)):
            layer = self.layers[layer_idx]
            
            # Cập nhật số lượng token đã thấy cho layer lượng tử
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = new_seen_tokens

            if quantized_to_keep > 0:
                dequant_key = layer._dequantize(layer._quantized_keys)[:, :, trim_length:, :]
                dequant_value = layer._dequantize(layer._quantized_values)[:, :, trim_length:, :]
                
                layer._quantized_keys = layer._quantize(dequant_key, axis=self.axis_key)
                layer._quantized_values = layer._quantize(dequant_value, axis=self.axis_value)
            else:
                main_to_keep = new_seen_tokens - trim_length
                main_start_idx = self.residual_length - main_to_keep

                full_key_cache = layer.keys[:, :, main_start_idx:, :]
                full_value_cache = layer.values[:, :, main_start_idx:, :]

                self.set_full_cache(layer_idx, full_key_cache, full_value_cache)
import pytest
import torch
from transformers import DynamicCache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from dlite.modeling.kv_cache import PreallocatedLayer, make_target_cache


def test_append_crop_and_reuse_keep_storage_and_hide_stale_tokens():
    layer = PreallocatedLayer(8)
    keys = torch.randn(1, 2, 3, 4)
    values = torch.randn_like(keys)
    layer.update(keys, values)
    pointers = layer.keys.data_ptr(), layer.values.data_ptr()
    assert layer.keys.shape == (1, 2, 8, 4)
    layer.crop(0)
    assert layer.get_seq_length() == 3
    layer.crop(-2)
    replacement = torch.randn(1, 2, 2, 4)
    actual_keys, actual_values = layer.update(replacement, replacement)
    torch.testing.assert_close(actual_keys, torch.cat([keys[:, :, :1], replacement], dim=2))
    torch.testing.assert_close(actual_values, torch.cat([values[:, :, :1], replacement], dim=2))
    assert layer.get_mask_sizes(2) == (5, 0)
    layer.reset()
    assert layer.get_seq_length() == 0
    layer.update(keys, values)
    assert (layer.keys.data_ptr(), layer.values.data_ptr()) == pointers
    with pytest.raises(ValueError, match="capacity exceeded"):
        layer.update(keys.repeat(1, 1, 3, 1), values.repeat(1, 1, 3, 1))


@torch.inference_mode()
def test_qwen3_prefill_block_verification_and_rollback():
    config = Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=4,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).eval()
    dynamic = DynamicCache(config=config)
    fixed = make_target_cache(model, 16)
    for tokens in [torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6]])]:
        expected = model(tokens, past_key_values=dynamic).logits
        actual = model(tokens, past_key_values=fixed).logits
        torch.testing.assert_close(actual, expected)
    pointers = [(layer.keys.data_ptr(), layer.values.data_ptr()) for layer in fixed.layers]
    dynamic.crop(-2)
    fixed.crop(-2)
    expected = model(torch.tensor([[7, 8]]), past_key_values=dynamic).logits
    actual = model(torch.tensor([[7, 8]]), past_key_values=fixed).logits
    torch.testing.assert_close(actual, expected)
    assert fixed.get_seq_length() == 6
    assert [(layer.keys.data_ptr(), layer.values.data_ptr()) for layer in fixed.layers] == pointers

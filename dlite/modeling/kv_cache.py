"""Fixed-capacity target KV storage, allocated on the first prefill write.

Like LiteSpec's benchmark cache, append in place and expose only the valid
prefix to attention. Adapted to the Transformers 5.x cache API.
"""

from transformers.cache_utils import Cache, StaticLayer


class PreallocatedLayer(StaticLayer):
    is_compileable = False
    is_croppable = True

    def __init__(self, max_cache_len: int):
        if max_cache_len < 1:
            raise ValueError("max_cache_len must be positive")
        super().__init__(max_cache_len=max_cache_len)
        self.logical_length = 0

    def update(self, key_states, value_states, *args, **kwargs):
        append_length = key_states.shape[-2]
        start = self.logical_length
        end = start + append_length
        if end > self.max_cache_len:
            raise ValueError(
                f"KV cache capacity exceeded: need {end}, max {self.max_cache_len}"
            )
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.keys.narrow(2, start, append_length).copy_(key_states)
        self.values.narrow(2, start, append_length).copy_(value_states)
        self.logical_length = end
        return self.keys.narrow(2, 0, end), self.values.narrow(2, 0, end)

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.logical_length + query_length, 0

    def get_seq_length(self) -> int:
        return self.logical_length

    def crop(self, tokens_to_remove: int):
        if tokens_to_remove > 0:
            raise ValueError("Use a negative token count to crop the cache")
        self.logical_length = max(self.logical_length + tokens_to_remove, 0)

    def reset(self):
        self.logical_length = 0


def make_target_cache(target, max_cache_len: int) -> Cache:
    config = target.config.get_text_config()
    layer_types = getattr(config, "layer_types", None)
    if layer_types and any(kind != "full_attention" for kind in layer_types):
        raise ValueError("Preallocated KV currently supports full-attention targets only")
    return Cache(layers=[
        PreallocatedLayer(max_cache_len) for _ in range(config.num_hidden_layers)
    ])

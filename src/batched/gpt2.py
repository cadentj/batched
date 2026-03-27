from __future__ import annotations

from collections.abc import Callable
import types
from typing import Any

import torch as t

HookDispatcher = Callable[[int, str, t.Tensor, t.Tensor | None], t.Tensor]


def patch_gpt2_transformer_for_trimmed_sequences(transformer: Any) -> None:
    """
    Minimal causal forward (no KV cache, no encoder cross-attn, no incremental
    decode). Custom block forwards may shorten the sequence; the final reshape
    uses length after ln_f instead of the input length.
    """
    import torch
    from transformers.modeling_outputs import (
        BaseModelOutputWithPastAndCrossAttentions,
    )

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        cache_position=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        **kwargs,
    ):
        del past_key_values, use_cache, encoder_hidden_states, encoder_attention_mask, inputs_embeds, token_type_ids,
        cache_position

        assert position_ids is not None

        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)


        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        inputs_embeds = self.wte(input_ids)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        hidden_states = self.drop(hidden_states)
        self._batched_position_ids = position_ids

        for block in self.h:
            current_cache_position = None
            hidden_states = block(
                hidden_states,
                None,
                current_cache_position,
                None,
                None,
                encoder_attention_mask=None,
                use_cache=False,
                position_ids=self._batched_position_ids,
                **kwargs,
            )

        hidden_states = self.ln_f(hidden_states)
        seq_len = hidden_states.shape[-2]
        output_shape = (
            (-1,)
            + input_shape[1:-1]
            + (seq_len,)
            + (hidden_states.size(-1),)
        )
        hidden_states = hidden_states.view(output_shape)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )

    transformer.forward = types.MethodType(forward, transformer)


def _create_async_gpt2_block_forward(
    dispatch_hook: HookDispatcher,
):
    def forward(
        self,
        hidden_states: t.FloatTensor | None,
        past_key_values: None = None,
        _cache_position: t.LongTensor | None = None,
        attention_mask: t.FloatTensor | None = None,
        encoder_hidden_states: t.Tensor | None = None,
        encoder_attention_mask: t.FloatTensor | None = None,
        use_cache: bool | None = False,
        **kwargs,
    ) -> t.Tensor:
        del encoder_hidden_states, encoder_attention_mask

        layer = self._batched_layer_idx

        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, _ = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = dispatch_hook(layer, "attn", attn_output, residual)

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = dispatch_hook(layer, "mlp", mlp_output, residual)

        hidden_states = dispatch_hook(layer, "resid", hidden_states, None)
        return hidden_states

    return forward


def patch_gpt2_blocks_for_async(
    transformer: Any,
    dispatch_hook: HookDispatcher,
) -> None:
    for layer, block in enumerate(transformer.h):
        block._batched_layer_idx = layer
        block.forward = types.MethodType(
            _create_async_gpt2_block_forward(dispatch_hook),
            block,
        )


__all__ = [
    "patch_gpt2_blocks_for_async",
    "patch_gpt2_transformer_for_trimmed_sequences",
]

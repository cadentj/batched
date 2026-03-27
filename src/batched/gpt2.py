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
    from transformers.masking_utils import create_causal_mask
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
        del past_key_values, use_cache, encoder_hidden_states, encoder_attention_mask

        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if cache_position is None:
            cache_position = torch.arange(
                0, inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        for block in self.h:
            hidden_states = block(
                hidden_states,
                None,
                cache_position,
                causal_mask,
                None,
                encoder_attention_mask=None,
                use_cache=False,
                position_ids=position_ids,
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
        cache_position: t.LongTensor | None = None,
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

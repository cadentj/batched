import types

import torch as t
from pydantic import BaseModel
from typing import Literal

class Job(BaseModel):
    status: Literal["queued", "running", "paused", "finished"]

    # Sequences in a job are consecutive, so only maintain the start idx
    start_idx: int

    # Keep track of the sequence lengths for each sequence in the job
    seq_lens: list[int]

class Batch(BaseModel):
    jobs: list[Job]


def patch_gpt2_transformer_for_trimmed_sequences(transformer) -> None:
    """
    Minimal causal forward (no KV cache, no encoder cross-attn, no incremental
    decode). Custom block forwards may shorten the sequence; the final reshape
    uses length after ln_f instead of the input length.
    """
    import torch
    from transformers.masking_utils import create_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

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
        # Signature matches HF for model(**inputs); only the causal packed path is implemented.
        del past_key_values, use_cache, encoder_hidden_states, encoder_attention_mask

        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

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

        for _i, block in enumerate(self.h):
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
        output_shape = (-1,) + input_shape[1:-1] + (seq_len,) + (hidden_states.size(-1),)
        hidden_states = hidden_states.view(output_shape)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )

    transformer.forward = types.MethodType(forward, transformer)


def create_gpt2_forward(batch: Batch):
    trimmed: set[int] = set()

    def trim(x: t.Tensor) -> t.Tensor:
        for i, job in enumerate(batch.jobs):
            if job.status == "paused" and i not in trimmed:
                seq_len = sum(job.seq_lens)
                x = t.cat([x[:, :job.start_idx], x[:, job.start_idx + seq_len:]], dim=1)
                trimmed.add(i)
        return x

    def forward(
        self,
        hidden_states: tuple[t.FloatTensor] | None,
        past_key_values: None = None,
        cache_position: t.LongTensor | None = None,
        attention_mask: t.FloatTensor | None = None,
        encoder_hidden_states: t.Tensor | None = None,
        encoder_attention_mask: t.FloatTensor | None = None,
        use_cache: bool | None = False,
        **kwargs,
    ) -> t.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, _ = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs,
        )
        # residual connection
        hidden_states = trim(attn_output + residual)

        # NOTE(cadentj): Removed the encoder cross-attention here.

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = trim(residual + feed_forward_hidden_states)

        return hidden_states

    return forward
import torch
from typing import Dict
from collections import defaultdict

# dependency on accelerate
from accelerate import init_on_device
from transformers.modeling_utils import no_init_weights

from transformers import (
    AutoModelForCausalLM,
    GraniteMoeHybridForCausalLM,
    GraniteMoeHybridConfig,
    BambaForCausalLM,
)

def convert_bamba_state_dict(
    state_dict: Dict,
):
    table = {
        'input_layernorm.weight': None,
        'pre_ff_layernorm.weight': 'post_attention_layernorm.weight',
        'embed_tokens.weight': None,
        'mamba.conv1d.weight': None,
        'mamba.conv1d.bias': None,
        'mamba.in_proj.weight': None,
        'mamba.out_proj.weight': None,
        'mamba.A_log': None,
        'mamba.D': None,
        'mamba.dt_bias': None,
        'self_attn.q_proj.weight': None,
        'self_attn.k_proj.weight': None,
        'self_attn.v_proj.weight': None,
        'self_attn.o_proj.weight': None,
        'feed_forward.down_proj.weight': 'shared_mlp.output_linear.weight',
        (
            'feed_forward.gate_proj.weight',
            'feed_forward.up_proj.weight'
        ): 'shared_mlp.input_linear.weight',
        'mamba.norm.weight': None,
        'final_layernorm.weight': 'norm.weight',
        'lm_head.weight': None,
    }

    fused = defaultdict(dict)

    # sort in decreasing number of levels.
    # - more levels higher priority
    table_keys = sorted(
        table.keys(), key=lambda x: -x.count('.')
    )

    converted_sd: Dict = {}

    for key, tensor in state_dict.items():
        new_key = None

        pattern = None
        is_fused = False
        for ks in table_keys:

            # fused case
            if isinstance(ks, str):
                is_fused = False
                ks = (ks,)
            else:
                is_fused = True

            found = False
            for k in ks:
                if key.endswith(k):
                    pattern = k
                    found = True
                    break
            if found:
                break

        if (
            (is_fused and (ks in table))
            or (pattern in table)
        ):
            if is_fused or table[pattern]:
                new_key = key.replace(
                    pattern, (
                        table[pattern] if not is_fused else 
                        table[ks]
                    )
                )
            else:
                new_key = key
        
        if new_key:
            if not is_fused:
                converted_sd[new_key] = tensor
            else:
                fused[new_key][pattern] = tensor

                if len(fused[new_key]) == len(ks):
                    converted_sd[new_key] = torch.concat(
                        [fused[new_key][k] for k in ks]
                    )

    return converted_sd

def convert_bamba_checkpoint(
    checkpoint_dir: str,
    output_dir: str = None,
):

    # load the bamba checkpoint
    ref = BambaForCausalLM.from_pretrained(checkpoint_dir)

    with init_on_device('cpu'), no_init_weights():
        head_dim = ref.config.hidden_size // ref.config.num_attention_heads
        config = GraniteMoeHybridConfig(
            **{
                **{
                    k:v for k,v in ref.config.to_dict().items()
                    if k not in [
                        'z_loss_coefficient',
                        'attn_layer_indices',
                    ]
                },
                'architectures': ['GraniteMoeHybridForCausalLM'],
                'num_local_experts': 0,
                'shared_intermediate_size': ref.config.intermediate_size,
                'layer_types': [
                    'attention' if i in ref.config.attn_layer_indices else 'mamba' 
                    for i in range(ref.config.num_hidden_layers)
                ],
                'attention_multiplier': head_dim**-0.5,
            },
        )
        # only automodel has a from_config function
        model = AutoModelForCausalLM.from_config(config)

    sd = convert_bamba_state_dict(ref.state_dict())
    model.load_state_dict(sd)

    # save the new checkpoint
    if output_dir is None:
        return model

    model.save_pretrained(output_dir)


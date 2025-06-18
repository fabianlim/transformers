import torch
from typing import Dict
from collections import defaultdict

# dependency on accelerate
from accelerate import init_on_device
from transformers.modeling_utils import no_init_weights

from transformers import (
    AutoModelForCausalLM,
    GraniteMoeHybridConfig,
    BambaForCausalLM,
    BambaConfig,
)

import os, json

def convert_state_dict(
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

def convert_to_granite_model(
    config: BambaConfig,
    state_dict: Dict,
):
    # only supports these now
    position_emb_type = {
        0.: None,
        1.: 'rope'
    }

    with init_on_device('cpu'), no_init_weights():
        head_dim = config.hidden_size // config.num_attention_heads
        config = GraniteMoeHybridConfig(
            **{
                **{
                    k:v for k,v in config.to_dict().items()
                    if k not in [
                        'z_loss_coefficient',
                        'attn_layer_indices',
                        'partial_rotary_factor'
                    ]
                },
                'architectures': ['GraniteMoeHybridForCausalLM'],
                'num_local_experts': 0,
                'shared_intermediate_size': config.intermediate_size,
                'layer_types': [
                    'attention' if i in config.attn_layer_indices else 'mamba' 
                    for i in range(config.num_hidden_layers)
                ],
                'attention_multiplier': head_dim**-0.5,
                'position_embedding_type': position_emb_type[
                    config.partial_rotary_factor
                ]
            },
        )
        # only automodel has a from_config function
        model = AutoModelForCausalLM.from_config(config)

    sd = convert_state_dict(state_dict)
    model.load_state_dict(sd)

    # save the new checkpoint
    return model

def main(
    convert_from_dir: str,
    output_to_dir: str,
    dtype: str = 'fp16',
    strategy: str = 'mamba_ssm',
):
    DTYPES = {
        'fp16': torch.float16,
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
    }
    assert dtype in DTYPES
    assert strategy in {'mamba_ssm', 'bamba'}

    # get a bamba checkpoint first
    if strategy == 'mamba_ssm':
        from transformers.models.bamba.convert_mamba_ssm_checkpoint import (
            convert_ssm_config_to_hf_config,
            convert_state_dict_from_mamba_ssm
        )

        # FIXME: Not handled
        token_ids = {}
        # some code replication

        unsettables = {
            "mamba_d_head": 64,
            "mamba_d_state": 128,
            "mamba_n_groups": 1,
            "rms_norm_eps": 1e-5,
        }

        config_path = os.path.join(convert_from_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as json_file:
            config_data = json.load(json_file)

        # have the fix here
        if 'attn_rotary_emb' in config_data:
            head_dim = (
                config_data['hidden_size'] //
                config_data['num_attention_heads']
            )
            config_data['partial_rotary_config'] = (
                config_data['attn_rotary_emb'] //
                head_dim
            )
            del config_data['attn_rotary_emb']

        config = convert_ssm_config_to_hf_config(
            config_ssm=config_data,
            **token_ids,
            **unsettables,
        )

        # Load state dict of the original model and transfer to hf model
        state_dict = torch.load(
            os.path.join(convert_from_dir, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )

        state_dict = convert_state_dict_from_mamba_ssm(state_dict)
        
    else:
        ref = BambaForCausalLM.from_pretrained(convert_from_dir)
        config = ref.config
        state_dict = ref.state_dict()

    # convert
    print ('converting to granite')
    model = convert_to_granite_model(
        config, state_dict,
    )

    # change dtype
    print ('change dtype to', dtype)
    model = model.to(DTYPES[dtype])

    # save
    print ('save model to', output_to_dir)
    model.save_pretrained(output_to_dir)

if __name__ == '__main__':

    import fire
    fire.Fire(main)


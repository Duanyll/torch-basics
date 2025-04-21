# Flux Control Adapters

## Official BFL Flux Control LoRA

Below is the list of fine-tuned layers in the official [BFL Flux Control LoRA](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev-lora/blob/main/flux1-canny-dev-lora.safetensors).

| BFL Name                                         | Diffusers Name                                   | LoRA or Full |
| ------------------------------------------------ | ------------------------------------------------ | ------------ |
| `double_blocks.*.img_attn.norm.key_norm.scale`   | `transformer_blocks.*.attn.norm_q.weight`        | Full         |
| `double_blocks.*.img_attn.norm.value_norm.scale` | `transformer_blocks.*.attn.norm_k.weight`        | Full         |
| `double_blocks.*.img_attn.qkv`                   | `transformer_blocks.*.attn.to_q/k/v`             | LoRA         |
| `double_blocks.*.img_attn.proj`                  | `transformer_blocks.*.to_out.0`                  | LoRA         |
| `double_blocks.*.img_mlp.0`                      | `transformer_blocks.*.ff.net.0.proj`             | LoRA         |
| `double_blocks.*.img_mlp.2`                      | `transformer_blocks.*.ff.net.2`                  | LoRA         |
| `double_blocks.*.img_mod.lin`                    | `transformer_blocks.*.norm1.linear`              | LoRA         |
| `double_blocks.*.txt_attn.norm.key_norm.scale`   | `transformer_blocks.*.attn.norm_added_q.weight`  | Full         |
| `double_blocks.*.txt_attn.norm.value_norm.scale` | `transformer_blocks.*.attn.norm_added_k.weight`  | Full         |
| `double_blocks.*.txt_attn.qkv`                   | `transformer_blocks.*.attn.add_q/k/v_proj`       | LoRA         |
| `double_blocks.*.txt_attn.proj`                  | `transformer_blocks.*.to_add_out`                | LoRA         |
| `double_blocks.*.txt_mlp.0`                      | `transformer_blocks.*.ff_context.net.0.proj`     | LoRA         |
| `double_blocks.*.txt_mlp.2`                      | `transformer_blocks.*.ff_context.net.2`          | LoRA         |
| `double_blocks.*.txt_mod.lin`                    | `transformer_blocks.*.norm1_context.linear`      | LoRA         |
| `single_blocks.*.linear1`                        | `single_transformer_blocks.*.attn.to_q/k/v`      | LoRA         |
|                                                  | `single_transformer_blocks.*.proj_mlp`           | LoRA [^1]    |
| `single_blocks.*.linear2`                        | `single_transformer_blocks.*.proj_out`           | LoRA         |
| `single_blocks.*.modulation.lin`                 | `single_transformer_blocks.*.norm.linear`        | LoRA         |
| `single_blocks.*.norm.key_norm.scale`            | `single_transformer_blocks.*.attn.norm_q.weight` | Full         |
| `single_blocks.*.norm.value_norm.scale`          | `single_transformer_blocks.*.attn.norm_k.weight` | Full         |
| `final_layer.adaLN_modulation.1`                 | `norm_out.linear`                                | LoRA         |
| `final_layer.linear`                             | `proj_out`                                       | LoRA         |
| `guidance_in.in_layer`                           | `time_text_embed.guidance_embedder.linear_1`     | LoRA         |
| `guidance_in.out_layer`                          | `time_text_embed.guidance_embedder.linear_2`     | LoRA         |
| `img_in`                                         | `x_embedder`                                     | LoRA [^2]    |
| `time_in.in_layer`                               | `time_text_embed.timestep_embedder.linear_1`     | LoRA         |
| `time_in.out_layer`                              | `time_text_embed.timestep_embedder.linear_2`     | LoRA         |
| `txt_in`                                         | `context_embedder`                               | LoRA         |
| `vector_in.in_layer`                             | `text_embedder.linear_1.lora_A.weight`           | LoRA         |
| `vector_in.out_layer`                            | `text_embedder.linear_1.lora_A.weight`           | LoRA         |

[^1]: These four layers are concatenated together in BFL.
[^2]: The rank is set to 128, which is equivalent to a full weight layer. All additional parameters here can be fully trained.

This indicates that in the official LoRA, all linear layers are fine-tuned, and RMSNorm scales are also fine-tuned. 
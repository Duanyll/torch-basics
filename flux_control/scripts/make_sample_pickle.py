import argparse
import pickle
import torch
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline

DTYPE = torch.bfloat16

def make_sample_pickle(input_file, outfile):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=DTYPE, transformer=None, vae=None
    ).to(device)

    output = {}
    # For each line in the input file
    with open(input_file, "r") as f:
        for i, line in enumerate(f):
            prompt = line.strip()
            if not prompt:
                continue
            print(f"Processing prompt {i}: {prompt}")
            prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt,
                device=device,
            )
            output[f"{i:04d}"] = {
                "prompt_embeds": prompt_embeds.cpu(),
                "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
            }
            
    with open(outfile, "wb") as f:
        pickle.dump(output, f)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create sample pickle file")
    parser.add_argument("input_file", type=str, help="Path to the input text file")
    parser.add_argument("outfile", type=str, help="Path to the output pickle file")
    args = parser.parse_args()
    
    make_sample_pickle(args.input_file, args.outfile)
import os
import torch
from diffusers import StableDiffusionPipeline
from typing import Dict, Set, List
import inspect

# Set environment variables to control model caching
os.environ["TORCH_HOME"] = "./cache/torch"
os.environ["HF_HOME"] = "./cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "./cache/huggingface"

def get_nested_attr(obj, attr_path):
    """Get nested attribute from an object using dot notation"""
    attrs = attr_path.split('.')
    for attr in attrs:
        # Handle cases where attribute names contain numbers
        if attr.replace('_', '').isdigit():
            obj = obj[int(attr)]
        else:
            obj = getattr(obj, attr)
    return obj

def test_attention_hook(module, input, output):
    """Hook function specifically for attention layers"""
    print(f"\nAttention Hook called on {module.__class__.__name__}")
    print(f"Module path: {module._module_path}")  # Custom attribute we'll add
    print(f"Input shapes: {[x.shape if isinstance(x, torch.Tensor) else type(x) for x in input]}")
    if isinstance(output, torch.Tensor):
        print(f"Output shape: {output.shape}")
    elif isinstance(output, tuple):
        print(f"Output shapes: {[x.shape if isinstance(x, torch.Tensor) else type(x) for x in output]}")
    return output

def attach_hooks_to_attention_layers(pipe: StableDiffusionPipeline, target_layers: List[str]):
    """
    Attach hooks to specific attention layers in the pipeline
    Returns list of attached hooks for later removal
    """
    hooks = []
    
    for layer_path in target_layers:
        breakpoint()
        try:
            # Convert path format from down_blocks_0_attentions_0 to down_blocks.0.attentions.0
            # normalized_path = layer_path.replace('_', '.')
            normalized_path = layer_path
            module = get_nested_attr(pipe.unet, normalized_path)
            
            # Add path information to the module for reference
            module._module_path = layer_path
            
            # Register the hook
            hook = module.register_forward_hook(test_attention_hook)
            hooks.append(hook)
            print(f"Successfully attached hook to {layer_path}")
            
        except Exception as e:
            print(f"Failed to attach hook to {layer_path}: {str(e)}")
    
    return hooks

def main():
    # Create cache directories if they don't exist
    os.makedirs("./cache/torch", exist_ok=True)
    os.makedirs("./cache/huggingface", exist_ok=True)
    
    print("Loading pipeline (this might take a while the first time)...")
    # Load the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "valhalla/sd-wikiart-v2",
        torch_dtype=torch.float16,
        cache_dir="./cache/huggingface"
    ).to("cuda")
    print("Pipeline loaded successfully!")
    
    # List of specific attention layers to hook
    target_layers = [
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q",
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k",
        # "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v",
        # "mid_block.attentions.0.transformer_blocks.0.attn2.to_q",
        # "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_q"
    ]
    
    # Convert dots to underscores for consistent naming
    target_layers_normalized = [layer.replace('.', '_') for layer in target_layers]
    
    print("\nAttaching hooks to attention layers...")
    hooks = attach_hooks_to_attention_layers(pipe, target_layers)
    
    # Test the hooks with a simple inference
    prompt = "a photo of a cat"
    print("\nRunning inference with hooks...")
    with torch.no_grad():
        pipe(prompt, num_inference_steps=2)
    
    # Remove the hooks
    print("\nRemoving hooks...")
    for hook in hooks:
        hook.remove()
    print("Hooks removed")

    # Print cache location information
    print("\nCache Locations:")
    print(f"Torch Cache: {os.environ['TORCH_HOME']}")
    print(f"HuggingFace Cache: {os.environ['HF_HOME']}")
    print(f"Transformers Cache: {os.environ['TRANSFORMERS_CACHE']}")

if __name__ == "__main__":
    main()
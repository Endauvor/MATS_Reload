from sae_lens import SAE, HookedSAETransformer
import torch


def load_sae(layer_idx):
    """
    Load SAE for a specific layer with error handling and debug info.

    Args:
        layer_idx: Layer index to load SAE for

    Returns:
        SAE object if successful, None if failed
    """
    try:
    release = "gemma-scope-2b-pt-res-canonical"                              
    sae_id = f"layer_{layer_idx}/width_65k/canonical"  

        print(f"Loading SAE for layer {layer_idx}...")
        print(f"  Release: {release}")
        print(f"  SAE ID: {sae_id}")
        
        sae, cfg, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
        release=release, sae_id=sae_id, device="cpu"
    )
        
        # Move to CPU explicitly if not already there
        sae = sae.to("cpu")
        
        print(f"  ✅ Successfully loaded SAE for layer {layer_idx}")
        print(f"  SAE device: {sae.device if hasattr(sae, 'device') else 'unknown'}")
        print(f"  Config device: {cfg.device if hasattr(cfg, 'device') else 'unknown'}")
        print(f"  Sparsity: {sparsity}")
        print()

    return sae

    except Exception as e:
        print(f"  ❌ Failed to load SAE for layer {layer_idx}")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {str(e)}")
        print()
        return None


def load_all_saes(max_layers=26):
    """
    Load all SAEs with comprehensive error handling.
    
    Args:
        max_layers: Maximum number of layers to attempt loading
        
    Returns:
        dict: {layer_idx: sae} for successfully loaded SAEs
    """
    loaded_saes = {}
    failed_layers = []
    
    print(f"Starting to load SAEs for {max_layers} layers...")
    print("=" * 60)
    
    for layer_idx in range(max_layers):
        sae = load_sae(layer_idx)
        
        if sae is not None:
            loaded_saes[layer_idx] = sae
        else:
            failed_layers.append(layer_idx)
    
    # Summary
    print("=" * 60)
    print("LOADING SUMMARY:")
    print(f"✅ Successfully loaded: {len(loaded_saes)} SAEs")
    print(f"❌ Failed to load: {len(failed_layers)} SAEs")
    
    if loaded_saes:
        print(f"Successfully loaded layers: {sorted(loaded_saes.keys())}")
    
    if failed_layers:
        print(f"Failed layers: {failed_layers}")
    
    return loaded_saes, failed_layers


if __name__ == "__main__":
    # Force CPU usage
    torch.set_default_device("cpu")
    
    # Load all SAEs
    loaded_saes, failed_layers = load_all_saes(max_layers=26)
    
    # Additional memory info
    print("\nMEMORY INFO:")
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        print("CUDA available: No")
    
    print(f"Total loaded SAEs: {len(loaded_saes)}")
    
    # Test one SAE if any loaded successfully
    if loaded_saes:
        test_layer = min(loaded_saes.keys())
        test_sae = loaded_saes[test_layer]
        print(f"\nTesting SAE for layer {test_layer}:")
        print(f"  Type: {type(test_sae)}")
        print(f"  Device: {test_sae.device if hasattr(test_sae, 'device') else 'unknown'}")
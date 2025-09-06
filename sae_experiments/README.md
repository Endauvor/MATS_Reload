# SAE Experiments

This directory contains experimental analysis scripts for Sparse Autoencoder (SAE) research. **Most experiments are designed for GPT-2-small** unless otherwise specified.

## Experiment Scripts

### Core Analysis Scripts

**`sae6.py`** (1,845 lines) - **Main Analysis Module**
- Layer contribution analysis to model predictions
- SAE feature evolution tracking across prompt lengths
- Trajectory origin analysis with heatmap generation
- KL divergence analysis for attention heads
- Comprehensive feature decomposition and statistics
- Functions: `analyze_layer_contributions_to_prediction()`, `plot_indices_evolution()`, `analyze_trajectory_origins()`

**`sae.py`** - **Basic SAE Decomposition**
- Simple SAE feature extraction and analysis
- Cosine similarity analysis of top features
- Basic feature decomposition workflow
- Good starting point for understanding SAE analysis

### Multi-length Analysis Scripts

**`sae8.py`** - **Multi-length Graph Generation**
- Generates trajectory graphs for different prompt lengths
- GPT-2-small focused analysis
- Creates comprehensive graph structures with metadata
- Outputs multi-length trajectory JSON files

**`sae9.py`** - **Extended Multi-length Analysis** 
- **Gemma-2-2b** analysis (exception to GPT-2-small focus)
- Token generation and extended sequence analysis
- Creates evolutionary patterns for all prompt lengths
- Top-400 feature tracking across layers

### Specialized Analysis Scripts

**`sae1.py`** - **Foundation Analysis**
- Early experimental framework
- Token contribution analysis
- Basic feature similarity computations
- Statistical analysis of SAE features

**`sae3.py`** - **Advanced Feature Analysis**
- Extended experimental features
- Complex feature interaction analysis
- Advanced statistical computations

**`sae4.py`** - **Intermediate Analysis**
- Mid-level experimental features
- Feature evolution patterns
- Statistical analysis extensions

**`sae5.py`** - **Enhanced Analysis**
- Advanced feature tracking
- Extended statistical analysis
- Feature contribution computations

**`sae7.py`** - **Graph-focused Analysis**
- Similar to sae8.py but with different parameters
- Multi-length trajectory analysis
- Graph generation with filtering options

### Utility Scripts

**`api_gemma.py`** - **Gemma API Integration**
- Feature label retrieval for Gemma models
- API utilities for feature descriptions

**`permutations_analysis.py`** - **Permutation Analysis**
- Statistical analysis of feature permutations
- Advanced experimental analysis tools

## Common Parameters

Most scripts use similar parameters:
- **Model**: GPT-2-small (default)
- **Sentence**: "The final correct answer: Max, Mark and Smith are in the empty dark room. Smith left. Mark gave flashlight to"
- **k_top**: 40-400 (number of top features to track)
- **Layers**: 12 (for GPT-2-small)
- **Device**: CUDA when available

## Output Directories

Scripts typically create outputs in:
- `att_exp/evolution/` - Evolution analysis results
- `att_exp/token_ads/` - Token attention analysis
- `charts/` - Basic chart outputs

## Usage Examples

```bash
# Basic SAE analysis
python sae.py

# Main comprehensive analysis
python sae6.py

# Multi-length analysis for GPT-2
python sae8.py

# Extended analysis for Gemma-2b
python sae9.py
```

## Key Functions

- **Feature Evolution**: Track how SAE features change across prompt lengths
- **Trajectory Analysis**: Identify paths of feature emergence and disappearance
- **Layer Contributions**: Analyze how different layers contribute to final predictions
- **Statistical Analysis**: Compute similarities, norms, and other feature statistics
- **Visualization**: Generate heatmaps, plots, and interactive visualizations

## Notes

- Most experiments are configured for research on GPT-2-small
- `sae9.py` is the main exception, working with Gemma-2-2b
- Scripts generate both data files and visualizations
- Many experiments build upon findings from `sae6.py`
- Experiments focus on residual stream features (resid_post) 
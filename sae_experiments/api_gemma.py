import json
import os
from datetime import datetime
from sae6 import analyze_trajectory_origins
from graphs1 import analyze_trajectory_graph
from hooked import TransformerLensTransparentLlm
import torch
from collections import defaultdict
import requests

def get_feature_label_gemma_65k(layer_idx, sae_idx):

    try:
        url = f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{layer_idx}-gemmascope-res-65k/{sae_idx}"
        response = requests.get(
            url,
            headers={"x-api-key": "sk-np-mr5bqLiLmlKbpioVXmplW5m0BkNmtBSEdTsruyS55HA0"},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            pos_str = data.get("pos_str", [])
            explanations = data.get("explanations", [])
            description = explanations[0]["description"]
            
            return (pos_str, description) if pos_str else ("N/A", "N/A")
        else:
            return "API Err"
    except Exception:
        return "Req Err"
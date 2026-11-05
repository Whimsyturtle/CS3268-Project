"""Contrastive Activation Addition for text-only Gemma 3. See README.md."""

METHOD = "caa_last_response_token_v1"


def model_layer(model, layer):
    """Resolve the configured decoder-block output location."""
    if model.config.model_type != "gemma3_text":
        raise ValueError("CAA requires a text-only Gemma 3 model")
    layers = model.model.layers
    if len(layers) != model.config.num_hidden_layers:
        raise ValueError("Gemma layer count differs from its configuration")
    if type(layer) is not int or not 0 <= layer < len(layers):
        raise ValueError("CAA layer is outside the model")
    return layers[layer]

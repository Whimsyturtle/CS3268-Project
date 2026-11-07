# ICV

For cluster runs, first follow the [shared runtime setup](../../docs/Cluster_Runtime.md).
Construct the ICV from the shared demonstration pairs with

```sh
uv run --locked python -m steering.icv.build
```

The command uses the model in `configs/main.json` and writes `vector.pt` and
`config.json` to `steering_vectors/icv/`.

The implementation follows the authors' [released code at b187c63](https://github.com/shengliu66/ICV/tree/b187c6387ae41097f5e38182d5a8df7d06ca9ec4).
It constructs one joint principal component from final-token differences across
the embedding and decoder layers, then steers the MLP outputs.

Tune its strength with the [common tuner](../README.md).

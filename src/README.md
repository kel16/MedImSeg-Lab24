# Pseudocolor visualization lab
## End-to-end pipeline
Switch to script's folder:
```
cd src/dimensionality_reduction/
```

The following loads an autoencoder of channel depth = 32 from a file and saves figures for 3 selected layer names:

```python run.py -c 32 --layers 'model.model.1.submodule.0.conv.unit0.adn.A, model.model.1.submodule.0.conv.unit1.adn.A, model.model.1.submodule.0.conv.unit3.adn.A' --autoencoder_name 'encoder_22-09_04-03-2025'```

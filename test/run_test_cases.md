# Test cases

In our test cases we proceed with the following structure in mind:
1. use the data generation pipeline we developed to produce an example of a dataset (smaller compared to the one we used to train the network);
2. evaluate the performance of the models we trained on the new dataset;
3. show the training pipeline building a new (weak) model on the new (small) dataset.

## Test 1. Cantilevered beam

> In this case the upper plate is clamped at the left end and free at the right one (i.e. it's a cantilever).

First you can create the __dataset__. Some plots of the domain and the (traditional) numerical solutions corresponding a combination of parameters present in the dataset you've just generated.
If you want you can play with the number of __workers__ to make the generation faster. 

```bash
python -m src.data.generate --folder "test/test1" --data_file "test/test1.csv" --geometry_input "geometries/cantilever1.geo" --workers 2
```

> Note the real bottleneck of the process in this case is the mesh generation section, since the FOM (Full Order Model) is quite fast, since it's only a laplacian.

Then you can __evaluate__ the model for the electrostatic potential we already trained on the test set you generated.
```bash
python -m src.surrogate.evaluate --folder "test/test1" --model_path "models/potential1.keras" --target "potential" 
```

> Since the process data splitting is random, we actually don't know if the model has already seen the data we are generating here. So this is an unbiased estimator of the real performance of the model.

Now you can visualize with some plots the predictions that the model can make.
```bash
python -m src.surrogate.predict --folder "test/test1" --model_path "models/potential1.keras"  --target "potential" 
```

You can do the same for with the model for the normal derivative of the potential on the boundary of the upper plate.
```bash
python -m src.surrogate.evaluate --folder "test/test1" --model_path "models/derivative1.keras" --target "normal_derivative" 
```
```bash
python -m src.surrogate.predict --folder "test/test1" --model_path "models/derivative1.keras"  --target "normal_derivative" 
```


If you want you can try training the model, with the dedicated module.
```bash
python -m src.surrogate.train --folder "test/test1" --model_path "models/potential.keras" --target "potential"
```

```bash
python -m src.surrogate.train --folder "test/test1" --model_path "models/derivative.keras" --target "normal_derivative"
```

You can also use a GPU if available, but you must satisfy the CUDA requirements yourself by properly configuring your environment. This setup may not be straightforward and, if done incorrectly, can lead to warnings. For example, you may want to run in the terminal something like:
```bash
mamba activate env-name
mamba install cuda-cudart cuda-version=12 -y
```

Logs of the training are available and you can open them in your browser using tensorboard:
```bash
tensorboard --logdir logs
```

## Test 2: Bigger Deformations

You can just do the same as above but with the following changes:
- use `test2` instead of `test1`;
- changing the reference geometry to `geometries/cantilever2.geo`; 
- consider now the models `models/potential2.keras` and `models/derivative2.keras`.

## Test 3: Clamped-Clamped Beam

You can just do the same as above but with the following changes:
- use `test3` instead of `test1`;
- changing the reference geometry to `geometries/clamped.geo`; 
- consider now the models `models/potential3.keras` and `models/derivative3.keras`.

# Introduction

## Strucutre of the repository
```bash
Geom-DeepONet
├── README.md
├── data (module)
├── surrogate (module)
├── docs (...)
├── geometries (...)
├── models (...)
├── test (...)
├── build_docs.sh
└── setup.sh
```

## Setup
> Conda (or Mamba) is required.

Create the project environment by running this command (or the equivalent Conda command):
```bash
mamba env create -f environment.yml
```

## Test cases
You can find the instruction on how to run them in the readme in the `test` folder, or equivalently in the documentation provided in the `docs` folder.

## Documentation
The documentation is present in the `docs` folder in HTML format, and online [here](https://alessandropedone.github.io/deeponet-for-mems/).
The deeper version with private functions is compiled, 
but if you want the more compact version with only the public functions, 
you can just run the following command, and select the right option, to generate a new version of the documentation.
```bash
./build_docs.sh
```
# Installation
## Python packages and Rust crates
In addition to the packages specified in `requirements.txt`, this project requires:
* Madnis dev branch installation
```bash
git clone git@github.com:madgraph-ml/madnis.git
```
rev:
```bash
062515f559e847f6ede1b407a555c66b5c7602a3
```
Install in editable mode
* Momtrop fork
```bash
git clone git@github.com:Fink-Nic/momtrop.git
```
* Gammaloop hedge_numerator branch installation, rev:
```bash
5b5fcff74707769b1d99e9509a2deb72770183bb
```
## File system setup
* Place the contents of `gammaloop_files` into your gammaloop folder
* Specify the absolute paths in `resources/PATHS.json`:
    * `tropnis`: path to your tropnis folder.
    * `gammaloop_states`: path to your gammaloop states folder. It should be called `PATH_TO_GAMMALOOP_FOLDER/gl_states`.
    * `path_to_dot`: should be left at `processes/amplitudes`, unless you are changing the structure of your gammaloop states. 
    * `default_settings`: path to the default settings file. It should be called `PATH_TO_TROPNIS_FOLDER/settings/default.toml`

# Workflow
## Setting up your gammaloop states
You can use the makefile provided inside of `gammaloop_files` in order to:
* Install gammaloop:
```bash
make -f gl.makefile build_all
```
* Generate a state from one of the provided runcards inside of `gammaloop_files/runcards`. Example:
```bash
make -f gl.makefile generate NAME=scalar_box
```
* Integrate an example:
```bash
make -f gl.makefile integrate NAME=scalar_box
```

## Settings files
Inside of `settings`, there are a number of examples. A list and explanation of the parameters can be found within `settings/default.toml`. It is recommended to follow the naming convention of the examples, this also applies to the names of the gammaloop runcards, `.dot` files and state folders.

## Testing the compatibility with tropnis
Example:
```bash
python3 gammaloop_state_test.py -s settings/scalar_box.toml
```


## Minimal working training run, including output of data
Example:
```bash
python3 training_prog.py -s settings/scalar_box.toml
```

## Quirks
Follow the example in `training_prog.py`:
* In order to properly catch `KeyboardInterrupt` exceptions, set the `SIGINT` signal handling via
```python
signal.signal(signal.SIGINT, signal.default_int_handler)
```
* Once you are done (or want to prematurely exit), call `end()` on your `GammaLoopIntegrand` objects:
```python
integrand.end()
```
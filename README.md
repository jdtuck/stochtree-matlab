# stochtree for MATLAB

A MATLAB wrapper for [stochtree](https://github.com/StochasticTree/stochtree) — stochastic tree ensembles (BART / XBART) for supervised learning and causal inference.

This sits alongside the project's existing R and Python interfaces and wraps the same C++ core, so results are comparable across all three languages.

## Requirements

- MATLAB R2018a or newer (the gateway is built with the `-R2018a` interleaved complex API)
- A C++17 compiler configured for MEX (`mex -setup C++`)
- A recursive clone of the stochtree repository

## Installation

```bash
git clone --recurse-submodules https://github.com/jdtuck/stochtree-matlab
```

The `--recurse-submodules` flag matters: Eigen, Boost.Math, fmt and fast_double_parser are git submodules. If you already cloned without it, run `git submodule update --init --recursive` inside the checkout.

Then, in MATLAB:

```matlab
cd stochtree-matlab
build_stochtree('StochtreeRoot', '/path/to/stochtree');
```

If the stochtree checkout sits next to or inside this folder, `build_stochtree()` finds it on its own. The build compiles the nine core stochtree sources plus the gateway into a single MEX file, which takes a couple of minutes. Add `'OpenMP', true` for multithreaded sampling — note that Apple Clang ships without OpenMP, so macOS users generally need Homebrew LLVM or libomp for that.

Verify with:

```matlab
runtests('test/testStochtree.m')
```

## Quick start

### BART

```matlab
n = 500;
X = rand(n, 5);
y = 10*sin(pi*X(:,1).*X(:,2)) + 20*(X(:,3)-0.5).^2 + 10*X(:,4) + randn(n,1);

model = stochtree.bart(X, y, 'NumMCMC', 500, 'RandomSeed', 1);

yhat = mean(model.YHatTrain, 2);              % posterior mean
ci   = quantile(model.YHatTrain, [.025 .975], 2);   % 95% interval
out  = model.predict(Xnew);                   % draws for new data
```

### Bayesian Causal Forest

```matlab
propensity = 0.25 + 0.5*X(:,1);
Z = double(rand(n,1) < propensity);
y = mu(X) + tau(X).*Z + 0.5*randn(n,1);

model = stochtree.bcf(X, Z, y, 'PropensityTrain', propensity, 'NumMCMC', 500);

tauHat = mean(model.TauHatTrain, 2);   % posterior mean CATE
model.ateSummary()                     % ATE with a credible interval
```

Omit `'PropensityTrain'` and the propensity score is estimated internally with a small BART model, which is then carried along so `predict` works without you supplying scores.

See `demo/demoBART.m`, `demo/demoBCF.m` and `demo/demoLowLevel.m`.

## Design

The wrapper has the same two-layer shape as the R and Python packages.

**Layer 1 — `src/mex_stochtree.cpp`.** A single MEX gateway that dispatches on a leading command string and hands back opaque `uint64` handles into a process-global object registry. It is the direct counterpart of `src/py_stochtree.cpp` (pybind11) and `src/cpp11.cpp` (R), and the argument order of `sampler_sample_one_iteration` deliberately matches `ForestSamplerCpp::SampleOneIteration` so the three bindings stay easy to diff.

**Layer 2 — `+stochtree/`.** MATLAB `handle` classes wrapping each C++ object, plus `bart` and `bcf` implementing the sampling loops, prior calibration and standardization in MATLAB — exactly as `bart.py` and `bcf.py` do in Python.

Three details worth knowing:

- **Column-major throughout.** MATLAB matrices go straight into stochtree with `is_row_major = false`; no transposes, no hidden copies beyond the one stochtree makes itself.
- **Errors unwind properly.** Failures are raised as C++ exceptions and only converted to `mexErrMsgIdAndTxt` at the outermost frame. Calling MATLAB's error function from deep inside would `longjmp` past live destructors and leak the registry.
- **Handles survive `clear`.** The MEX file locks itself in memory while any object is alive, so `clear mex` cannot pull the registry out from under live MATLAB objects. Destructors tolerate an already-unloaded gateway, which is what happens during MATLAB shutdown.

## API reference

### High-level

| Function | Purpose |
|---|---|
| `stochtree.bart(X, y, ...)` | BART / XBART regression, returns `stochtree.BARTModel` |
| `stochtree.bcf(X, Z, y, ...)` | Bayesian Causal Forest, returns `stochtree.BCFModel` |

Both accept `NumGFR` (grow-from-root warm start), `NumBurnin` and `NumMCMC`. `NumGFR = 0` gives plain BART; `NumMCMC = 0` gives plain XBART. See `help stochtree.bart` for the full option list.

`BARTModel` provides `predict`, `posteriorMean`, `credibleInterval`, `variableSplitCounts`, `summary`, `toStruct`/`fromStruct` and `save`/`load`. `BCFModel` adds `ateSamples`, `ateSummary` and `posteriorCATE`.

Models serialize through the C++ JSON representation, so `save`/`load` produce ordinary `.mat` files with no live handles in them.

### Low-level

`stochtree.Dataset`, `Residual`, `RNG`, `Forest`, `ForestContainer`, `ForestSampler`, `GlobalVarianceModel`, `LeafVarianceModel`.

These map one-to-one onto the C++ classes and let you build custom models — see `demo/demoLowLevel.m`, which reimplements plain BART in about 40 lines. Leaf model codes are `0` constant, `1` univariate regression, `2` multivariate regression, `3` log-linear variance, `4` cloglog ordinal. Feature type codes are `0` numeric, `1` ordered categorical, `2` unordered categorical.

Sample indices in the MATLAB classes are 1-based and translated to the C++ 0-based indices internally.

## Validation

The gateway and both sampling loops were verified end to end before release:

| Check | Result |
|---|---|
| BART on Friedman (n=500, σ=1) | R² 0.966, recovered σ 0.958 |
| BCF on confounded data (n=1000) | corr(τ̂, τ) 0.978, ATE 2.06 vs true 2.02, σ 0.485 vs true 0.5 |
| Forest JSON round-trip | bit-identical predictions |
| Handle lifetime | stale handles, double-free and type confusion all rejected |

`test/testStochtree.m` covers the same ground from MATLAB, plus leaf-basis regression, heteroskedastic BART, seed reproducibility, serialization and input validation.

## Current limitations

- **Random effects are not exposed.** The `RandomEffects*` classes in the C++ core have no MATLAB binding yet. Adding them means about six more object types in the gateway.
- **Single chain only.** `num_chains > 1` from the Python package is not implemented. Run `bart` several times with different seeds and pool the draws if you need multiple chains.
- **No ordinal or probit outcomes.** The gateway exposes the cloglog ordinal leaf model (code 4), but `bart` does not yet drive it, and the probit path is not implemented.
- **Categorical preprocessing is manual.** Pass integer-coded categoricals and mark them via `'CategoricalIndices'` or `'FeatureTypes'`; there is no equivalent of Python's `CovariateTransformer` yet.
- **`SampleWeights` and a variance forest conflict.** Initializing a variance forest overwrites the dataset's variance weights. This matches the Python package's behaviour, but it means the two options should not be combined.

## License

The wrapper follows stochtree's MIT license.

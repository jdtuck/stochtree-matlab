/*!
 * mex_stochtree.cpp
 *
 * MATLAB MEX gateway for the stochtree C++ core
 * (https://github.com/StochasticTree/stochtree).
 *
 * This file plays the same role for MATLAB that `src/py_stochtree.cpp`
 * (pybind11) plays for Python and `src/cpp11.cpp` plays for R: it exposes the
 * stochtree sampling primitives -- datasets, residuals, RNGs, forests, forest
 * containers, forest samplers and variance models -- as opaque handles that a
 * high-level language can drive.
 *
 * Design notes
 * ------------
 *  * A single MEX entry point dispatches on a leading command string, because
 *    MATLAB gives us one shared library per MEX file and we want all objects to
 *    live in one registry.
 *  * Objects are stored in a process-global registry keyed by a uint64 handle.
 *    MATLAB-side `handle` classes own a handle and free it in their destructor.
 *  * The MEX file is locked in memory while any object is alive so that
 *    `clear mex` cannot pull the library (and the registry) out from under
 *    live MATLAB objects.
 *  * MATLAB arrays are column-major, so every call into stochtree that takes an
 *    `is_row_major` flag is passed `false`.
 *  * Numeric buffers handed to stochtree are copied first. stochtree copies
 *    incoming data into its own Eigen storage, but MATLAB's inputs are const
 *    (and may be shared copy-on-write), so we never hand out a pointer into
 *    `prhs` that a callee could conceivably write through.
 *
 * Build with `build_stochtree.m`.
 */

#include "mex.h"

#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/ensemble.h>
#include <stochtree/leaf_model.h>
#include <stochtree/meta.h>
#include <stochtree/partition_tracker.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using data_size_t = StochTree::data_size_t;

/* ------------------------------------------------------------------ */
/* Small MEX helpers                                                    */
/* ------------------------------------------------------------------ */

inline double* RealPtr(const mxArray* a) {
#if defined(MX_HAS_INTERLEAVED_COMPLEX) && MX_HAS_INTERLEAVED_COMPLEX
  return mxGetDoubles(a);
#else
  return mxGetPr(a);
#endif
}

inline double* RealPtrMutable(mxArray* a) {
#if defined(MX_HAS_INTERLEAVED_COMPLEX) && MX_HAS_INTERLEAVED_COMPLEX
  return mxGetDoubles(a);
#else
  return mxGetPr(a);
#endif
}

/*!
 * Errors are raised as C++ exceptions and converted to `mexErrMsgIdAndTxt` at the
 * outermost frame. Calling `mexErrMsgIdAndTxt` directly from deep inside the
 * gateway would longjmp past live destructors and leak the objects we hold.
 */
struct MexFailure {
  std::string id;
  std::string msg;
};

[[noreturn]] void Fail(const char* id, const std::string& msg) { throw MexFailure{id, msg}; }

void RequireArgs(int nrhs, int expected, const char* command) {
  // nrhs counts the command string itself.
  if (nrhs != expected) {
    Fail("stochtree:nargin",
         std::string("Command '") + command + "' expects " +
             std::to_string(expected - 1) + " argument(s), got " +
             std::to_string(nrhs - 1) + ".");
  }
}

std::string GetString(const mxArray* a, const char* what) {
  if (!mxIsChar(a)) Fail("stochtree:type", std::string(what) + " must be a char row vector.");
  char* raw = mxArrayToString(a);
  if (raw == nullptr) Fail("stochtree:type", std::string("Could not read ") + what + ".");
  std::string out(raw);
  mxFree(raw);
  return out;
}

double GetScalar(const mxArray* a, const char* what) {
  if (!mxIsNumeric(a) && !mxIsLogical(a)) {
    Fail("stochtree:type", std::string(what) + " must be numeric or logical.");
  }
  if (mxGetNumberOfElements(a) != 1) {
    Fail("stochtree:type", std::string(what) + " must be a scalar.");
  }
  return mxGetScalar(a);
}

int GetInt(const mxArray* a, const char* what) {
  return static_cast<int>(GetScalar(a, what));
}

bool GetBool(const mxArray* a, const char* what) {
  return GetScalar(a, what) != 0.0;
}

/*! Copy a MATLAB real double array into a std::vector<double> (column-major order preserved). */
std::vector<double> GetDoubleVector(const mxArray* a, const char* what) {
  if (!mxIsDouble(a) || mxIsComplex(a)) {
    Fail("stochtree:type", std::string(what) + " must be a real double array.");
  }
  const mwSize n = mxGetNumberOfElements(a);
  std::vector<double> out(static_cast<size_t>(n));
  if (n > 0) std::memcpy(out.data(), RealPtr(a), sizeof(double) * static_cast<size_t>(n));
  return out;
}

/*! Copy a MATLAB numeric array into a std::vector<int>. Accepts double or int32. */
std::vector<int> GetIntVector(const mxArray* a, const char* what) {
  if (!mxIsNumeric(a) || mxIsComplex(a)) {
    Fail("stochtree:type", std::string(what) + " must be a real numeric array.");
  }
  const mwSize n = mxGetNumberOfElements(a);
  std::vector<int> out(static_cast<size_t>(n));
  if (mxIsDouble(a)) {
    const double* p = RealPtr(a);
    for (mwSize i = 0; i < n; i++) out[i] = static_cast<int>(p[i]);
  } else if (mxIsInt32(a)) {
    const int32_T* p = static_cast<const int32_T*>(mxGetData(a));
    for (mwSize i = 0; i < n; i++) out[i] = static_cast<int>(p[i]);
  } else {
    Fail("stochtree:type", std::string(what) + " must be double or int32.");
  }
  return out;
}

mxArray* MakeScalar(double v) { return mxCreateDoubleScalar(v); }
mxArray* MakeBool(bool v) { return mxCreateLogicalScalar(v ? 1 : 0); }

/*! Build an n x 1 MATLAB column vector from a std::vector<double>. */
mxArray* MakeColumn(const std::vector<double>& v) {
  mxArray* out = mxCreateDoubleMatrix(static_cast<mwSize>(v.size()), 1, mxREAL);
  if (!v.empty()) std::memcpy(RealPtrMutable(out), v.data(), sizeof(double) * v.size());
  return out;
}

mxArray* MakeColumn(const std::vector<int>& v) {
  mxArray* out = mxCreateDoubleMatrix(static_cast<mwSize>(v.size()), 1, mxREAL);
  double* p = RealPtrMutable(out);
  for (size_t i = 0; i < v.size(); i++) p[i] = static_cast<double>(v[i]);
  return out;
}

/*! Convert a vector of integer codes into stochtree FeatureType enums. */
std::vector<StochTree::FeatureType> ToFeatureTypes(const std::vector<int>& codes) {
  std::vector<StochTree::FeatureType> out(codes.size());
  for (size_t i = 0; i < codes.size(); i++) {
    if (codes[i] < 0 || codes[i] > 2) {
      Fail("stochtree:featureType",
           "Feature type codes must be 0 (numeric), 1 (ordered categorical) or 2 "
           "(unordered categorical).");
    }
    out[i] = static_cast<StochTree::FeatureType>(codes[i]);
  }
  return out;
}

StochTree::ModelType ToModelType(int leaf_model_int) {
  switch (leaf_model_int) {
    case 0: return StochTree::ModelType::kConstantLeafGaussian;
    case 1: return StochTree::ModelType::kUnivariateRegressionLeafGaussian;
    case 2: return StochTree::ModelType::kMultivariateRegressionLeafGaussian;
    case 3: return StochTree::ModelType::kLogLinearVariance;
    case 4: return StochTree::ModelType::kCloglogOrdinal;
    default:
      Fail("stochtree:leafModel",
           "leaf_model must be 0 (constant Gaussian), 1 (univariate regression), 2 "
           "(multivariate regression), 3 (log-linear variance) or 4 (cloglog ordinal).");
  }
}

/* ------------------------------------------------------------------ */
/* Handle registry                                                      */
/* ------------------------------------------------------------------ */

enum ObjTag {
  kTagDataset = 1,
  kTagResidual,
  kTagRng,
  kTagForest,
  kTagContainer,
  kTagSampler,
  kTagGlobalVar,
  kTagLeafVar
};

const char* TagName(int tag) {
  switch (tag) {
    case kTagDataset: return "Dataset";
    case kTagResidual: return "Residual";
    case kTagRng: return "RNG";
    case kTagForest: return "Forest";
    case kTagContainer: return "ForestContainer";
    case kTagSampler: return "ForestSampler";
    case kTagGlobalVar: return "GlobalVarianceModel";
    case kTagLeafVar: return "LeafVarianceModel";
    default: return "Unknown";
  }
}

struct BaseObj {
  explicit BaseObj(int t) : tag(t) {}
  virtual ~BaseObj() = default;
  int tag;
};

struct DatasetObj : BaseObj {
  DatasetObj() : BaseObj(kTagDataset) {}
  StochTree::ForestDataset value;
};

struct ResidualObj : BaseObj {
  ResidualObj(double* data, data_size_t n) : BaseObj(kTagResidual), value(data, n) {}
  StochTree::ColumnVector value;
};

struct RngObj : BaseObj {
  explicit RngObj(int seed) : BaseObj(kTagRng) {
    if (seed < 0) {
      std::random_device rd;
      value.seed(rd());
    } else {
      value.seed(static_cast<std::mt19937::result_type>(seed));
    }
  }
  std::mt19937 value;
};

struct ForestObj : BaseObj {
  ForestObj(int num_trees, int output_dim, bool leaf_constant, bool exponentiated)
      : BaseObj(kTagForest), value(num_trees, output_dim, leaf_constant, exponentiated) {}
  StochTree::TreeEnsemble value;
};

struct ContainerObj : BaseObj {
  ContainerObj(int num_trees, int output_dim, bool leaf_constant, bool exponentiated)
      : BaseObj(kTagContainer), value(num_trees, output_dim, leaf_constant, exponentiated) {}
  StochTree::ForestContainer value;
};

struct SamplerObj : BaseObj {
  SamplerObj() : BaseObj(kTagSampler) {}
  std::unique_ptr<StochTree::ForestTracker> tracker;
  std::unique_ptr<StochTree::TreePrior> prior;
};

struct GlobalVarObj : BaseObj {
  GlobalVarObj() : BaseObj(kTagGlobalVar) {}
  StochTree::GlobalHomoskedasticVarianceModel value;
};

struct LeafVarObj : BaseObj {
  LeafVarObj() : BaseObj(kTagLeafVar) {}
  StochTree::LeafNodeHomoskedasticVarianceModel value;
};

std::unordered_map<uint64_t, std::unique_ptr<BaseObj>>& Registry() {
  static std::unordered_map<uint64_t, std::unique_ptr<BaseObj>> registry;
  return registry;
}

uint64_t NextHandle() {
  static uint64_t counter = 0;
  return ++counter;
}

bool g_locked = false;

void LockIfNeeded() {
  if (!g_locked) {
    mexLock();
    g_locked = true;
  }
}

void UnlockIfEmpty() {
  if (g_locked && Registry().empty()) {
    mexUnlock();
    g_locked = false;
  }
}

mxArray* Register(std::unique_ptr<BaseObj> obj) {
  LockIfNeeded();
  const uint64_t id = NextHandle();
  Registry()[id] = std::move(obj);
  mxArray* out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
  *static_cast<uint64_T*>(mxGetData(out)) = static_cast<uint64_T>(id);
  return out;
}

uint64_t ReadHandle(const mxArray* a) {
  if (mxGetNumberOfElements(a) != 1) {
    Fail("stochtree:handle", "Object handle must be a scalar.");
  }
  if (mxIsUint64(a)) return static_cast<uint64_t>(*static_cast<const uint64_T*>(mxGetData(a)));
  if (mxIsDouble(a)) return static_cast<uint64_t>(mxGetScalar(a));
  Fail("stochtree:handle", "Object handle must be a uint64 scalar.");
}

template <typename T>
T* Fetch(const mxArray* a, int tag, const char* what) {
  const uint64_t id = ReadHandle(a);
  auto it = Registry().find(id);
  if (it == Registry().end()) {
    Fail("stochtree:handle",
         std::string(what) + " handle is stale or invalid (the object has been deleted).");
  }
  if (it->second->tag != tag) {
    Fail("stochtree:handle",
         std::string(what) + " handle refers to a " + TagName(it->second->tag) +
             ", expected a " + TagName(tag) + ".");
  }
  return static_cast<T*>(it->second.get());
}

void CleanupAll() {
  Registry().clear();
  UnlockIfEmpty();
}

void AtExitHandler() { Registry().clear(); }

/* ------------------------------------------------------------------ */
/* Command implementations                                              */
/* ------------------------------------------------------------------ */

/* ---- Dataset ---- */

void CmdDatasetCreate(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 1, "dataset_create");
  plhs[0] = Register(std::make_unique<DatasetObj>());
}

void CmdDatasetAddCovariates(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "dataset_add_covariates");
  auto* obj = Fetch<DatasetObj>(prhs[1], kTagDataset, "dataset");
  if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || mxGetNumberOfDimensions(prhs[2]) != 2) {
    Fail("stochtree:type", "Covariates must be a real 2-D double matrix.");
  }
  std::vector<double> buf = GetDoubleVector(prhs[2], "covariates");
  const auto n = static_cast<data_size_t>(mxGetM(prhs[2]));
  const int p = static_cast<int>(mxGetN(prhs[2]));
  obj->value.AddCovariates(buf.data(), n, p, /*is_row_major=*/false);
}

void CmdDatasetAddBasis(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "dataset_add_basis");
  auto* obj = Fetch<DatasetObj>(prhs[1], kTagDataset, "dataset");
  std::vector<double> buf = GetDoubleVector(prhs[2], "basis");
  const auto n = static_cast<data_size_t>(mxGetM(prhs[2]));
  const int k = static_cast<int>(mxGetN(prhs[2]));
  obj->value.AddBasis(buf.data(), n, k, /*is_row_major=*/false);
}

void CmdDatasetUpdateBasis(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "dataset_update_basis");
  auto* obj = Fetch<DatasetObj>(prhs[1], kTagDataset, "dataset");
  std::vector<double> buf = GetDoubleVector(prhs[2], "basis");
  const auto n = static_cast<data_size_t>(mxGetM(prhs[2]));
  const int k = static_cast<int>(mxGetN(prhs[2]));
  obj->value.UpdateBasis(buf.data(), n, k, /*is_row_major=*/false);
}

void CmdDatasetAddWeights(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "dataset_add_weights");
  auto* obj = Fetch<DatasetObj>(prhs[1], kTagDataset, "dataset");
  std::vector<double> buf = GetDoubleVector(prhs[2], "weights");
  obj->value.AddVarianceWeights(buf.data(), static_cast<data_size_t>(buf.size()));
}

void CmdDatasetUpdateWeights(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 4, "dataset_update_weights");
  auto* obj = Fetch<DatasetObj>(prhs[1], kTagDataset, "dataset");
  std::vector<double> buf = GetDoubleVector(prhs[2], "weights");
  const bool exponentiate = GetBool(prhs[3], "exponentiate");
  obj->value.UpdateVarWeights(buf.data(), static_cast<data_size_t>(buf.size()), exponentiate);
}

void CmdDatasetNumRows(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "dataset_num_rows");
  plhs[0] = MakeScalar(Fetch<DatasetObj>(prhs[1], kTagDataset, "dataset")->value.NumObservations());
}

void CmdDatasetNumCovariates(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "dataset_num_covariates");
  plhs[0] = MakeScalar(Fetch<DatasetObj>(prhs[1], kTagDataset, "dataset")->value.NumCovariates());
}

void CmdDatasetNumBasis(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "dataset_num_basis");
  plhs[0] = MakeScalar(Fetch<DatasetObj>(prhs[1], kTagDataset, "dataset")->value.NumBasis());
}

void CmdDatasetHasBasis(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "dataset_has_basis");
  plhs[0] = MakeBool(Fetch<DatasetObj>(prhs[1], kTagDataset, "dataset")->value.HasBasis());
}

void CmdDatasetHasWeights(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "dataset_has_weights");
  plhs[0] = MakeBool(Fetch<DatasetObj>(prhs[1], kTagDataset, "dataset")->value.HasVarWeights());
}

void CmdDatasetGetCovariates(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "dataset_get_covariates");
  auto* obj = Fetch<DatasetObj>(prhs[1], kTagDataset, "dataset");
  const auto n = static_cast<mwSize>(obj->value.NumObservations());
  const auto p = static_cast<mwSize>(obj->value.NumCovariates());
  plhs[0] = mxCreateDoubleMatrix(n, p, mxREAL);
  double* out = RealPtrMutable(plhs[0]);
  for (mwSize j = 0; j < p; j++) {
    for (mwSize i = 0; i < n; i++) {
      out[j * n + i] = obj->value.CovariateValue(static_cast<data_size_t>(i), static_cast<int>(j));
    }
  }
}

void CmdDatasetGetBasis(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "dataset_get_basis");
  auto* obj = Fetch<DatasetObj>(prhs[1], kTagDataset, "dataset");
  const auto n = static_cast<mwSize>(obj->value.NumObservations());
  const auto k = static_cast<mwSize>(obj->value.NumBasis());
  plhs[0] = mxCreateDoubleMatrix(n, k, mxREAL);
  double* out = RealPtrMutable(plhs[0]);
  for (mwSize j = 0; j < k; j++) {
    for (mwSize i = 0; i < n; i++) {
      out[j * n + i] = obj->value.BasisValue(static_cast<data_size_t>(i), static_cast<int>(j));
    }
  }
}

void CmdDatasetGetWeights(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "dataset_get_weights");
  auto* obj = Fetch<DatasetObj>(prhs[1], kTagDataset, "dataset");
  const auto n = static_cast<mwSize>(obj->value.NumObservations());
  plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
  double* out = RealPtrMutable(plhs[0]);
  for (mwSize i = 0; i < n; i++) out[i] = obj->value.VarWeightValue(static_cast<data_size_t>(i));
}

/* ---- Residual ---- */

void CmdResidualCreate(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "residual_create");
  std::vector<double> buf = GetDoubleVector(prhs[1], "outcome");
  plhs[0] = Register(std::make_unique<ResidualObj>(buf.data(),
                                                   static_cast<data_size_t>(buf.size())));
}

void CmdResidualGet(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "residual_get");
  auto* obj = Fetch<ResidualObj>(prhs[1], kTagResidual, "residual");
  Eigen::VectorXd& v = obj->value.GetData();
  const auto n = static_cast<mwSize>(obj->value.NumRows());
  plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
  double* out = RealPtrMutable(plhs[0]);
  for (mwSize i = 0; i < n; i++) out[i] = v(static_cast<Eigen::Index>(i));
}

void CmdResidualReplace(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "residual_replace");
  auto* obj = Fetch<ResidualObj>(prhs[1], kTagResidual, "residual");
  std::vector<double> buf = GetDoubleVector(prhs[2], "values");
  obj->value.OverwriteData(buf.data(), static_cast<data_size_t>(buf.size()));
}

void CmdResidualAdd(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "residual_add");
  auto* obj = Fetch<ResidualObj>(prhs[1], kTagResidual, "residual");
  std::vector<double> buf = GetDoubleVector(prhs[2], "values");
  obj->value.AddToData(buf.data(), static_cast<data_size_t>(buf.size()));
}

void CmdResidualSubtract(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "residual_subtract");
  auto* obj = Fetch<ResidualObj>(prhs[1], kTagResidual, "residual");
  std::vector<double> buf = GetDoubleVector(prhs[2], "values");
  obj->value.SubtractFromData(buf.data(), static_cast<data_size_t>(buf.size()));
}

/* ---- RNG ---- */

void CmdRngCreate(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "rng_create");
  plhs[0] = Register(std::make_unique<RngObj>(GetInt(prhs[1], "seed")));
}

/* ---- Forest (the "active" forest) ---- */

void CmdForestCreate(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 5, "forest_create");
  plhs[0] = Register(std::make_unique<ForestObj>(
      GetInt(prhs[1], "num_trees"), GetInt(prhs[2], "output_dimension"),
      GetBool(prhs[3], "is_leaf_constant"), GetBool(prhs[4], "is_exponentiated")));
}

void CmdForestNumTrees(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "forest_num_trees");
  plhs[0] = MakeScalar(Fetch<ForestObj>(prhs[1], kTagForest, "forest")->value.NumTrees());
}

void CmdForestOutputDimension(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "forest_output_dimension");
  plhs[0] = MakeScalar(Fetch<ForestObj>(prhs[1], kTagForest, "forest")->value.OutputDimension());
}

void CmdForestSetRootValue(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "forest_set_root_value");
  Fetch<ForestObj>(prhs[1], kTagForest, "forest")
      ->value.SetLeafValue(GetScalar(prhs[2], "leaf_value"));
}

void CmdForestSetRootVector(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "forest_set_root_vector");
  auto* obj = Fetch<ForestObj>(prhs[1], kTagForest, "forest");
  std::vector<double> v = GetDoubleVector(prhs[2], "leaf_vector");
  obj->value.SetLeafVector(v);
}

void CmdForestResetRoot(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "forest_reset_root");
  Fetch<ForestObj>(prhs[1], kTagForest, "forest")->value.ResetRoot();
}

void CmdForestResetFromContainer(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 4, "forest_reset_from_container");
  auto* forest = Fetch<ForestObj>(prhs[1], kTagForest, "forest");
  auto* container = Fetch<ContainerObj>(prhs[2], kTagContainer, "container");
  const int idx = GetInt(prhs[3], "sample_index");
  if (idx < 0 || idx >= container->value.NumSamples()) {
    Fail("stochtree:index", "sample_index is out of range for this forest container.");
  }
  forest->value.ReconstituteFromForest(*(container->value.GetEnsemble(idx)));
}

void CmdForestPredict(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "forest_predict");
  auto* forest = Fetch<ForestObj>(prhs[1], kTagForest, "forest");
  auto* dataset = Fetch<DatasetObj>(prhs[2], kTagDataset, "dataset");
  std::vector<double> raw = forest->value.Predict(dataset->value);
  plhs[0] = MakeColumn(raw);
}

void CmdForestPredictRaw(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "forest_predict_raw");
  auto* forest = Fetch<ForestObj>(prhs[1], kTagForest, "forest");
  auto* dataset = Fetch<DatasetObj>(prhs[2], kTagDataset, "dataset");
  std::vector<double> raw = forest->value.PredictRaw(dataset->value);
  const auto n = static_cast<mwSize>(dataset->value.NumObservations());
  const auto d = static_cast<mwSize>(forest->value.OutputDimension());
  plhs[0] = mxCreateDoubleMatrix(n, d, mxREAL);
  double* out = RealPtrMutable(plhs[0]);
  // raw is laid out row-major as [obs, output_dim]; MATLAB wants column-major.
  for (mwSize i = 0; i < n; i++) {
    for (mwSize j = 0; j < d; j++) out[j * n + i] = raw[i * d + j];
  }
}

void CmdForestNumLeaves(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "forest_num_leaves");
  plhs[0] = MakeScalar(Fetch<ForestObj>(prhs[1], kTagForest, "forest")->value.NumLeaves());
}

void CmdForestSumLeafSquared(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "forest_sum_leaf_squared");
  plhs[0] = MakeScalar(Fetch<ForestObj>(prhs[1], kTagForest, "forest")->value.SumLeafSquared());
}

/* ---- Forest container (stored posterior samples) ---- */

void CmdContainerCreate(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 5, "container_create");
  plhs[0] = Register(std::make_unique<ContainerObj>(
      GetInt(prhs[1], "num_trees"), GetInt(prhs[2], "output_dimension"),
      GetBool(prhs[3], "is_leaf_constant"), GetBool(prhs[4], "is_exponentiated")));
}

void CmdContainerNumSamples(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "container_num_samples");
  plhs[0] = MakeScalar(Fetch<ContainerObj>(prhs[1], kTagContainer, "container")->value.NumSamples());
}

void CmdContainerNumTrees(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "container_num_trees");
  plhs[0] = MakeScalar(Fetch<ContainerObj>(prhs[1], kTagContainer, "container")->value.NumTrees());
}

void CmdContainerOutputDimension(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "container_output_dimension");
  plhs[0] = MakeScalar(
      Fetch<ContainerObj>(prhs[1], kTagContainer, "container")->value.OutputDimension());
}

void CmdContainerDeleteSample(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "container_delete_sample");
  auto* obj = Fetch<ContainerObj>(prhs[1], kTagContainer, "container");
  const int idx = GetInt(prhs[2], "sample_index");
  if (idx < 0 || idx >= obj->value.NumSamples()) {
    Fail("stochtree:index", "sample_index is out of range for this forest container.");
  }
  obj->value.DeleteSample(idx);
}

void CmdContainerPredict(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "container_predict");
  auto* container = Fetch<ContainerObj>(prhs[1], kTagContainer, "container");
  auto* dataset = Fetch<DatasetObj>(prhs[2], kTagDataset, "dataset");
  const auto n = static_cast<mwSize>(dataset->value.NumObservations());
  const auto s = static_cast<mwSize>(container->value.NumSamples());
  std::vector<double> raw = container->value.Predict(dataset->value);
  // raw is column-major [obs, sample], which is exactly MATLAB's layout.
  plhs[0] = mxCreateDoubleMatrix(n, s, mxREAL);
  if (!raw.empty()) {
    std::memcpy(RealPtrMutable(plhs[0]), raw.data(), sizeof(double) * raw.size());
  }
}

void CmdContainerPredictRaw(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "container_predict_raw");
  auto* container = Fetch<ContainerObj>(prhs[1], kTagContainer, "container");
  auto* dataset = Fetch<DatasetObj>(prhs[2], kTagDataset, "dataset");
  const auto n = static_cast<mwSize>(dataset->value.NumObservations());
  const auto s = static_cast<mwSize>(container->value.NumSamples());
  const auto d = static_cast<mwSize>(container->value.OutputDimension());
  std::vector<double> raw = container->value.PredictRaw(dataset->value);
  mwSize dims[3] = {n, d, s};
  plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
  double* out = RealPtrMutable(plhs[0]);
  // raw[k*(d*n) + i*d + j] is sample k, observation i, output dim j.
  for (mwSize k = 0; k < s; k++) {
    for (mwSize i = 0; i < n; i++) {
      for (mwSize j = 0; j < d; j++) {
        out[k * (d * n) + j * n + i] = raw[k * (d * n) + i * d + j];
      }
    }
  }
}

void CmdContainerPredictRawSingle(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 4, "container_predict_raw_single");
  auto* container = Fetch<ContainerObj>(prhs[1], kTagContainer, "container");
  auto* dataset = Fetch<DatasetObj>(prhs[2], kTagDataset, "dataset");
  const int idx = GetInt(prhs[3], "sample_index");
  if (idx < 0 || idx >= container->value.NumSamples()) {
    Fail("stochtree:index", "sample_index is out of range for this forest container.");
  }
  const auto n = static_cast<mwSize>(dataset->value.NumObservations());
  const auto d = static_cast<mwSize>(container->value.OutputDimension());
  std::vector<double> raw = container->value.PredictRaw(dataset->value, idx);
  plhs[0] = mxCreateDoubleMatrix(n, d, mxREAL);
  double* out = RealPtrMutable(plhs[0]);
  for (mwSize i = 0; i < n; i++) {
    for (mwSize j = 0; j < d; j++) out[j * n + i] = raw[i * d + j];
  }
}

void CmdContainerAddToForest(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 4, "container_add_to_forest");
  auto* obj = Fetch<ContainerObj>(prhs[1], kTagContainer, "container");
  obj->value.AddToForest(GetInt(prhs[2], "sample_index"), GetScalar(prhs[3], "constant_value"));
}

void CmdContainerMultiplyForest(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 4, "container_multiply_forest");
  auto* obj = Fetch<ContainerObj>(prhs[1], kTagContainer, "container");
  obj->value.MultiplyForest(GetInt(prhs[2], "sample_index"), GetScalar(prhs[3], "multiplier"));
}

/*! Tally how often each feature is split on within a single tree ensemble. */
void AccumulateSplitCounts(StochTree::TreeEnsemble* ensemble, std::vector<int>& counts) {
  const int num_trees = ensemble->NumTrees();
  for (int i = 0; i < num_trees; i++) {
    StochTree::Tree* tree = ensemble->GetTree(i);
    std::vector<std::int32_t> split_nodes = tree->GetInternalNodes();
    for (std::size_t j = 0; j < split_nodes.size(); j++) {
      const auto split_feature = tree->SplitIndex(split_nodes[j]);
      if (split_feature >= 0 && static_cast<std::size_t>(split_feature) < counts.size()) {
        counts[static_cast<std::size_t>(split_feature)]++;
      }
    }
  }
}

void CmdContainerSplitCounts(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "container_overall_split_counts");
  auto* obj = Fetch<ContainerObj>(prhs[1], kTagContainer, "container");
  const int num_features = GetInt(prhs[2], "num_features");
  if (num_features <= 0) Fail("stochtree:value", "num_features must be positive.");
  std::vector<int> counts(static_cast<std::size_t>(num_features), 0);
  const int num_samples = obj->value.NumSamples();
  for (int i = 0; i < num_samples; i++) {
    AccumulateSplitCounts(obj->value.GetEnsemble(i), counts);
  }
  plhs[0] = MakeColumn(counts);
}

void CmdContainerForestSplitCounts(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 4, "container_forest_split_counts");
  auto* obj = Fetch<ContainerObj>(prhs[1], kTagContainer, "container");
  const int idx = GetInt(prhs[2], "sample_index");
  const int num_features = GetInt(prhs[3], "num_features");
  if (idx < 0 || idx >= obj->value.NumSamples()) {
    Fail("stochtree:index", "sample_index is out of range for this forest container.");
  }
  if (num_features <= 0) Fail("stochtree:value", "num_features must be positive.");
  std::vector<int> counts(static_cast<std::size_t>(num_features), 0);
  AccumulateSplitCounts(obj->value.GetEnsemble(idx), counts);
  plhs[0] = MakeColumn(counts);
}

void CmdContainerDumpJson(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "container_dump_json");
  auto* obj = Fetch<ContainerObj>(prhs[1], kTagContainer, "container");
  plhs[0] = mxCreateString(obj->value.DumpJsonString().c_str());
}

void CmdContainerLoadJson(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "container_load_json");
  auto* obj = Fetch<ContainerObj>(prhs[1], kTagContainer, "container");
  std::string s = GetString(prhs[2], "json string");
  obj->value.LoadFromJsonString(s);
}

void CmdContainerSaveJsonFile(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "container_save_json_file");
  auto* obj = Fetch<ContainerObj>(prhs[1], kTagContainer, "container");
  obj->value.SaveToJsonFile(GetString(prhs[2], "filename"));
}

void CmdContainerLoadJsonFile(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "container_load_json_file");
  auto* obj = Fetch<ContainerObj>(prhs[1], kTagContainer, "container");
  obj->value.LoadFromJsonFile(GetString(prhs[2], "filename"));
}

/* ---- Forest sampler ---- */

void CmdSamplerCreate(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 9, "sampler_create");
  auto* dataset = Fetch<DatasetObj>(prhs[1], kTagDataset, "dataset");
  std::vector<StochTree::FeatureType> feature_types =
      ToFeatureTypes(GetIntVector(prhs[2], "feature_types"));
  const int num_trees = GetInt(prhs[3], "num_trees");
  const auto num_obs = static_cast<data_size_t>(GetScalar(prhs[4], "num_observations"));
  const double alpha = GetScalar(prhs[5], "alpha");
  const double beta = GetScalar(prhs[6], "beta");
  const int min_samples_leaf = GetInt(prhs[7], "min_samples_leaf");
  const int max_depth = GetInt(prhs[8], "max_depth");

  auto obj = std::make_unique<SamplerObj>();
  obj->tracker = std::make_unique<StochTree::ForestTracker>(
      dataset->value.GetCovariates(), feature_types, num_trees, num_obs);
  obj->prior = std::make_unique<StochTree::TreePrior>(alpha, beta, min_samples_leaf, max_depth);
  plhs[0] = Register(std::move(obj));
}

/*!
 * Initialize an "active" forest to a constant value and propagate that
 * initialization through the sampler's tracking data structures.
 * Mirrors ForestSamplerCpp::InitializeForestModel.
 */
void CmdSamplerInitializeForest(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 7, "sampler_initialize_forest");
  auto* sampler = Fetch<SamplerObj>(prhs[1], kTagSampler, "sampler");
  auto* dataset = Fetch<DatasetObj>(prhs[2], kTagDataset, "dataset");
  auto* residual = Fetch<ResidualObj>(prhs[3], kTagResidual, "residual");
  auto* forest = Fetch<ForestObj>(prhs[4], kTagForest, "forest");
  const int leaf_model_int = GetInt(prhs[5], "leaf_model");
  std::vector<double> init_values = GetDoubleVector(prhs[6], "initial_values");
  if (init_values.empty()) Fail("stochtree:value", "initial_values must be non-empty.");

  const StochTree::ModelType model_type = ToModelType(leaf_model_int);
  StochTree::TreeEnsemble* forest_ptr = &forest->value;
  StochTree::ForestDataset* data_ptr = &dataset->value;
  StochTree::ColumnVector* resid_ptr = &residual->value;
  const auto num_trees = static_cast<double>(forest_ptr->NumTrees());

  if (model_type == StochTree::ModelType::kConstantLeafGaussian) {
    forest_ptr->SetLeafValue(init_values[0] / num_trees);
    StochTree::UpdateResidualEntireForest(*(sampler->tracker), *data_ptr, *resid_ptr, forest_ptr,
                                          false, std::minus<double>());
    sampler->tracker->UpdatePredictions(forest_ptr, *data_ptr);
  } else if (model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian) {
    forest_ptr->SetLeafValue(init_values[0] / num_trees);
    StochTree::UpdateResidualEntireForest(*(sampler->tracker), *data_ptr, *resid_ptr, forest_ptr,
                                          true, std::minus<double>());
    sampler->tracker->UpdatePredictions(forest_ptr, *data_ptr);
  } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
    std::vector<double> scaled(init_values.size());
    for (size_t i = 0; i < init_values.size(); i++) scaled[i] = init_values[i] / num_trees;
    forest_ptr->SetLeafVector(scaled);
    StochTree::UpdateResidualEntireForest(*(sampler->tracker), *data_ptr, *resid_ptr, forest_ptr,
                                          true, std::minus<double>());
    sampler->tracker->UpdatePredictions(forest_ptr, *data_ptr);
  } else if (model_type == StochTree::ModelType::kLogLinearVariance) {
    forest_ptr->SetLeafValue(std::log(init_values[0]) / num_trees);
    sampler->tracker->UpdatePredictions(forest_ptr, *data_ptr);
    const auto n = data_ptr->NumObservations();
    std::vector<double> initial_preds(static_cast<size_t>(n), init_values[0]);
    data_ptr->AddVarianceWeights(initial_preds.data(), n);
  } else {  // kCloglogOrdinal
    forest_ptr->SetLeafValue(init_values[0] / num_trees);
    StochTree::UpdateResidualEntireForest(*(sampler->tracker), *data_ptr, *resid_ptr, forest_ptr,
                                          false, std::minus<double>());
    sampler->tracker->UpdatePredictions(forest_ptr, *data_ptr);
  }
}

void CmdSamplerReconstitute(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 6, "sampler_reconstitute");
  auto* sampler = Fetch<SamplerObj>(prhs[1], kTagSampler, "sampler");
  auto* forest = Fetch<ForestObj>(prhs[2], kTagForest, "forest");
  auto* dataset = Fetch<DatasetObj>(prhs[3], kTagDataset, "dataset");
  auto* residual = Fetch<ResidualObj>(prhs[4], kTagResidual, "residual");
  const bool is_mean_model = GetBool(prhs[5], "is_mean_model");
  sampler->tracker->ReconstituteFromForest(forest->value, dataset->value, residual->value,
                                           is_mean_model);
}

/*!
 * sampler_sample_one_iteration(sampler, container, forest, dataset, residual, rng,
 *                              feature_types, sweep_update_indices, cutpoint_grid_size,
 *                              leaf_model_scale, variable_weights, a_forest, b_forest,
 *                              global_variance, leaf_model, num_features_subsample,
 *                              keep_forest, gfr, num_threads)
 *
 * Argument order deliberately mirrors ForestSamplerCpp::SampleOneIteration so the
 * three language wrappers stay easy to diff against each other.
 */
void CmdSamplerSampleOneIteration(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 20, "sampler_sample_one_iteration");
  auto* sampler = Fetch<SamplerObj>(prhs[1], kTagSampler, "sampler");
  auto* container = Fetch<ContainerObj>(prhs[2], kTagContainer, "container");
  auto* forest = Fetch<ForestObj>(prhs[3], kTagForest, "forest");
  auto* dataset = Fetch<DatasetObj>(prhs[4], kTagDataset, "dataset");
  auto* residual = Fetch<ResidualObj>(prhs[5], kTagResidual, "residual");
  auto* rng = Fetch<RngObj>(prhs[6], kTagRng, "rng");

  std::vector<StochTree::FeatureType> feature_types =
      ToFeatureTypes(GetIntVector(prhs[7], "feature_types"));
  std::vector<int> sweep_update_indices = GetIntVector(prhs[8], "sweep_update_indices");
  const int cutpoint_grid_size = GetInt(prhs[9], "cutpoint_grid_size");

  if (!mxIsDouble(prhs[10]) || mxIsComplex(prhs[10])) {
    Fail("stochtree:type", "leaf_model_scale must be a real double matrix.");
  }
  const auto scale_rows = static_cast<int>(mxGetM(prhs[10]));
  const auto scale_cols = static_cast<int>(mxGetN(prhs[10]));
  const double* scale_ptr = RealPtr(prhs[10]);

  std::vector<double> variable_weights = GetDoubleVector(prhs[11], "variable_weights");
  const double a_forest = GetScalar(prhs[12], "a_forest");
  const double b_forest = GetScalar(prhs[13], "b_forest");
  const double global_variance = GetScalar(prhs[14], "global_variance");
  const int leaf_model_int = GetInt(prhs[15], "leaf_model");
  const int num_features_subsample = GetInt(prhs[16], "num_features_subsample");
  const bool keep_forest = GetBool(prhs[17], "keep_forest");
  const bool gfr = GetBool(prhs[18], "gfr");
  const int num_threads = GetInt(prhs[19], "num_threads");

  const StochTree::ModelType model_type = ToModelType(leaf_model_int);

  double leaf_scale = 0.0;
  Eigen::MatrixXd leaf_scale_matrix;
  if (model_type == StochTree::ModelType::kConstantLeafGaussian ||
      model_type == StochTree::ModelType::kUnivariateRegressionLeafGaussian) {
    if (scale_rows < 1 || scale_cols < 1) {
      Fail("stochtree:value", "leaf_model_scale must have at least one element.");
    }
    leaf_scale = scale_ptr[0];
  } else if (model_type == StochTree::ModelType::kMultivariateRegressionLeafGaussian) {
    leaf_scale_matrix.resize(scale_rows, scale_cols);
    for (int j = 0; j < scale_cols; j++) {
      for (int i = 0; i < scale_rows; i++) {
        leaf_scale_matrix(i, j) = scale_ptr[j * scale_rows + i];  // column-major in
      }
    }
  }

  StochTree::LeafModelVariant leaf_model =
      StochTree::leafModelFactory(model_type, leaf_scale, leaf_scale_matrix, a_forest, b_forest);

  StochTree::ForestContainer* container_ptr = &container->value;
  StochTree::TreeEnsemble* forest_ptr = &forest->value;
  StochTree::ForestDataset* data_ptr = &dataset->value;
  StochTree::ColumnVector* resid_ptr = &residual->value;
  StochTree::ForestTracker* tracker_ptr = sampler->tracker.get();
  StochTree::TreePrior* prior_ptr = sampler->prior.get();
  std::mt19937* rng_ptr = &rng->value;
  int num_basis = data_ptr->NumBasis();  // bound by reference below, so not const
  const bool pre_initialized = true;

  if (gfr) {
    switch (model_type) {
      case StochTree::ModelType::kConstantLeafGaussian:
        StochTree::GFRSampleOneIter<StochTree::GaussianConstantLeafModel,
                                    StochTree::GaussianConstantSuffStat>(
            *forest_ptr, *tracker_ptr, *container_ptr,
            std::get<StochTree::GaussianConstantLeafModel>(leaf_model), *data_ptr, *resid_ptr,
            *prior_ptr, *rng_ptr, variable_weights, sweep_update_indices, global_variance,
            feature_types, cutpoint_grid_size, keep_forest, pre_initialized, true,
            num_features_subsample, num_threads);
        break;
      case StochTree::ModelType::kUnivariateRegressionLeafGaussian:
        StochTree::GFRSampleOneIter<StochTree::GaussianUnivariateRegressionLeafModel,
                                    StochTree::GaussianUnivariateRegressionSuffStat>(
            *forest_ptr, *tracker_ptr, *container_ptr,
            std::get<StochTree::GaussianUnivariateRegressionLeafModel>(leaf_model), *data_ptr,
            *resid_ptr, *prior_ptr, *rng_ptr, variable_weights, sweep_update_indices,
            global_variance, feature_types, cutpoint_grid_size, keep_forest, pre_initialized, true,
            num_features_subsample, num_threads);
        break;
      case StochTree::ModelType::kMultivariateRegressionLeafGaussian:
        StochTree::GFRSampleOneIter<StochTree::GaussianMultivariateRegressionLeafModel,
                                    StochTree::GaussianMultivariateRegressionSuffStat, int>(
            *forest_ptr, *tracker_ptr, *container_ptr,
            std::get<StochTree::GaussianMultivariateRegressionLeafModel>(leaf_model), *data_ptr,
            *resid_ptr, *prior_ptr, *rng_ptr, variable_weights, sweep_update_indices,
            global_variance, feature_types, cutpoint_grid_size, keep_forest, pre_initialized, true,
            num_features_subsample, num_threads, num_basis);
        break;
      case StochTree::ModelType::kLogLinearVariance:
        StochTree::GFRSampleOneIter<StochTree::LogLinearVarianceLeafModel,
                                    StochTree::LogLinearVarianceSuffStat>(
            *forest_ptr, *tracker_ptr, *container_ptr,
            std::get<StochTree::LogLinearVarianceLeafModel>(leaf_model), *data_ptr, *resid_ptr,
            *prior_ptr, *rng_ptr, variable_weights, sweep_update_indices, global_variance,
            feature_types, cutpoint_grid_size, keep_forest, pre_initialized, false,
            num_features_subsample, num_threads);
        break;
      case StochTree::ModelType::kCloglogOrdinal:
        StochTree::GFRSampleOneIter<StochTree::CloglogOrdinalLeafModel,
                                    StochTree::CloglogOrdinalSuffStat>(
            *forest_ptr, *tracker_ptr, *container_ptr,
            std::get<StochTree::CloglogOrdinalLeafModel>(leaf_model), *data_ptr, *resid_ptr,
            *prior_ptr, *rng_ptr, variable_weights, sweep_update_indices, global_variance,
            feature_types, cutpoint_grid_size, keep_forest, pre_initialized, false,
            num_features_subsample, num_threads);
        break;
    }
  } else {
    switch (model_type) {
      case StochTree::ModelType::kConstantLeafGaussian:
        StochTree::MCMCSampleOneIter<StochTree::GaussianConstantLeafModel,
                                     StochTree::GaussianConstantSuffStat>(
            *forest_ptr, *tracker_ptr, *container_ptr,
            std::get<StochTree::GaussianConstantLeafModel>(leaf_model), *data_ptr, *resid_ptr,
            *prior_ptr, *rng_ptr, variable_weights, sweep_update_indices, global_variance,
            keep_forest, pre_initialized, true, num_threads);
        break;
      case StochTree::ModelType::kUnivariateRegressionLeafGaussian:
        StochTree::MCMCSampleOneIter<StochTree::GaussianUnivariateRegressionLeafModel,
                                     StochTree::GaussianUnivariateRegressionSuffStat>(
            *forest_ptr, *tracker_ptr, *container_ptr,
            std::get<StochTree::GaussianUnivariateRegressionLeafModel>(leaf_model), *data_ptr,
            *resid_ptr, *prior_ptr, *rng_ptr, variable_weights, sweep_update_indices,
            global_variance, keep_forest, pre_initialized, true, num_threads);
        break;
      case StochTree::ModelType::kMultivariateRegressionLeafGaussian:
        StochTree::MCMCSampleOneIter<StochTree::GaussianMultivariateRegressionLeafModel,
                                     StochTree::GaussianMultivariateRegressionSuffStat, int>(
            *forest_ptr, *tracker_ptr, *container_ptr,
            std::get<StochTree::GaussianMultivariateRegressionLeafModel>(leaf_model), *data_ptr,
            *resid_ptr, *prior_ptr, *rng_ptr, variable_weights, sweep_update_indices,
            global_variance, keep_forest, pre_initialized, true, num_threads, num_basis);
        break;
      case StochTree::ModelType::kLogLinearVariance:
        StochTree::MCMCSampleOneIter<StochTree::LogLinearVarianceLeafModel,
                                     StochTree::LogLinearVarianceSuffStat>(
            *forest_ptr, *tracker_ptr, *container_ptr,
            std::get<StochTree::LogLinearVarianceLeafModel>(leaf_model), *data_ptr, *resid_ptr,
            *prior_ptr, *rng_ptr, variable_weights, sweep_update_indices, global_variance,
            keep_forest, pre_initialized, false, num_threads);
        break;
      case StochTree::ModelType::kCloglogOrdinal:
        StochTree::MCMCSampleOneIter<StochTree::CloglogOrdinalLeafModel,
                                     StochTree::CloglogOrdinalSuffStat>(
            *forest_ptr, *tracker_ptr, *container_ptr,
            std::get<StochTree::CloglogOrdinalLeafModel>(leaf_model), *data_ptr, *resid_ptr,
            *prior_ptr, *rng_ptr, variable_weights, sweep_update_indices, global_variance,
            keep_forest, pre_initialized, false, num_threads);
        break;
    }
  }
}

void CmdSamplerCachedPredictions(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "sampler_cached_predictions");
  auto* sampler = Fetch<SamplerObj>(prhs[1], kTagSampler, "sampler");
  const auto n = static_cast<mwSize>(sampler->tracker->GetNumObservations());
  plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
  double* out = RealPtrMutable(plhs[0]);
  for (mwSize i = 0; i < n; i++) {
    out[i] = sampler->tracker->GetSamplePrediction(static_cast<data_size_t>(i));
  }
}

void CmdSamplerPropagateBasis(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 5, "sampler_propagate_basis_update");
  auto* sampler = Fetch<SamplerObj>(prhs[1], kTagSampler, "sampler");
  auto* dataset = Fetch<DatasetObj>(prhs[2], kTagDataset, "dataset");
  auto* residual = Fetch<ResidualObj>(prhs[3], kTagResidual, "residual");
  auto* forest = Fetch<ForestObj>(prhs[4], kTagForest, "forest");
  StochTree::UpdateResidualNewBasis(*(sampler->tracker), dataset->value, residual->value,
                                    &forest->value);
}

void CmdSamplerPropagateResidual(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 3, "sampler_propagate_residual_update");
  auto* sampler = Fetch<SamplerObj>(prhs[1], kTagSampler, "sampler");
  auto* residual = Fetch<ResidualObj>(prhs[2], kTagResidual, "residual");
  StochTree::UpdateResidualNewOutcome(*(sampler->tracker), residual->value);
}

void CmdSamplerAdjustResidual(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 7, "sampler_adjust_residual");
  auto* sampler = Fetch<SamplerObj>(prhs[1], kTagSampler, "sampler");
  auto* dataset = Fetch<DatasetObj>(prhs[2], kTagDataset, "dataset");
  auto* residual = Fetch<ResidualObj>(prhs[3], kTagResidual, "residual");
  auto* forest = Fetch<ForestObj>(prhs[4], kTagForest, "forest");
  const bool requires_basis = GetBool(prhs[5], "requires_basis");
  const bool add = GetBool(prhs[6], "add");
  std::function<double(double, double)> op;
  if (add) {
    op = std::plus<double>();
  } else {
    op = std::minus<double>();
  }
  StochTree::UpdateResidualEntireForest(*(sampler->tracker), dataset->value, residual->value,
                                        &forest->value, requires_basis, op);
}

void CmdSamplerSetPrior(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 6, "sampler_set_prior");
  auto* sampler = Fetch<SamplerObj>(prhs[1], kTagSampler, "sampler");
  sampler->prior->SetAlpha(GetScalar(prhs[2], "alpha"));
  sampler->prior->SetBeta(GetScalar(prhs[3], "beta"));
  sampler->prior->SetMinSamplesLeaf(GetInt(prhs[4], "min_samples_leaf"));
  sampler->prior->SetMaxDepth(GetInt(prhs[5], "max_depth"));
}

void CmdSamplerGetPrior(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "sampler_get_prior");
  auto* sampler = Fetch<SamplerObj>(prhs[1], kTagSampler, "sampler");
  const char* fields[] = {"alpha", "beta", "min_samples_leaf", "max_depth"};
  plhs[0] = mxCreateStructMatrix(1, 1, 4, fields);
  mxSetField(plhs[0], 0, "alpha", MakeScalar(sampler->prior->GetAlpha()));
  mxSetField(plhs[0], 0, "beta", MakeScalar(sampler->prior->GetBeta()));
  mxSetField(plhs[0], 0, "min_samples_leaf", MakeScalar(sampler->prior->GetMinSamplesLeaf()));
  mxSetField(plhs[0], 0, "max_depth", MakeScalar(sampler->prior->GetMaxDepth()));
}

/* ---- Variance models ---- */

void CmdGlobalVarCreate(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 1, "global_var_create");
  plhs[0] = Register(std::make_unique<GlobalVarObj>());
}

void CmdGlobalVarSample(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 6, "global_var_sample");
  auto* model = Fetch<GlobalVarObj>(prhs[1], kTagGlobalVar, "global variance model");
  auto* residual = Fetch<ResidualObj>(prhs[2], kTagResidual, "residual");
  auto* rng = Fetch<RngObj>(prhs[3], kTagRng, "rng");
  const double a = GetScalar(prhs[4], "a");
  const double b = GetScalar(prhs[5], "b");
  plhs[0] = MakeScalar(
      model->value.SampleVarianceParameter(residual->value.GetData(), a, b, rng->value));
}

void CmdLeafVarCreate(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 1, "leaf_var_create");
  plhs[0] = Register(std::make_unique<LeafVarObj>());
}

void CmdLeafVarSample(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 6, "leaf_var_sample");
  auto* model = Fetch<LeafVarObj>(prhs[1], kTagLeafVar, "leaf variance model");
  auto* forest = Fetch<ForestObj>(prhs[2], kTagForest, "forest");
  auto* rng = Fetch<RngObj>(prhs[3], kTagRng, "rng");
  const double a = GetScalar(prhs[4], "a");
  const double b = GetScalar(prhs[5], "b");
  plhs[0] = MakeScalar(model->value.SampleVarianceParameter(&forest->value, a, b, rng->value));
}

/* ---- Lifetime management ---- */

void CmdDelete(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "delete");
  const uint64_t id = ReadHandle(prhs[1]);
  Registry().erase(id);  // no-op if already gone, which keeps destructors idempotent
  UnlockIfEmpty();
}

void CmdCleanupAll(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 1, "cleanup_all");
  CleanupAll();
}

void CmdNumObjects(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 1, "num_objects");
  plhs[0] = MakeScalar(static_cast<double>(Registry().size()));
}

void CmdIsValid(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  RequireArgs(nrhs, 2, "is_valid");
  plhs[0] = MakeBool(Registry().count(ReadHandle(prhs[1])) > 0);
}

/* ------------------------------------------------------------------ */
/* Dispatch table                                                       */
/* ------------------------------------------------------------------ */

using CommandFn = void (*)(int, mxArray**, int, const mxArray**);

const std::unordered_map<std::string, CommandFn>& Commands() {
  static const std::unordered_map<std::string, CommandFn> table = {
      {"dataset_create", CmdDatasetCreate},
      {"dataset_add_covariates", CmdDatasetAddCovariates},
      {"dataset_add_basis", CmdDatasetAddBasis},
      {"dataset_update_basis", CmdDatasetUpdateBasis},
      {"dataset_add_weights", CmdDatasetAddWeights},
      {"dataset_update_weights", CmdDatasetUpdateWeights},
      {"dataset_num_rows", CmdDatasetNumRows},
      {"dataset_num_covariates", CmdDatasetNumCovariates},
      {"dataset_num_basis", CmdDatasetNumBasis},
      {"dataset_has_basis", CmdDatasetHasBasis},
      {"dataset_has_weights", CmdDatasetHasWeights},
      {"dataset_get_covariates", CmdDatasetGetCovariates},
      {"dataset_get_basis", CmdDatasetGetBasis},
      {"dataset_get_weights", CmdDatasetGetWeights},

      {"residual_create", CmdResidualCreate},
      {"residual_get", CmdResidualGet},
      {"residual_replace", CmdResidualReplace},
      {"residual_add", CmdResidualAdd},
      {"residual_subtract", CmdResidualSubtract},

      {"rng_create", CmdRngCreate},

      {"forest_create", CmdForestCreate},
      {"forest_num_trees", CmdForestNumTrees},
      {"forest_output_dimension", CmdForestOutputDimension},
      {"forest_set_root_value", CmdForestSetRootValue},
      {"forest_set_root_vector", CmdForestSetRootVector},
      {"forest_reset_root", CmdForestResetRoot},
      {"forest_reset_from_container", CmdForestResetFromContainer},
      {"forest_predict", CmdForestPredict},
      {"forest_predict_raw", CmdForestPredictRaw},
      {"forest_num_leaves", CmdForestNumLeaves},
      {"forest_sum_leaf_squared", CmdForestSumLeafSquared},

      {"container_create", CmdContainerCreate},
      {"container_num_samples", CmdContainerNumSamples},
      {"container_num_trees", CmdContainerNumTrees},
      {"container_output_dimension", CmdContainerOutputDimension},
      {"container_delete_sample", CmdContainerDeleteSample},
      {"container_predict", CmdContainerPredict},
      {"container_predict_raw", CmdContainerPredictRaw},
      {"container_predict_raw_single", CmdContainerPredictRawSingle},
      {"container_add_to_forest", CmdContainerAddToForest},
      {"container_multiply_forest", CmdContainerMultiplyForest},
      {"container_overall_split_counts", CmdContainerSplitCounts},
      {"container_forest_split_counts", CmdContainerForestSplitCounts},
      {"container_dump_json", CmdContainerDumpJson},
      {"container_load_json", CmdContainerLoadJson},
      {"container_save_json_file", CmdContainerSaveJsonFile},
      {"container_load_json_file", CmdContainerLoadJsonFile},

      {"sampler_create", CmdSamplerCreate},
      {"sampler_initialize_forest", CmdSamplerInitializeForest},
      {"sampler_reconstitute", CmdSamplerReconstitute},
      {"sampler_sample_one_iteration", CmdSamplerSampleOneIteration},
      {"sampler_cached_predictions", CmdSamplerCachedPredictions},
      {"sampler_propagate_basis_update", CmdSamplerPropagateBasis},
      {"sampler_propagate_residual_update", CmdSamplerPropagateResidual},
      {"sampler_adjust_residual", CmdSamplerAdjustResidual},
      {"sampler_set_prior", CmdSamplerSetPrior},
      {"sampler_get_prior", CmdSamplerGetPrior},

      {"global_var_create", CmdGlobalVarCreate},
      {"global_var_sample", CmdGlobalVarSample},
      {"leaf_var_create", CmdLeafVarCreate},
      {"leaf_var_sample", CmdLeafVarSample},

      {"delete", CmdDelete},
      {"cleanup_all", CmdCleanupAll},
      {"num_objects", CmdNumObjects},
      {"is_valid", CmdIsValid},
  };
  return table;
}

}  // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  static bool registered_exit = false;
  if (!registered_exit) {
    mexAtExit(AtExitHandler);
    registered_exit = true;
  }

  if (nrhs < 1) {
    mexErrMsgIdAndTxt("stochtree:nargin",
                      "mex_stochtree requires at least one argument (the command name).");
    return;
  }

  // Everything below runs inside a try block so that C++ destructors unwind
  // normally before MATLAB's error mechanism (which longjmps) takes over.
  std::string error_id;
  std::string error_msg;
  try {
    const std::string command = GetString(prhs[0], "command");
    const auto& table = Commands();
    auto it = table.find(command);
    if (it == table.end()) {
      Fail("stochtree:command", "Unknown command '" + command + "'.");
    }
    it->second(nlhs, plhs, nrhs, prhs);
    return;
  } catch (const MexFailure& e) {
    error_id = e.id;
    error_msg = e.msg;
  } catch (const std::exception& e) {
    error_id = "stochtree:cpp";
    error_msg = std::string("stochtree C++ error: ") + e.what();
  } catch (...) {
    error_id = "stochtree:cpp";
    error_msg = "Unknown error in the stochtree C++ core.";
  }
  mexErrMsgIdAndTxt(error_id.c_str(), "%s", error_msg.c_str());
}

/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuml/common/logger.hpp>

// #TODO: Replace with public header when ready
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/gemv.cuh>
#include <raft/linalg/unary_op.cuh>

#include <iostream>
#include <limits>
#include <raft/core/handle.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <string>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>
#include <type_traits>

#include "kernelcache.cuh"
#include "smo_sets.cuh"
#include "smoblocksolve.cuh"
#include "workingset.cuh"
#include "ws_util.cuh"
#include <raft/distance/distance_types.hpp>
#include <raft/distance/kernels.cuh>
#include <raft/linalg/gemv.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/sparse/linalg/norm.cuh>

#include "results.cuh"
#include <cassert>
#include <sstream>
#include <string>

#include <chrono>
#include <cstdlib>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

namespace ML {
namespace SVM {

/**
 * @brief Solve the quadratic optimization problem using two level decomposition
 * and Sequential Minimal Optimization (SMO).
 *
 * The general decomposition idea by Osuna is to choose q examples from all the
 * training examples, and solve the QP problem for this subset (discussed in
 * section 11.2 by Joachims [1]). SMO is the extreme case where we choose q=2.
 *
 * Here we follow [2] and [3] and use two level decomposition. First we set
 * q_1=1024, and solve the QP sub-problem for that (let's call it QP1). This is
 * the outer iteration, implemented in SmoSolver::Solve.
 *
 * To solve QP1, we use another decomposition, specifically the SMO (q_2 = 2),
 * which is implemented in SmoBlockSolve.
 *
 * References:
 * - [1] Joachims, T. Making large-scale support vector machine learning
 *      practical. In B. Scholkopf, C. Burges, & A. Smola (Eds.), Advances in
 *      kernel methods: Support vector machines. Cambridge, MA: MIT Press (1998)
 * - [2] J. Vanek et al. A GPU-Architecture Optimized Hierarchical Decomposition
 *      Algorithm for Support VectorMachine Training, IEEE Transactions on
 *      Parallel and Distributed Systems, vol 28, no 12, 3330, (2017)
 * - [3] Z. Wen et al. ThunderSVM: A Fast SVM Library on GPUs and CPUs, Journal
 *      of Machine Learning Research, 19, 1-5 (2018)
 */
template <typename math_t>
class SmoSolver {
 public:
  SmoSolver(const raft::handle_t& handle,
            SvmParameter param,
            raft::distance::kernels::KernelType kernel_type,
            raft::distance::kernels::GramMatrixBase<math_t>* kernel)
    : handle(handle),
      C(param.C),
      tol(param.tol),
      kernel(kernel),
      kernel_type(kernel_type),
      cache_size(param.cache_size),
      nochange_steps(param.nochange_steps),
      epsilon(param.epsilon),
      svmType(param.svmType),
      stream(handle.get_stream()),
      return_buff(2, stream),
      alpha(0, stream),
      C_vec(0, stream),
      delta_alpha(0, stream),
      f(0, stream),
      y_label(0, stream)
  {
    ML::Logger::get().setLevel(param.verbosity);
  }

  void GetNonzeroDeltaAlpha(const math_t* vec,
                            int n_ws,
                            const int* idx,
                            math_t* nz_vec,
                            int* n_nz,
                            int* nz_idx,
                            cudaStream_t stream)
  {
    thrust::device_ptr<math_t> vec_ptr(const_cast<math_t*>(vec));
    thrust::device_ptr<math_t> nz_vec_ptr(nz_vec);
    thrust::device_ptr<int> idx_ptr(const_cast<int*>(idx));
    thrust::device_ptr<int> nz_idx_ptr(nz_idx);
    auto nonzero                   = [] __device__(math_t a) { return a != 0; };
    thrust::device_ptr<int> nz_end = thrust::copy_if(
      thrust::cuda::par.on(stream), idx_ptr, idx_ptr + n_ws, vec_ptr, nz_idx_ptr, nonzero);
    *n_nz = nz_end - nz_idx_ptr;
    thrust::copy_if(thrust::cuda::par.on(stream), vec_ptr, vec_ptr + n_ws, nz_vec_ptr, nonzero);
  }

#define SMO_WS_SIZE 1024
  /**
   * @brief Solve the quadratic optimization problem.
   *
   * The output arrays (dual_coefs, support_matrix, idx) will be allocated on the
   * device, they should be unallocated on entry.
   *
   * @param [in] matrix training vectors in matrix format(MLCommon::Matrix::Matrix),
   * size [n_rows x * n_cols]
   * @param [in] n_rows number of rows (training vectors)
   * @param [in] n_cols number of columns (features)
   * @param [in] y labels (values +/-1), size [n_rows]
   * @param [in] sample_weight device array of sample weights (or nullptr if not
   *     applicable)
   * @param [out] dual_coefs size [n_support] on exit
   * @param [out] n_support number of support vectors
   * @param [out] support_matrix support vectors in matrix format, size [n_support, n_cols]
   * @param [out] idx the original training set indices of the support vectors, size [n_support]
   * @param [out] b scalar constant for the decision function
   * @param [in] max_outer_iter maximum number of outer iteration (default 100 * n_rows)
   * @param [in] max_inner_iter maximum number of inner iterations (default 10000)
   */
  template <typename MatrixViewType>
  void Solve(MatrixViewType matrix,
             int n_rows,
             int n_cols,
             math_t* y,
             const math_t* sample_weight,
             math_t** dual_coefs,
             int* n_support,
             SupportStorage<math_t>* support_matrix,
             int** idx,
             math_t* b,
             int max_outer_iter = -1,
             int max_inner_iter = 10000)
  {
    // Prepare data structures for SMO
    WorkingSet<math_t> ws(handle, stream, n_rows, SMO_WS_SIZE, svmType);
    n_ws = ws.GetSize();
    Initialize(&y, sample_weight, n_rows, n_cols);
    KernelCache<math_t, MatrixViewType> cache(
      handle, matrix, n_rows, n_cols, n_ws, kernel, kernel_type, cache_size, svmType);

    // Init counters
    max_outer_iter        = GetDefaultMaxIter(n_train, max_outer_iter);
    n_iter                = 0;
    int n_inner_iter      = 0;
    diff_prev             = 0;
    n_small_diff          = 0;
    n_increased_diff      = 0;
    report_increased_diff = true;
    bool keep_going       = true;

    rmm::device_uvector<math_t> nz_da(n_ws, stream);
    rmm::device_uvector<int> nz_da_idx(n_ws, stream);

    while (n_iter < max_outer_iter && keep_going) {
      RAFT_CUDA_TRY(cudaMemsetAsync(delta_alpha.data(), 0, n_ws * sizeof(math_t), stream));
      raft::common::nvtx::push_range("SmoSolver::ws_select");
      ws.Select(f.data(), alpha.data(), y, C_vec.data());
      raft::common::nvtx::pop_range();
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      raft::common::nvtx::push_range("SmoSolver::Kernel");

      cache.InitWorkingSet(ws.GetIndices());

      math_t* cacheTile = cache.getSquareTileWithoutCaching();

      raft::common::nvtx::pop_range();
      raft::common::nvtx::push_range("SmoSolver::SmoBlockSolve");
      SmoBlockSolve<math_t, SMO_WS_SIZE><<<1, n_ws, 0, stream>>>(y,
                                                                 n_train,
                                                                 alpha.data(),
                                                                 n_ws,
                                                                 delta_alpha.data(),
                                                                 f.data(),
                                                                 cacheTile,
                                                                 cache.getKernelIndices(true),
                                                                 C_vec.data(),
                                                                 tol,
                                                                 return_buff.data(),
                                                                 max_inner_iter,
                                                                 svmType);

      RAFT_CUDA_TRY(cudaPeekAtLastError());

      raft::update_host(host_return_buff, return_buff.data(), 2, stream);
      raft::common::nvtx::pop_range();
      raft::common::nvtx::push_range("SmoSolver::UpdateF");
      raft::common::nvtx::push_range("SmoSolver::UpdateF::getNnzDaRows");
      int nnz_da;
      GetNonzeroDeltaAlpha(delta_alpha.data(),
                           n_ws,
                           cache.getKernelIndices(false),
                           nz_da.data(),
                           &nnz_da,
                           nz_da_idx.data(),
                           stream);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      // The following should be performed only for elements with nonzero delta_alpha
      if (nnz_da > 0) {
        auto batch_descriptor = cache.InitFullTileBatching(nz_da_idx.data(), nnz_da);

        while (cache.getNextBatchKernel(batch_descriptor)) {
          raft::common::nvtx::pop_range();
          raft::common::nvtx::push_range("SmoSolver::UpdateF::updateBatch");
          // do (partial) update
          UpdateF(f.data() + batch_descriptor.offset,
                  batch_descriptor.batch_size,
                  nz_da.data(),
                  nnz_da,
                  batch_descriptor.kernel_data);
          RAFT_CUDA_TRY(cudaPeekAtLastError());
        }
      }
      handle.sync_stream(stream);
      raft::common::nvtx::pop_range();
      raft::common::nvtx::pop_range();  // ("SmoSolver::UpdateF");

      math_t diff = host_return_buff[0];
      keep_going  = CheckStoppingCondition(diff);
      n_inner_iter += host_return_buff[1];
      n_iter++;
      if (n_iter % 500 == 0) { CUML_LOG_DEBUG("SMO iteration %d, diff %lf", n_iter, (double)diff); }
    }

    CUML_LOG_DEBUG(
      "SMO solver finished after %d outer iterations, total inner %d"
      " iterations, and diff %lf",
      n_iter,
      n_inner_iter,
      diff_prev);

    Results<math_t, MatrixViewType> res(handle, matrix, n_rows, n_cols, y, C_vec.data(), svmType);
    res.Get(alpha.data(), f.data(), dual_coefs, n_support, idx, support_matrix, b);

    ReleaseBuffers();
  }

  /**
   * @brief Update the f vector after a block solve step.
   *
   * \f[ f_i = f_i + \sum_{k\in WS} K_{i,k} * \Delta \alpha_k, \f]
   * where i = [0..n_train-1], WS is the set of workspace indices,
   * and \f$K_{i,k}\f$ is the kernel function evaluated for training vector x_i and workspace vector
   * x_k.
   *
   * @param f size [n_train]
   * @param n_rows
   * @param delta_alpha size [n_ws]
   * @param n_ws
   * @param cacheTile kernel function evaluated for the following set K[X,x_ws],
   *   size [n_rows, n_ws]
   */
  void UpdateF(math_t* f, int n_rows, const math_t* delta_alpha, int n_ws, const math_t* cacheTile)
  {
    // multipliers used in the equation : f = 1*cachtile * delta_alpha + 1*f
    math_t one = 1;
    // #TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(handle.get_cublas_handle(),
                                                     CUBLAS_OP_N,
                                                     n_rows,
                                                     n_ws,
                                                     &one,
                                                     cacheTile,
                                                     n_rows,
                                                     delta_alpha,
                                                     1,
                                                     &one,
                                                     f,
                                                     1,
                                                     stream));
    if (svmType == EPSILON_SVR) {
      // SVR has doubled the number of training vectors and we need to update
      // alpha for both batches individually
      // #TODO: Call from public API when ready
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(handle.get_cublas_handle(),
                                                       CUBLAS_OP_N,
                                                       n_rows,
                                                       n_ws,
                                                       &one,
                                                       cacheTile,
                                                       n_rows,
                                                       delta_alpha,
                                                       1,
                                                       &one,
                                                       f + n_rows,
                                                       1,
                                                       stream));
    }
  }

  /** @brief Initialize the problem to solve.
   *
   * Both SVC and SVR are solved as a classification problem.
   * The optimization target (W) does not appear directly in the SMO
   * formulation, only its derivative through f (optimality indicator vector):
   * \f[ f_i = y_i \frac{\partial W }{\partial \alpha_i}. \f]
   *
   * The f_i values are initialized here, and updated at every solver iteration
   * when alpha changes. The update step is the same for SVC and SVR, only the
   * init step differs.
   *
   * Additionally, we zero init the dual coefficients (alpha), and initialize
   * class labels for SVR.
   *
   * @param[inout] y on entry class labels or target values,
   *    on exit device pointer to class labels
   * @param[in] sample_weight sample weights (can be nullptr, otherwise device
   *    array of size [n_rows])
   * @param[in] n_rows
   * @param[in] n_cols
   */
  void Initialize(math_t** y, const math_t* sample_weight, int n_rows, int n_cols)
  {
    this->n_rows = n_rows;
    this->n_cols = n_cols;
    n_train      = (svmType == EPSILON_SVR) ? n_rows * 2 : n_rows;
    ResizeBuffers(n_train, n_cols);
    // Zero init alpha
    RAFT_CUDA_TRY(cudaMemsetAsync(alpha.data(), 0, n_train * sizeof(math_t), stream));
    InitPenalty(C_vec.data(), sample_weight, n_rows);
    // Init f (and also class labels for SVR)
    switch (svmType) {
      case C_SVC: SvcInit(*y); break;
      case EPSILON_SVR:
        SvrInit(*y, n_rows, y_label.data(), f.data());
        // We return the pointer to the class labels (the target values are
        // not needed anymore, they are incorporated in f).
        *y = y_label.data();
        break;
      default: THROW("SMO initialization not implemented SvmType=%d", svmType);
    }
  }

  void InitPenalty(math_t* C_vec, const math_t* sample_weight, int n_rows)
  {
    if (sample_weight == nullptr) {
      thrust::device_ptr<math_t> c_ptr(C_vec);
      thrust::fill(thrust::cuda::par.on(stream), c_ptr, c_ptr + n_train, C);
    } else {
      math_t C = this->C;
      raft::linalg::unaryOp(
        C_vec, sample_weight, n_rows, [C] __device__(math_t w) { return C * w; }, stream);
      if (n_train > n_rows) {
        // Set the same penalty parameter for the duplicate set of vectors
        raft::linalg::unaryOp(
          C_vec + n_rows,
          sample_weight,
          n_rows,
          [C] __device__(math_t w) { return C * w; },
          stream);
      }
    }
  }
  /** @brief Initialize Support Vector Classification
   *
   * We would like to maximize the following quantity
   * \f[ W(\mathbf{\alpha}) = -\mathbf{\alpha}^T \mathbf{1}
   *   + \frac{1}{2} \mathbf{\alpha}^T Q \mathbf{\alpha}, \f]
   *
   * We initialize f as:
   * \f[ f_i = y_i \frac{\partial W(\mathbf{\alpha})}{\partial \alpha_i} =
   *          -y_i +   y_j \alpha_j K(\mathbf{x}_i, \mathbf{x}_j) \f]
   *
   * @param [in] y device pointer of class labels size [n_rows]
   */
  void SvcInit(const math_t* y)
  {
    raft::linalg::unaryOp(
      f.data(), y, n_rows, [] __device__(math_t y) { return -y; }, stream);
  }

  /**
   * @brief Initializes the solver for epsilon-SVR.
   *
   * For regression we are optimizing the following quantity
   * \f[
   * W(\alpha^+, \alpha^-) =
   * \epsilon \sum_{i=1}^l (\alpha_i^+ + \alpha_i^-)
   * - \sum_{i=1}^l yc_i (\alpha_i^+ - \alpha_i^-)
   * + \frac{1}{2} \sum_{i,j=1}^l
   *   (\alpha_i^+ - \alpha_i^-)(\alpha_j^+ - \alpha_j^-) K(\bm{x}_i, \bm{x}_j)
   * \f]
   *
   * Then \f$ f_i = y_i \frac{\partial W(\alpha}{\partial \alpha_i} \f$
   *      \f$     = yc_i*epsilon - yr_i \f$
   *
   * Additionally we set class labels for the training vectors.
   *
   * References:
   * [1] B. Schölkopf et. al (1998): New support vector algorithms,
   *     NeuroCOLT2 Technical Report Series, NC2-TR-1998-031, Section 6
   * [2] A.J. Smola, B. Schölkopf (2004): A tutorial on support vector
   *     regression, Statistics and Computing 14, 199–222
   * [3] Orchel M. (2011) Support Vector Regression as a Classification Problem
   *     with a Priori Knowledge in the Form of Detractors,
   *     Man-Machine Interactions 2. Advances in Intelligent and Soft Computing,
   *     vol 103
   *
   * @param [in] yr device pointer with values for regression, size [n_rows]
   * @param [in] n_rows
   * @param [out] yc device pointer to classes associated to the dual
   *     coefficients, size [n_rows*2]
   * @param [out] f device pointer f size [n_rows*2]
   */
  void SvrInit(const math_t* yr, int n_rows, math_t* yc, math_t* f)
  {
    // Init class labels to [1, 1, 1, ..., -1, -1, -1, ...]
    thrust::device_ptr<math_t> yc_ptr(yc);
    thrust::constant_iterator<math_t> one(1);
    thrust::copy(thrust::cuda::par.on(stream), one, one + n_rows, yc_ptr);
    thrust::constant_iterator<math_t> minus_one(-1);
    thrust::copy(thrust::cuda::par.on(stream), minus_one, minus_one + n_rows, yc_ptr + n_rows);

    // f_i = epsilon - y_i, for i \in [0..n_rows-1]
    math_t epsilon = this->epsilon;
    raft::linalg::unaryOp(
      f, yr, n_rows, [epsilon] __device__(math_t y) { return epsilon - y; }, stream);

    // f_i = -epsilon - y_i, for i \in [n_rows..2*n_rows-1]
    raft::linalg::unaryOp(
      f + n_rows, yr, n_rows, [epsilon] __device__(math_t y) { return -epsilon - y; }, stream);
  }

 private:
  const raft::handle_t& handle;
  cudaStream_t stream;

  int n_rows  = 0;  //!< training data number of rows
  int n_cols  = 0;  //!< training data number of columns
  int n_ws    = 0;  //!< size of the working set
  int n_train = 0;  //!< number of training vectors (including duplicates for SVR)

  // Buffers for the domain [n_train]
  rmm::device_uvector<math_t> alpha;    //!< dual coordinates
  rmm::device_uvector<math_t> f;        //!< optimality indicator vector
  rmm::device_uvector<math_t> y_label;  //!< extra label for regression

  rmm::device_uvector<math_t> C_vec;    //!< penalty parameter vector

  // Buffers for the working set [n_ws]
  //! change in alpha parameter during a blocksolve step
  rmm::device_uvector<math_t> delta_alpha;

  // Buffers to return some parameters from the kernel (iteration number, and
  // convergence information)
  rmm::device_uvector<math_t> return_buff;
  math_t host_return_buff[2];

  math_t C;
  math_t tol;      //!< tolerance for stopping condition
  math_t epsilon;  //!< epsilon parameter for epsiolon-SVR

  raft::distance::kernels::GramMatrixBase<math_t>* kernel;
  raft::distance::kernels::KernelType kernel_type;
  float cache_size;  //!< size of kernel cache in MiB

  SvmType svmType;   ///!< Type of the SVM problem to solve

  // Variables to track convergence of training
  math_t diff_prev;
  int n_small_diff;
  int nochange_steps;
  int n_increased_diff;
  int n_iter;
  bool report_increased_diff;

  bool CheckStoppingCondition(math_t diff)
  {
    if (diff > diff_prev * 1.5 && n_iter > 0) {
      // Ideally, diff should decrease monotonically. In practice we can have
      // small fluctuations (10% increase is not uncommon). Here we consider a
      // 50% increase in the diff value large enough to indicate a problem.
      // The 50% value is an educated guess that triggers the convergence debug
      // message for problematic use cases while avoids false alarms in many
      // other cases.
      n_increased_diff++;
    }
    if (report_increased_diff && n_iter > 100 && n_increased_diff > n_iter * 0.1) {
      CUML_LOG_DEBUG(
        "Solver is not converging monotonically. This might be caused by "
        "insufficient normalization of the feature columns. In that case "
        "MinMaxScaler((0,1)) could help. Alternatively, for nonlinear kernels, "
        "you can try to increase the gamma parameter. To limit execution time, "
        "you can also adjust the number of iterations using the max_iter "
        "parameter.");
      report_increased_diff = false;
    }
    bool keep_going = true;
    if (abs(diff - diff_prev) < 0.001 * tol) {
      n_small_diff++;
    } else {
      diff_prev    = diff;
      n_small_diff = 0;
    }
    if (n_small_diff > nochange_steps) {
      CUML_LOG_ERROR(
        "SMO error: Stopping due to unchanged diff over %d"
        " consecutive steps",
        nochange_steps);
      keep_going = false;
    }
    if (diff < tol) keep_going = false;
    if (isnan(diff)) {
      std::string txt;
      if (std::is_same<float, math_t>::value) {
        txt +=
          " This might be caused by floating point overflow. In such case using"
          " fp64 could help. Alternatively, try gamma='scale' kernel"
          " parameter.";
      }
      THROW("SMO error: NaN found during fitting.%s", txt.c_str());
    }
    return keep_going;
  }

  /// Return the number of maximum iterations.
  int GetDefaultMaxIter(int n_train, int max_outer_iter)
  {
    if (max_outer_iter == -1) {
      max_outer_iter = n_train < std::numeric_limits<int>::max() / 100
                         ? n_train * 100
                         : std::numeric_limits<int>::max();
      max_outer_iter = max(100000, max_outer_iter);
    }
    // else we have user defined iteration count which we do not change
    return max_outer_iter;
  }

  void ResizeBuffers(int n_train, int n_cols)
  {
    // This needs to know n_train, therefore it can be only called during solve
    alpha.resize(n_train, stream);
    C_vec.resize(n_train, stream);
    f.resize(n_train, stream);
    delta_alpha.resize(n_ws, stream);
    if (svmType == EPSILON_SVR) y_label.resize(n_train, stream);
  }

  void ReleaseBuffers()
  {
    alpha.release();
    delta_alpha.release();
    f.release();
    y_label.release();
  }
};

};  // end namespace SVM
};  // end namespace ML

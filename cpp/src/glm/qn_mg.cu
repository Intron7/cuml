/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <raft/core/comms.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/core/handle.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/linear_model/qn.h> 
#include "qn/simple_mat/dense.hpp"
#include "qn/qn_util.cuh"
#include "qn/glm_logistic.cuh"
#include "qn/glm_regularizer.cuh"

#include "qn/glm_base_mg.cuh"

#include <cuda_runtime.h>

namespace ML {
namespace GLM {
namespace opg {

template<typename T, typename I>
void qnFit_impl(const raft::handle_t &handle, 
                const qn_params& pams,
                T* X,
                bool X_col_major,
                T *y,
                I N,
                I D,
                I C,
                T* w0,
                T* f,
                int* num_iters,
                I n_samples,
                int rank,
                int n_ranks) 
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  auto X_simple = SimpleDenseMat<T>(X, N, D, X_col_major? COL_MAJOR : ROW_MAJOR);

  auto y_simple = SimpleVec<T>(y, N);

  SimpleVec<T> coef_simple(w0, D + pams.fit_intercept);

  auto coef_size = (D + pams.fit_intercept) * (C == 2 ? 1 : C); 

  ML::GLM::detail::LBFGSParam<T> opt_param(pams);

  // prepare regularizer regularizer_obj
  ML::GLM::detail::LogisticLoss<T> loss_func(handle, D, pams.fit_intercept);
  T l1 = pams.penalty_l1;
  T l2 = pams.penalty_l2;
  if (pams.penalty_normalized) {
      l1 /= n_samples; // l1 /= 1/X.m
      l2 /= n_samples; // l2 /= 1/X.m
  }
  ML::GLM::detail::Tikhonov<T> reg(l2);
  ML::GLM::detail::RegularizedGLM<T, ML::GLM::detail::LogisticLoss<T>, decltype(reg)> regularizer_obj(&loss_func, &reg);


  // prepare GLMWithDataMG
  int n_targets = C == 2 ? 1 : C;
  rmm::device_uvector<T> tmp(n_targets * N, stream);
  SimpleDenseMat<T> Z(tmp.data(), n_targets, N);
  auto obj_function = GLMWithDataMG(handle, rank, n_ranks, n_samples, &regularizer_obj, X_simple, y_simple, Z);

  // prepare temporary variables fx, k, workspace 
  float fx = -1; 
  int k = -1;
  rmm::device_uvector<float> tmp_workspace(lbfgs_workspace_size(opt_param, coef_simple.len), stream);
  SimpleVec<float> workspace(tmp_workspace.data(), tmp_workspace.size());

  // call min_lbfgs
  min_lbfgs(opt_param, obj_function, coef_simple, fx, &k, workspace, stream, 5);

}

void qnFit(const raft::handle_t &handle, 
           const qn_params& pams,
           float* X,
           bool X_col_major,
           float *y,
           int N,
           int D,
           int C,
           float* w0,
           float* f,
           int* num_iters,
           int n_samples,
           int rank,
           int n_ranks)
{
  qnFit_impl<float, int>(
    handle,
    pams,
    X,
    X_col_major,
    y,
    N,
    D,
    C,
    w0,
    f,
    num_iters,
    n_samples,
    rank,
    n_ranks);
}

};  // namespace OPG
};  // namespace GLM
};  // namespace ML

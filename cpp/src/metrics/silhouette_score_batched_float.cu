
/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <cuml/metrics/metrics.hpp>
#include <raft/core/handle.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/stats/silhouette_score.cuh>

namespace ML {

namespace Metrics {

namespace Batched {

float silhouette_score(const raft::handle_t& handle,
                       float* X,
                       int n_rows,
                       int n_cols,
                       int* y,
                       int n_labels,
                       float* scores,
                       int chunk,
                       raft::distance::DistanceType metric)
{
  return raft::stats::silhouette_score_batched<float, int, int>(
    handle, X, n_rows, n_cols, y, n_labels, scores, chunk, metric);
}

}  // namespace Batched
}  // namespace Metrics
}  // namespace ML

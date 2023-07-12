#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from cuml.common.kernel_utils import cuda_kernel_factory
import cuml.internals
import math
from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")
cupyx = gpu_only_import("cupyx")


cov_kernel_str = r"""
({0} *cov_values, {0} *gram_matrix, {0} *mean_x, {0} *mean_y, int n_cols) {

    int rid = blockDim.x * blockIdx.x + threadIdx.x;
    int cid = blockDim.y * blockIdx.y + threadIdx.y;

    if(rid >= n_cols || cid >= n_cols) return;

    cov_values[rid * n_cols + cid] = \
        gram_matrix[rid * n_cols + cid] - mean_x[rid] * mean_y[cid];
}
"""

cov_kernel_str_sparse_xx = r"""
(const int *indptr, const int *index, {0} *data, int nrows, int ncols, {0} *out) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if(row >= nrows) return;
    int start_idx = indptr[row];
    int stop_idx = indptr[row+1];

    for(int idx = start_idx; idx < stop_idx; idx++){
        int index1 = index[idx];
        {0} data1 = data[idx];
        long long int outidx = \
            static_cast<long long int>(index1) * ncols + index1;
        atomicAdd(&out[outidx], data1 * data1 / 2);
        for(int idx2 = idx+1; idx2 < stop_idx; idx2++){
            int index2 = index[idx2];
            {0} data2 = data[idx2];
            long long int outidx2 = \
                static_cast<long long int>(index1) * ncols + index2;
            atomicAdd(&out[outidx2], data1 * data2);
        }
    }
}
"""

cov_kernel_str_sparse = r"""
(int *indptr_x, int *index_x, {0} *data_x, \
    int *indptr_y, int *index_y, {0} *data_y, \
    int nrows, int ncols, {0} *out) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if(row >= nrows) return;
    int start_idx_x = indptr_x[row];
    int stop_idx_x = indptr_x[row+1];

    int start_idx_y = indptr_y[row];
    int stop_idx_y = indptr_y[row+1];

    for(int idx = start_idx_x; idx < stop_idx_x; idx++){
        int index1 = index_x[idx];
        {0} data1 = data_x[idx];
        for(int idx2 = start_idx_y; idx2 < stop_idx_y; idx2++){
            int index2 = index_y[idx2];
            {0} data2 = data_y[idx2];
            long long int outidx2 = \
                static_cast<long long int>(index1) * ncols + index2;
            atomicAdd(&out[outidx2], data1 * data2);
        }
    }
}
"""


def _cov_kernel(dtype):
    return cuda_kernel_factory(cov_kernel_str, (dtype,), "cov_kernel")


def _cov_kernel_sparse_xx(dtype):
    return cuda_kernel_factory(
        cov_kernel_str_sparse_xx, (dtype,), "cov_kernel_sprase_xx"
    )


def _cov_kernel_sprase(dtype):
    return cuda_kernel_factory(
        cov_kernel_str_sparse, (dtype,), "cov_kernel_sprase"
    )


@cuml.internals.api_return_any()
def cov(x, y, mean_x=None, mean_y=None, return_gram=False, return_mean=False):
    """
    Computes a covariance between two matrices using
    the form Cov(X, Y) = E(XY) - E(X)E(Y)

    This function prevents the need to explicitly
    compute the outer product E(X)E(Y) by taking
    advantage of the symmetry of that matrix
    and computes per element in a kernel.

    When E(XY) is approximately equal to E(X)E(Y),
    this method is prone to catastrophic cancellation.
    In such cases, a rectangular solver should be
    preferred.

    Parameters
    ----------

    x : device-array or cupyx.scipy.sparse of size (m, n)
    y : device-array or cupyx.scipy.sparse of size (m, n)
    mean_x : float (default = None)
        device-array of size (n, ) which is the mean
        of x across rows
    mean_x : float (default = None)
        device-array of size (n, ) which is the mean
        of x across rows
    return_gram : boolean (default = False)
        If True, gram matrix of the form (1 / n) * X.T.dot(Y)
        will be returned.
        When True, a copy will be created
        to store the results of the covariance.
        When False, the local gram matrix result
        will be overwritten
    return_mean: boolean (default = False)
        If True, the Maximum Likelihood Estimate used to
        calculate the mean of X and Y will be returned,
        of the form (1 / n) * mean(X) and (1 / n) * mean(Y)

    Returns
    -------

    result : cov(X, Y) when return_gram and return_mean are False
             cov(X, Y), gram(X, Y) when return_gram is True,
                return_mean is False
             cov(X, Y), mean(X), mean(Y) when return_gram is False,
                return_mean is True
             cov(X, Y), gram(X, Y), mean(X), mean(Y)
                when return_gram is True and return_mean is True
    """

    if x.dtype != y.dtype:
        raise ValueError(
            "X and Y must have same dtype (%s != %s)" % (x.dtype, y.dtype)
        )

    if x.shape != y.shape:
        raise ValueError(
            "X and Y must have same shape %s != %s" % (x.shape, y.shape)
        )

    # Fix for cupy issue #7699: addressing problems with sparse matrix multiplication (spGEMM)
    if cupyx.scipy.sparse.issparse(x) and cupyx.scipy.sparse.issparse(y):
        return _cov_sparse(
            x,
            y,
            mean_x,
            mean_y,
            return_gram=return_gram,
            return_mean=return_mean,
        )

    if mean_x is not None and mean_y is not None:
        if mean_x.dtype != mean_y.dtype:
            raise ValueError(
                "Mean of X and Mean of Y must have same dtype"
                "(%s != %s)" % (mean_x.dtype, mean_y.dtype)
            )

        if mean_x.shape != mean_y.shape:
            raise ValueError(
                "Mean of X and Mean of Y must have same shape"
                "%s != %s" % (mean_x.shape, mean_y.shape)
            )

    gram_matrix = x.T.dot(y) * (1 / x.shape[0])

    if cupyx.scipy.sparse.issparse(gram_matrix):
        gram_matrix = gram_matrix.todense()

    if mean_x is None:
        mean_x = x.sum(axis=0) * (1 / x.shape[0])

    if mean_y is None:
        mean_y = y.sum(axis=0) * (1 / y.shape[0])

    if return_gram:
        cov_result = cp.zeros(
            (gram_matrix.shape[0], gram_matrix.shape[1]),
            dtype=gram_matrix.dtype,
        )
    else:
        cov_result = gram_matrix

    compute_cov = _cov_kernel(x.dtype)

    block_size = (32, 32)
    grid_size = (
        math.ceil(gram_matrix.shape[0] / 32),
        math.ceil(gram_matrix.shape[1] / 32),
    )

    compute_cov(
        grid_size,
        block_size,
        (cov_result, gram_matrix, mean_x, mean_y, gram_matrix.shape[0]),
    )

    if not return_gram and not return_mean:
        return cov_result
    elif return_gram and not return_mean:
        return cov_result, gram_matrix
    elif not return_gram and return_mean:
        return cov_result, mean_x, mean_y
    elif return_gram and return_mean:
        return cov_result, gram_matrix, mean_x, mean_y


@cuml.internals.api_return_any()
def _cov_sparse(
    x, y, mean_x=None, mean_y=None, return_gram=False, return_mean=False
):
    """
    Computes a covariance between two matrices using
    the form Cov(X, Y) = E(XY) - E(X)E(Y)

    This function prevents the need to explicitly
    compute the outer product E(X)E(Y) by taking
    advantage of the symmetry of that matrix
    and computes per element in a kernel.

    When E(XY) is approximately equal to E(X)E(Y),
    this method is prone to catastrophic cancellation.
    In such cases, a rectangular solver should be
    preferred.

    This is a temporary fix for cupy issue #7699, where the
    operation `x.T.dot(x)` did not work for larger
    sparse matrices.

    Parameters
    ----------

    x : device-array or cupyx.scipy.sparse of size (m, n)
    y : device-array or cupyx.scipy.sparse of size (m, n)
    mean_x : float (default = None)
        device-array of size (n, ) which is the mean
        of x across rows
    mean_x : float (default = None)
        device-array of size (n, ) which is the mean
        of x across rows
    return_gram : boolean (default = False)
        If True, gram matrix of the form (1 / n) * X.T.dot(Y)
        will be returned.
        When True, a copy will be created
        to store the results of the covariance.
        When False, the local gram matrix result
        will be overwritten
    return_mean: boolean (default = False)
        If True, the Maximum Likelihood Estimate used to
        calculate the mean of X and Y will be returned,
        of the form (1 / n) * mean(X) and (1 / n) * mean(Y)

    Returns
    -------

    result : cov(X, Y) when return_gram and return_mean are False
             cov(X, Y), gram(X, Y) when return_gram is True,
                return_mean is False
             cov(X, Y), mean(X), mean(Y) when return_gram is False,
                return_mean is True
             cov(X, Y), gram(X, Y), mean(X), mean(Y)
                when return_gram is True and return_mean is True

    """
    if not cupyx.scipy.sparse.isspmatrix_csr(x):
        x = x.tocsr()
    if not cupyx.scipy.sparse.isspmatrix_csr(y):
        y = y.tocsr()
    gram_matrix = cp.zeros((x.shape[1], y.shape[1]), dtype=x.data.dtype)

    block = (8,)
    grid = (math.ceil(x.shape[0] / block[0]),)
    if x is y:
        compute_mean_cov = _cov_kernel_sparse_xx(x.data.dtype)
        compute_mean_cov(
            grid,
            block,
            (
                x.indptr,
                x.indices,
                x.data,
                x.shape[0],
                x.shape[1],
                gram_matrix,
            ),
        )
        gram_matrix = gram_matrix + gram_matrix.T

    else:
        compute_mean_cov = _cov_kernel_sprase(x.data.dtype)
        compute_mean_cov(
            grid,
            block,
            (
                x.indptr,
                x.indices,
                x.data,
                y.indptr,
                y.indices,
                y.data,
                x.shape[0],
                y.shape[1],
                gram_matrix,
            ),
        )

    gram_matrix *= 1 / x.shape[0]

    if mean_x is None:
        mean_x = x.sum(axis=0) * (1 / x.shape[0])

    if mean_y is None:
        mean_y = y.sum(axis=0) * (1 / y.shape[0])

    if return_gram:
        cov_result = cp.zeros(
            (gram_matrix.shape[0], gram_matrix.shape[1]),
            dtype=gram_matrix.dtype,
        )
    else:
        cov_result = gram_matrix

    compute_cov = _cov_kernel(x.dtype)

    block_size = (8, 8)
    grid_size = (
        math.ceil(gram_matrix.shape[0] / 8),
        math.ceil(gram_matrix.shape[1] / 8),
    )
    compute_cov(
        grid_size,
        block_size,
        (cov_result, gram_matrix, mean_x, mean_y, gram_matrix.shape[0]),
    )

    if not return_gram and not return_mean:
        return cov_result
    elif return_gram and not return_mean:
        return cov_result, gram_matrix
    elif not return_gram and return_mean:
        return cov_result, mean_x, mean_y
    elif return_gram and return_mean:
        return cov_result, gram_matrix, mean_x, mean_y

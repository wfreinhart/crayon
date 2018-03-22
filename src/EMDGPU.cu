//
// EMDGPU.cu
// CUDA implementation of Gary Doran's EMD code
//
// Copyright (c) 2018 Wesley Reinhart.
// This file is part of the crayon project, released under the Modified BSD License.

#include "EMDGPU.cuh"

namespace crayon
{
namespace gpu
{
__device__ double *vector_malloc(int n) {
    double *v;
    v = (double *) malloc(n*sizeof(double));
    return v;
}
__device__ double *vector_copy(double *v, int n) {
    double *copy;
    int i;
    copy = (double *) malloc(n*sizeof(double));
    for (i = 0; i < n; i++) {
        copy[i] = v[i];
    }
    return copy;
}
__device__ struct basic_variable *init_basic(int row, int col, int ncols, double flow) {
    struct basic_variable *var;
    var = (struct basic_variable*)malloc(sizeof(struct basic_variable));
    var->row = row;
    var->col = col;
    var->idx = row * ncols + col;
    var->flow = flow;
    var->adjacency = NULL;
    var->current_adj = NULL;
    var->back_ptr = NULL;
    var->color = WHITE;
    return var;
}
__device__ void insert_basic(struct basic_variable **basis, int size,
                             struct basic_variable *node) {
    struct adj_node *adj;
    int i;
    basis[size] = node;
    for (i = 0; i < size; i++) {
        if (basis[i]->row == node->row ||
            basis[i]->col == node->col) {
            adj = (struct adj_node*)malloc(sizeof(struct adj_node));
            adj->variable = node;
            adj->next = basis[i]->adjacency;
            basis[i]->adjacency = adj;

            adj = (struct adj_node*)malloc(sizeof(struct adj_node));
            adj->variable = basis[i];
            adj->next = node->adjacency;
            node->adjacency = adj;
        }
    }
}
__device__ void remove_basic(struct basic_variable **basis, int size,
                  struct basic_variable *node) {
    // Find node in list
    int i;
    for (i = 0; i < size; i++) {
        if (node == basis[i]) { break; }
    }
    assert (i < size);

    // Remove entry from adjacency lists of other nodes
    struct adj_node *a, *o, *last, *next;
    a = node->adjacency;
    while (a != NULL) {
        last = NULL;
        for (o = a->variable->adjacency; o != NULL;
             o = o->next) {
            if (o->variable == node) {
                if (last == NULL) {
                    a->variable->adjacency = o->next;
                } else {
                    last->next = o->next;
                }
                free(o);
                break;
            }
            last = o;
        }

        next = a->next;
        free(a);
        a = next;
    }
    free(basis[i]);

    basis[i] = basis[size];
}
/*
 * Initialize the basic variables with a feasible solution
 * using the "Northwest Corner Rule"
 */
__device__ struct basic_variable **initialize_flow(int n_x, double *weight_x,
                                       int n_y, double *weight_y,
                                       double *cost){
    struct basic_variable **basis;
    struct basic_variable *basic;
    double *remaining_x;
    double *remaining_y;
    int fx, fy, b, B;

    b = 0;
    B = n_x + n_y - 1;

    basis = (struct basic_variable**)malloc(
        (B+1)*sizeof(struct basic_variable*));

    remaining_x = vector_copy(weight_x, n_x);
    remaining_y = vector_copy(weight_y, n_y);
    fx = 0;
    fy = 0;
    while (1) {
        if (fx == (n_x - 1)) {
            for ( ; fy < n_y; fy++) {
                basic = init_basic(fx, fy, n_y, remaining_y[fy]);
                insert_basic(basis, b, basic);
                b++;
            }
            break;
        }
        if (fy == (n_y - 1)) {
            for ( ; fx < n_x; fx++) {
                basic = init_basic(fx, fy, n_y, remaining_x[fx]);
                insert_basic(basis, b, basic);
                b++;
            }
            break;
        }
        if (remaining_x[fx] <= remaining_y[fy]) {
            basic = init_basic(fx, fy, n_y, remaining_x[fx]);
            insert_basic(basis, b, basic);
            b++;
            remaining_y[fy] -= remaining_x[fx];
            fx++;
        } else {
            basic = init_basic(fx, fy, n_y, remaining_y[fy]);
            insert_basic(basis, b, basic);
            b++;
            remaining_x[fx] -= remaining_y[fy];
            fy++;
        }
    }

    free(remaining_x);
    free(remaining_y);
    return basis;
}
__device__ void reset_current_adj(struct basic_variable **basis, int size) {
    int i;
    for (i = 0; i < size; i++) {
        basis[i]->current_adj = basis[i]->adjacency;
        basis[i]->back_ptr = NULL;
        basis[i]->color = WHITE;
    }
}
__device__ void destruct_basis(struct basic_variable **basis, int size) {
    int i;
    struct adj_node *adj;
    struct adj_node *next_adj;
    for (i = 0; i < size; i++) {
        adj = basis[i]->adjacency;
        while (adj != NULL) {
            next_adj = adj->next;
            free(adj);
            adj = next_adj;
        }
        free(basis[i]);
    }
    free(basis);
}
namespace kernel
{
__global__ void add(float *z, float *x, float *y)
    {
    z[threadIdx.x] = x[threadIdx.x] + y[threadIdx.x];
    }
__global__ void set(float val, float *z)
    {
    z[threadIdx.x] = val;
    }
__global__ void pyemd(int n_x, double *weight_x,
    int n_y, double *weight_y,
    double *cost, double *flows,
    double *d)
    {
    struct basic_variable **basis;
    struct basic_variable *var, *root, *to_remove;
    struct adj_node *adj;
    double *dual_x, *dual_y;
    int *solved_x, *solved_y;
    int i, j, B, min_row, min_col;
    double min_slack, slack, min_flow, sign, distance;
    B = n_x + n_y - 1;

    basis = initialize_flow(n_x, weight_x, n_y, weight_y, cost);

    // Iterate until optimality conditions satisfied
    dual_x = vector_malloc(n_x);
    dual_y = vector_malloc(n_y);
    solved_x = (int *) malloc(n_x*sizeof(int));
    solved_y = (int *) malloc(n_y*sizeof(int));
    while (1) {
        for (i = 0; i < n_x; i++) { solved_x[i] = 0; }
        for (i = 0; i < n_y; i++) { solved_y[i] = 0; }
        reset_current_adj(basis, B);
        var = basis[0];
        dual_x[var->row] = 0.0;
        solved_x[var->row] = 1;
        while (1) {
            var->color = GRAY;
            if (solved_x[var->row]){
                dual_y[var->col] = (cost[var->idx] - dual_x[var->row]);
                solved_y[var->col] = 1;
            } else if (solved_y[var->col]) {
                dual_x[var->row] = (cost[var->idx] - dual_y[var->col]);
                solved_x[var->row] = 1;
            } else {
                assert(FALSE);
            }
            for (adj = var->current_adj; adj != NULL; adj = adj->next) {
                if (adj->variable->color == WHITE) { break; }
            }
            if (adj == NULL) {
                var->color = BLACK;
                var = var->back_ptr;
                if (var == NULL) {
                    break;
                }
            } else {
                var->current_adj = adj->next;
                adj->variable->back_ptr = var;
                var = adj->variable;
            }
        }
        // Check for optimality
        min_row = -1;
        min_col = -1;
        min_slack = 0.0;
        for (i = 0; i < n_x; i++) {
            for (j = 0; j < n_y; j++) {
                int k = i * n_y + j;
                slack = cost[k] - dual_x[i] - dual_y[j];
                if (min_row < 0 || slack < min_slack) {
                    min_row = i;
                    min_col = j;
                    min_slack = slack;
                }
            }
        }
        for (i = 0; i < B; i++) {
            // If the pivot variable is any of the
            // basis variables, then the optimal
            // solution has been found; set
            // min_slack = 0.0 explicitly to avoid
            // floating point issues in comparison.
            if (basis[i]->row == min_row &&
                basis[i]->col == min_col) {
                min_slack = 0.0;
                break;
            }
        }
        if (min_slack >= -EPSILON) { break; }
        // Introduce a new variable
        var = init_basic(min_row, min_col, n_y, 0.0);
        insert_basic(basis, B, var);
        root = var;
        reset_current_adj(basis, B + 1);
        while (1) {
            var->color = GRAY;
            for (adj = var->current_adj; adj != NULL; adj = adj->next) {
                if (var->back_ptr != NULL
                    && (var->back_ptr->row == adj->variable->row
                     || var->back_ptr->col == adj->variable->col)) {
                    continue;
                }
                if (adj->variable == root) {
                    // Found a cycle
                    break;
                }
                if (adj->variable->color == WHITE) { break; }
            }
            if (adj == NULL) {
                var->color = BLACK;
                var = var->back_ptr;
                if (var == NULL) {
                    // Couldn't find a cycle
                    assert(FALSE);
                }
            } else {
                if (adj->variable->color == GRAY) {
                    // We found a cycle
                    root->back_ptr = var;
                    break;
                } else {
                    var->current_adj = adj->next;
                    adj->variable->back_ptr = var;
                    var = adj->variable;
                }
            }
        }
        // Find the largest flow that can be subtracted
        sign = -1.0;
        min_flow = 0;
        to_remove = NULL;
        for (var = root->back_ptr; var != root; var = var->back_ptr) {
            if (sign < 0 && (to_remove == NULL || var->flow < min_flow)) {
                min_flow = var->flow;
                to_remove = var;
            }
            sign *= -1.0;
        }
        // Adjust flows
        sign = -1.0;
        root->flow = min_flow;
        for (var = root->back_ptr; var != root; var = var->back_ptr) {
            var->flow += (sign * min_flow);
            sign *= -1.0;
        }
        // Remove the basic variable that went to zero
        remove_basic(basis, B, to_remove);
    }
    distance = 0;
    for (i = 0; i < B; i++) {
        distance += (basis[i]->flow * cost[basis[i]->idx]);
    }
    if (flows != NULL) {
        // Initialize to zero
        for (int k = 0; k < n_x * n_y; k++) {
            flows[i] = 0;
        }
        for (i = 0; i < B; i++) {
            flows[basis[i]->idx] = basis[i]->flow;
        }
    }
    free(dual_x);
    free(dual_y);
    free(solved_x);
    free(solved_y);
    destruct_basis(basis, B);
    d[0] = distance;
    }
} // end namespace kernel
void add(int N)
    {

    // allocate host memory
    float *x = (float *)malloc(N * sizeof(float));
    float *y = (float *)malloc(N * sizeof(float));
    float *z = (float *)malloc(N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
        z[i] = -1.0f;
        }

    // allocate device memory
    float *x_gpu;
    float *y_gpu;
    float *z_gpu;
    cudaMalloc(&x_gpu, N*sizeof(float));
    cudaMalloc(&y_gpu, N*sizeof(float));
    cudaMalloc(&z_gpu, N*sizeof(float));

    // copy from host to device
    cudaMemcpy(x_gpu, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_gpu, y, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(z_gpu, -2.0f, N*sizeof(float));

    // run kernel on N elements on the gpu
    kernel::set<<< 1, N >>>(4.5f, z_gpu);

    // wait for gpu to finish
    cudaDeviceSynchronize();

    // copy back from gpu
    cudaMemcpy(z, z_gpu, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // check for errors (all values should be 4.5f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        {
        std::cout << z[i] << " ";
        maxError = fmax(maxError, fabs(z[i]-4.5f));
        }
    std::cout << std::endl;
    std::cout << "Max error: " << maxError << std::endl;

    // run kernel on N elements on the gpu
    cudaMemset(z_gpu, -2.0f, N*sizeof(float));
    kernel::add<<< 1, N >>>(z_gpu, x_gpu, y_gpu);

    // wait for gpu to finish
    cudaDeviceSynchronize();

    // copy back from gpu
    cudaMemcpy(z, z_gpu, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // check for errors (all values should be 3.0f)
    maxError = 0.0f;
    for (int i = 0; i < N; i++)
        {
        std::cout << z[i] << " ";
        maxError = fmax(maxError, fabs(z[i]-3.0f));
        }
    std::cout << std::endl;
    std::cout << "Max error: " << maxError << std::endl;

    // free host memory
    free(x);
    free(y);
    free(z);

    // free gpu memory
    cudaFree(x_gpu);
    cudaFree(y_gpu);
    cudaFree(z_gpu);

    // return cudaSuccess;
    return;
    }
double pyemd(int n_x, double *weight_x,
    int n_y, double *weight_y,
    double **cost, double **flows)
    {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    std::cout << "running gpu::pyemd, " << nDevices << "devices detected" << std::endl;
    // copy data to gpu
    double *d = (double *)malloc(sizeof(double));
    memset(d,-1,sizeof(double));
    double *d_gpu;
    cudaMalloc(&d_gpu,sizeof(double));
    double *weight_x_gpu;
    cudaMalloc(&weight_x_gpu,n_x*sizeof(double));
    double *weight_y_gpu;
    cudaMalloc(&weight_y_gpu,n_y*sizeof(double));
    double *flows_gpu;
    cudaMalloc(&flows_gpu,n_x*n_y*sizeof(double));
    // convert cost array to vector
    double *cost_host = (double *)malloc(n_x*n_y*sizeof(double));
    for( int i = 0; i < n_x; i++ )
        {
        for( int j = 0; j < n_y; j++ )
            {
            int k = i * n_y + j;
            cost_host[k] = cost[i][j];
            }
        }
    double *cost_gpu;
    cudaMalloc(&cost_gpu,n_x*n_y*sizeof(double));
    cudaMemcpy(cost_gpu, cost_host, n_x*n_y*sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    memset(cost_host,0,n_x*n_y*sizeof(double));
    cudaMemcpy(cost_host, cost_gpu, n_x*n_y*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout << "cost matrix read back from gpu:" << std::endl;
    for( int i = 0; i < n_x; i++ )
        {
        for( int j = 0; j < n_y; j++ )
            {
            int k = i * n_y + j;
            std::cout << cost_host[k] << " ";
            }
        std::cout << std::endl;
        }
    // set default values for debugging
    memset(d, -1.0, sizeof(double));
    cudaMemset(d_gpu, -2.0, sizeof(double));
    // call emd kernel
    kernel::pyemd<<< 1, 1 >>>(n_x, weight_x_gpu,
        n_y, weight_y_gpu,
        cost_gpu, flows_gpu,
        d_gpu
        );
    // wait for gpu to finish and copy back
    cudaDeviceSynchronize();
    cudaMemcpy(d, d_gpu, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return d[0];
}
} // end namespace gpu
} // end namespace crayon
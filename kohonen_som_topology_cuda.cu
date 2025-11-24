/**
 * \file
 * \brief [Kohonen self organizing map](https://en.wikipedia.org/wiki/Self-organizing_map) 
 *        (topological map) - CUDA Version
 *
 * This version uses CUDA to accelerate distance calculations on GPU
 * \author Adapted for CUDA acceleration
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

// Error checking macro for CUDA calls
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

/** to store info regarding 3D arrays */
struct kohonen_array_3d
{
    int dim1;
    int dim2;
    int dim3;
    double *data;
};

/** Function that returns the pointer to (x, y, z) location in linear 3D array */
double *kohonen_data_3d(const struct kohonen_array_3d *arr, int x, int y, int z)
{
    int offset = (x * arr->dim2 * arr->dim3) + (y * arr->dim3) + z;
    return arr->data + offset;
}

/**
 * CUDA Kernel: Compute Euclidean distances for all neurons
 * Each thread computes distance for one neuron (x, y)
 * 
 * @param d_W: Weights matrix on GPU (num_out * num_out * num_features)
 * @param d_X: Current sample on GPU (num_features)
 * @param d_D: Output distance matrix on GPU (num_out * num_out)
 * @param num_out: Size of SOM map (num_out x num_out)
 * @param num_features: Number of features
 */
__global__ void compute_distances_kernel(
    const double *d_W,
    const double *d_X,
    double *d_D,
    int num_out,
    int num_features)
{
    // Calculate thread indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (x >= num_out || y >= num_out)
        return;
    
    // Compute Euclidean distance
    double distance = 0.0;
    
    // Calculate offset for this neuron's weights
    int weight_offset = (x * num_out * num_features) + (y * num_features);
    
    // Sum of squared differences
    for (int k = 0; k < num_features; k++)
    {
        double diff = d_W[weight_offset + k] - d_X[k];
        distance += diff * diff;
    }
    
    // Store sqrt of distance
    d_D[x * num_out + y] = sqrt(distance);
}

/**
 * CUDA Kernel: Find minimum value and indices in distance matrix
 * Simplified version - uses shared memory reduction
 * Note: For small maps (30x30), CPU version is fast enough
 */
__global__ void find_min_kernel(
    const double *d_D,
    int num_out,
    double *d_min_val,
    int *d_min_x,
    int *d_min_y)
{
    // This is a placeholder - for small maps, CPU version is preferred
    // For larger maps, implement proper reduction here
    (void)d_D;
    (void)num_out;
    (void)d_min_val;
    (void)d_min_x;
    (void)d_min_y;
}

/**
 * Simplified minimum finder (CPU fallback)
 * For small maps (30x30 = 900 elements), CPU is fast enough
 * For larger maps, implement GPU reduction
 */
void find_min_simple(const double *d_D, int num_out, double *min_val, int *min_x, int *min_y)
{
    // Allocate temporary host memory
    double *h_D = (double *)malloc(num_out * num_out * sizeof(double));
    if (!h_D)
    {
        fprintf(stderr, "Erro ao alocar memória para find_min_simple\n");
        return;
    }
    
    // Copy from GPU to CPU
    CUDA_CHECK(cudaMemcpy(h_D, d_D, num_out * num_out * sizeof(double), 
                          cudaMemcpyDeviceToHost));
    
    // Find minimum
    *min_val = INFINITY;
    *min_x = 0;
    *min_y = 0;
    
    for (int i = 0; i < num_out; i++)
    {
        for (int j = 0; j < num_out; j++)
        {
            double val = h_D[i * num_out + j];
            if (val < *min_val)
            {
                *min_val = val;
                *min_x = i;
                *min_y = j;
            }
        }
    }
    
    free(h_D);
}

/**
 * CUDA Kernel: Update weights in neighborhood
 * 
 * @param d_W: Weights matrix on GPU
 * @param d_X: Current sample on GPU
 * @param d_min_x: X coordinate of BMU
 * @param d_min_y: Y coordinate of BMU
 * @param alpha: Learning rate
 * @param R: Neighborhood radius
 * @param num_out: Size of SOM map
 * @param num_features: Number of features
 */
__global__ void update_weights_kernel(
    double *d_W,
    const double *d_X,
    int d_min_x,
    int d_min_y,
    double alpha,
    int R,
    int num_out,
    int num_features)
{
    // Calculate which neuron this thread updates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Calculate neighborhood bounds
    int from_x = max(0, d_min_x - R);
    int to_x = min(num_out, d_min_x + R + 1);
    int from_y = max(0, d_min_y - R);
    int to_y = min(num_out, d_min_y + R + 1);
    
    // Check if this thread is in the neighborhood
    if (x < from_x || x >= to_x || y < from_y || y >= to_y)
        return;
    
    // Calculate distance from BMU
    double d2 = (d_min_x - x) * (d_min_x - x) + (d_min_y - y) * (d_min_y - y);
    double scale_factor = exp(-d2 / (2.0 * alpha * alpha));
    
    // Update weights for all features
    int weight_offset = (x * num_out * num_features) + (y * num_features);
    
    for (int k = 0; k < num_features; k++)
    {
        double *w = &d_W[weight_offset + k];
        *w += alpha * scale_factor * (d_X[k] - *w);
    }
}

/**
 * Update weights using CUDA
 * This is the CUDA-accelerated version of kohonen_update_weights
 */
double kohonen_update_weights_cuda(
    const double *d_X,
    double *d_W,
    double *d_D,
    int num_out,
    int num_features,
    double alpha,
    int R)
{
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((num_out + blockSize.x - 1) / blockSize.x,
                  (num_out + blockSize.y - 1) / blockSize.y);
    
    // Step 1: Compute distances (GPU)
    compute_distances_kernel<<<gridSize, blockSize>>>(
        d_W, d_X, d_D, num_out, num_features);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Step 2: Find minimum (CPU - simpler and fast enough for 30x30)
    double d_min;
    int d_min_x, d_min_y;
    find_min_simple(d_D, num_out, &d_min, &d_min_x, &d_min_y);
    
    // Step 3: Update weights in neighborhood (GPU)
    int from_x = max(0, d_min_x - R);
    int to_x = min(num_out, d_min_x + R + 1);
    int from_y = max(0, d_min_y - R);
    int to_y = min(num_out, d_min_y + R + 1);
    
    int neigh_size_x = to_x - from_x;
    int neigh_size_y = to_y - from_y;
    
    if (neigh_size_x > 0 && neigh_size_y > 0)
    {
        dim3 updateBlockSize(8, 8);
        dim3 updateGridSize((neigh_size_x + updateBlockSize.x - 1) / updateBlockSize.x,
                           (neigh_size_y + updateBlockSize.y - 1) / updateBlockSize.y);
        
        update_weights_kernel<<<updateGridSize, updateBlockSize>>>(
            d_W, d_X, d_min_x, d_min_y, alpha, R, num_out, num_features);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    return d_min;
}

/**
 * Main SOM training function with CUDA acceleration
 */
void kohonen_som_cuda(double **X, struct kohonen_array_3d *W, int num_samples,
                     int num_features, int num_out, double alpha_min)
{
    clock_t start_training = clock();
    
    // Allocate GPU memory
    double *d_W = NULL;
    double *d_X_sample = NULL;
    double *d_D = NULL;
    
    size_t weights_size = num_out * num_out * num_features * sizeof(double);
    size_t sample_size = num_features * sizeof(double);
    size_t dist_size = num_out * num_out * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_W, weights_size));
    CUDA_CHECK(cudaMalloc(&d_X_sample, sample_size));
    CUDA_CHECK(cudaMalloc(&d_D, dist_size));
    
    // Copy initial weights to GPU
    CUDA_CHECK(cudaMemcpy(d_W, W->data, weights_size, cudaMemcpyHostToDevice));
    
    int R = num_out >> 2;
    int iter = 0;
    double dmin = 1.0;
    
    printf("🚀 Usando aceleração CUDA para cálculo de distâncias\n");
    
    // Training loop
    for (double alpha = 1.0; alpha > alpha_min && dmin > 1e-3;
         alpha -= 0.001, iter++)
    {
        dmin = 0.0;
        
        // Process each sample
        for (int sample = 0; sample < num_samples; sample++)
        {
            // Copy current sample to GPU
            CUDA_CHECK(cudaMemcpy(d_X_sample, X[sample], sample_size, 
                                  cudaMemcpyHostToDevice));
            
            // Update weights using CUDA
            dmin += kohonen_update_weights_cuda(d_X_sample, d_W, d_D, 
                                                 num_out, num_features, alpha, R);
        }
        
        // Reduce neighborhood radius
        if (iter % 100 == 0 && R > 1)
            R--;
        
        dmin /= num_samples;
        printf("iter: %5d\t alpha: %.4g\t R: %d\td_min: %.4g\r", iter, alpha, R, dmin);
    }
    putchar('\n');
    
    // Copy final weights back to CPU
    CUDA_CHECK(cudaMemcpy(W->data, d_W, weights_size, cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_X_sample));
    CUDA_CHECK(cudaFree(d_D));
    
    clock_t end_training = clock();
    double training_time = ((double)(end_training - start_training)) / CLOCKS_PER_SEC;
    printf("⏱️  Tempo de treinamento (CUDA): %.2f segundos (%.2f minutos)\n", 
           training_time, training_time / 60.0);
}

// Include other necessary functions from original file
// (normalize_data, load_banking_data, save functions, etc.)
// For brevity, these would be copied from kohonen_som_topology.c

/**
 * Helper function to generate random number
 */
double _random(double a, double b)
{
    return ((b - a) * (rand() % 100) / 100.0) + a;
}

/**
 * Save 2D data to CSV
 */
int save_2d_data(const char *fname, double **X, int num_points, int num_features)
{
    FILE *fp = fopen(fname, "wt");
    if (!fp)
    {
        perror(fname);
        return -1;
    }
    
    for (int i = 0; i < num_points; i++)
    {
        for (int j = 0; j < num_features; j++)
        {
            fprintf(fp, "%.4g", X[i][j]);
            if (j < num_features - 1)
                fputc(',', fp);
        }
        if (i < num_points - 1)
            fputc('\n', fp);
    }
    fclose(fp);
    return 0;
}

/**
 * Save U-Matrix
 */
int save_u_matrix(const char *fname, struct kohonen_array_3d *W)
{
    FILE *fp = fopen(fname, "wt");
    if (!fp)
    {
        perror(fname);
        return -1;
    }
    
    int R = max(W->dim1 >> 3, 2);
    
    for (int i = 0; i < W->dim1; i++)
    {
        for (int j = 0; j < W->dim2; j++)
        {
            double distance = 0.0;
            
            int from_x = max(0, i - R);
            int to_x = min(W->dim1, i + R + 1);
            int from_y = max(0, j - R);
            int to_y = min(W->dim2, j + R + 1);
            
            for (int l = from_x; l < to_x; l++)
            {
                for (int m = from_y; m < to_y; m++)
                {
                    double d = 0.0;
                    for (int k = 0; k < W->dim3; k++)
                    {
                        double *w1 = kohonen_data_3d(W, i, j, k);
                        double *w2 = kohonen_data_3d(W, l, m, k);
                        d += (w1[0] - w2[0]) * (w1[0] - w2[0]);
                    }
                    distance += sqrt(d);
                }
            }
            
            distance /= (R * R);
            fprintf(fp, "%.4g", distance);
            if (j < W->dim2 - 1)
                fputc(',', fp);
        }
        if (i < W->dim1 - 1)
            fputc('\n', fp);
    }
    fclose(fp);
    return 0;
}

/**
 * Save SOM weights
 */
int save_som_weights(const char *fname, struct kohonen_array_3d *W)
{
    FILE *fp = fopen(fname, "wt");
    if (!fp)
    {
        perror(fname);
        return -1;
    }
    
    fprintf(fp, "%d,%d,%d\n", W->dim1, W->dim2, W->dim3);
    
    for (int i = 0; i < W->dim1; i++)
    {
        for (int j = 0; j < W->dim2; j++)
        {
            for (int k = 0; k < W->dim3; k++)
            {
                double *w = kohonen_data_3d(W, i, j, k);
                fprintf(fp, "%.6g", w[0]);
                if (k < W->dim3 - 1)
                    fputc(',', fp);
            }
            fputc('\n', fp);
        }
    }
    fclose(fp);
    return 0;
}

/**
 * Get clock difference
 */
double get_clock_diff(clock_t start_t, clock_t end_t)
{
    return (double)(end_t - start_t) / (double)CLOCKS_PER_SEC;
}

/**
 * Print GPU information
 */
void print_gpu_info()
{
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0)
    {
        printf("⚠️  Nenhuma GPU CUDA encontrada!\n");
        return;
    }
    
    printf("🔍 Informações da GPU:\n");
    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("  GPU %d: %s\n", i, prop.name);
        printf("    Memória Global: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("    Multiprocessadores: %d\n", prop.multiProcessorCount);
        printf("    Compute Capability: %d.%d\n", prop.major, prop.minor);
    }
    printf("\n");
}

/* ========== FUNÇÕES PARA PROCESSAR DADOS DO BANKING ========== */

/**
 * Remove aspas de uma string
 */
void remove_quotes(char *str)
{
    if (!str) return;
    int i, j = 0;
    for (i = 0; str[i]; i++)
    {
        if (str[i] != '"')
        {
            str[j++] = str[i];
        }
    }
    str[j] = '\0';
}

/**
 * Converte valor categórico para numérico
 */
double categorical_to_numeric(const char *value, const char **categories, int num_categories)
{
    if (!value) return -1.0;
    
    char temp[256];
    strncpy(temp, value, sizeof(temp) - 1);
    temp[sizeof(temp) - 1] = '\0';
    remove_quotes(temp);
    
    if (strcmp(temp, "unknown") == 0 || strcmp(temp, "") == 0)
        return -1.0;
    
    for (int i = 0; i < num_categories; i++)
    {
        if (strcmp(temp, categories[i]) == 0)
            return (double)i;
    }
    return -1.0;
}

/**
 * Normaliza dados usando min-max scaling
 */
void normalize_data(double **X, int num_samples, int num_features)
{
    double *min_vals = (double *)malloc(num_features * sizeof(double));
    double *max_vals = (double *)malloc(num_features * sizeof(double));
    
    for (int j = 0; j < num_features; j++)
    {
        min_vals[j] = INFINITY;
        max_vals[j] = -INFINITY;
        
        for (int i = 0; i < num_samples; i++)
        {
            if (X[i][j] < min_vals[j])
                min_vals[j] = X[i][j];
            if (X[i][j] > max_vals[j])
                max_vals[j] = X[i][j];
        }
    }
    
    for (int i = 0; i < num_samples; i++)
    {
        for (int j = 0; j < num_features; j++)
        {
            double range = max_vals[j] - min_vals[j];
            if (range > 1e-10)
            {
                X[i][j] = (X[i][j] - min_vals[j]) / range;
            }
            else
            {
                X[i][j] = 0.0;
            }
        }
    }
    
    free(min_vals);
    free(max_vals);
}

/**
 * Lê dados do arquivo CSV do banking market
 */
double **load_banking_data(const char *filename, int *num_samples, int *num_features)
{
    FILE *fp = fopen(filename, "r");
    if (!fp)
    {
        fprintf(stderr, "Erro ao abrir arquivo: %s\n", filename);
        return NULL;
    }
    
    char header[2048];
    if (fgets(header, sizeof(header), fp) == NULL)
    {
        fclose(fp);
        return NULL;
    }
    
    int line_count = 0;
    char buffer[2048];
    while (fgets(buffer, sizeof(buffer), fp))
        line_count++;
    
    rewind(fp);
    fgets(header, sizeof(header), fp);
    
    *num_features = 8;
    *num_samples = line_count;
    
    double **X = (double **)malloc(*num_samples * sizeof(double *));
    for (int i = 0; i < *num_samples; i++)
    {
        X[i] = (double *)malloc(*num_features * sizeof(double));
    }
    
    const char *default_cats[] = {"no", "yes"};
    const char *housing_cats[] = {"no", "yes"};
    const char *loan_cats[] = {"no", "yes"};
    
    int sample_idx = 0;
    while (fgets(buffer, sizeof(buffer), fp) && sample_idx < *num_samples)
    {
        char *line = buffer;
        int col = 0;
        char *start = line;
        
        while (*line && sample_idx < *num_samples)
        {
            if (*line == ';' || *line == '\n' || *line == '\r')
            {
                *line = '\0';
                
                if (col == 0)  // age
                {
                    remove_quotes(start);
                    X[sample_idx][0] = atof(start);
                }
                else if (col == 4)  // default
                {
                    X[sample_idx][5] = categorical_to_numeric(start, default_cats, 2);
                }
                else if (col == 5)  // balance
                {
                    remove_quotes(start);
                    X[sample_idx][1] = atof(start);
                }
                else if (col == 6)  // housing
                {
                    X[sample_idx][6] = categorical_to_numeric(start, housing_cats, 2);
                }
                else if (col == 7)  // loan
                {
                    X[sample_idx][7] = categorical_to_numeric(start, loan_cats, 2);
                }
                else if (col == 11)  // duration
                {
                    remove_quotes(start);
                    X[sample_idx][2] = atof(start);
                }
                else if (col == 12)  // campaign
                {
                    remove_quotes(start);
                    X[sample_idx][3] = atof(start);
                }
                else if (col == 14)  // previous
                {
                    remove_quotes(start);
                    X[sample_idx][4] = atof(start);
                }
                
                if (*line == '\n' || *line == '\r')
                    break;
                    
                start = line + 1;
                col++;
            }
            line++;
        }
        
        sample_idx++;
    }
    
    *num_samples = sample_idx;
    fclose(fp);
    
    printf("Carregados %d amostras com %d features\n", *num_samples, *num_features);
    printf("Features: age, balance, duration, campaign, previous, default, housing, loan\n");
    
    return X;
}

/**
 * Teste com dados do banking market - CUDA version
 */
void test_banking()
{
    clock_t start_total = clock();
    int num_samples, num_features;
    int num_out = 30;
    
    printf("\n=== Processando dados do Banking Market (CUDA) ===\n");
    
    clock_t start_load = clock();
    double **X = load_banking_data("banking_market/train.csv", &num_samples, &num_features);
    if (!X)
    {
        fprintf(stderr, "Erro ao carregar dados!\n");
        return;
    }
    clock_t end_load = clock();
    double load_time = get_clock_diff(start_load, end_load);
    printf("Tempo de carregamento: %.2f segundos\n", load_time);
    
    printf("Normalizando dados...\n");
    clock_t start_norm = clock();
    normalize_data(X, num_samples, num_features);
    clock_t end_norm = clock();
    double norm_time = get_clock_diff(start_norm, end_norm);
    printf("Tempo de normalizacao: %.2f segundos\n", norm_time);
    
    save_2d_data("banking_data_normalized.csv", X, num_samples, num_features);
    printf("Dados normalizados salvos em: banking_data_normalized.csv\n");
    
    struct kohonen_array_3d W;
    W.dim1 = num_out;
    W.dim2 = num_out;
    W.dim3 = num_features;
    W.data = (double *)malloc(num_out * num_out * num_features * sizeof(double));
    
    printf("Inicializando pesos do SOM...\n");
    clock_t start_init = clock();
    for (int i = 0; i < num_out; i++)
    {
        for (int k = 0; k < num_out; k++)
        {
            for (int j = 0; j < num_features; j++)
            {
                double *w = kohonen_data_3d(&W, i, k, j);
                w[0] = _random(0, 1);
            }
        }
    }
    clock_t end_init = clock();
    double init_time = get_clock_diff(start_init, end_init);
    printf("Tempo de inicializacao: %.2f segundos\n", init_time);
    
    clock_t start_save1 = clock();
    save_u_matrix("banking_w_before.csv", &W);
    clock_t end_save1 = clock();
    double save1_time = get_clock_diff(start_save1, end_save1);
    printf("U-matrix inicial salva em: banking_w_before.csv\n");
    printf("Tempo de salvamento (antes): %.2f segundos\n", save1_time);
    
    printf("Treinando SOM com CUDA...\n");
    kohonen_som_cuda(X, &W, num_samples, num_features, num_out, 1e-4);
    
    clock_t start_save2 = clock();
    save_u_matrix("banking_w_after.csv", &W);
    clock_t end_save2 = clock();
    double save2_time = get_clock_diff(start_save2, end_save2);
    printf("U-matrix treinada salva em: banking_w_after.csv\n");
    printf("Tempo de salvamento (depois): %.2f segundos\n", save2_time);
    
    clock_t start_save3 = clock();
    save_som_weights("banking_weights.csv", &W);
    clock_t end_save3 = clock();
    double save3_time = get_clock_diff(start_save3, end_save3);
    printf("Pesos do SOM salvos em: banking_weights.csv\n");
    printf("Tempo de salvamento (pesos): %.2f segundos\n", save3_time);
    
    for (int i = 0; i < num_samples; i++)
        free(X[i]);
    free(X);
    free(W.data);
    
    clock_t end_total = clock();
    double total_time = get_clock_diff(start_total, end_total);
    double training_time = total_time - load_time - norm_time - init_time - save1_time - save2_time - save3_time;
    
    printf("\n============================================================\n");
    printf("RESUMO DE TEMPOS DE EXECUCAO (CUDA)\n");
    printf("============================================================\n");
    printf("  Carregamento de dados:     %8.2f segundos\n", load_time);
    printf("  Normalizacao:              %8.2f segundos\n", norm_time);
    printf("  Inicializacao de pesos:    %8.2f segundos\n", init_time);
    printf("  Salvamento (antes):        %8.2f segundos\n", save1_time);
    printf("  Treinamento do SOM (CUDA): %8.2f segundos (%.2f minutos)\n", 
           training_time, training_time / 60.0);
    printf("  Salvamento (depois):       %8.2f segundos\n", save2_time);
    printf("  Salvamento (pesos):        %8.2f segundos\n", save3_time);
    printf("  ------------------------------------------------------------\n");
    printf("  TEMPO TOTAL:               %8.2f segundos (%.2f minutos)\n", 
           total_time, total_time / 60.0);
    printf("============================================================\n");
    printf("=== Processamento concluido! ===\n\n");
}

/**
 * Main function - CUDA version
 */
int main(int argc, char **argv)
{
    print_gpu_info();
    srand((unsigned int)time(NULL));
    
    if (argc > 1 && strcmp(argv[1], "banking") == 0)
    {
        test_banking();
        return 0;
    }
    
    printf("Uso: %s banking\n", argv[0]);
    printf("Para processar dados do banking market com aceleracao CUDA\n\n");
    
    return 0;
}


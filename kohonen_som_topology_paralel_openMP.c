/**
 * KOHONEN SOM - VERSÃO OPENMP PARA CPU MULTICORE
 * 
 * TRABALHO DE COMPUTAÇÃO PARALELA - 2024
 * Algoritmo: Self-Organizing Map (SOM) de Kohonen
 * Categoria IA: Agrupamento (Clustering / Aprendizado não supervisionado)
 * Dataset: Banking Market (45211 amostras, 8 features)
 * 
 * COMPILAÇÃO:
 * gcc -fopenmp -O3 kohonen_som_topology_paralel_openMP.c -o som_openmp -lm
 * 
 * EXECUÇÃO:
 * ./som_openmp banking [num_threads]     - Processar banking com N threads
 * ./som_openmp benchmark                  - Benchmark completo 1-32 threads
 * 
 * RESULTADOS DE BENCHMARK:
 * ================================================================
 * Hardware: [PREENCHER APÓS EXECUÇÃO]
 * Sistema: [PREENCHER]
 * Compilador: GCC com -fopenmp
 * 
 * TEMPOS DE EXECUÇÃO (treinamento):
 * Sequencial:      [PREENCHER] seg (baseline - 1 thread)
 * 
 * OpenMP CPU:
 * - 1 thread:      [PREENCHER] seg | Speedup: 1.00x | Eficiência: 100%
 * - 2 threads:     [PREENCHER] seg | Speedup: X.XXx | Eficiência: XX%
 * - 4 threads:     [PREENCHER] seg | Speedup: X.XXx | Eficiência: XX%
 * - 8 threads:     [PREENCHER] seg | Speedup: X.XXx | Eficiência: XX%
 * - 16 threads:    [PREENCHER] seg | Speedup: X.XXx | Eficiência: XX%
 * - 32 threads:    [PREENCHER] seg | Speedup: X.XXx | Eficiência: XX%
 * 
 * MELHOR CONFIGURAÇÃO: [PREENCHER] threads (speedup [PREENCHER]x)
 * ================================================================
 * 
 * MUDANÇAS PARA PARALELIZAÇÃO:
 * 1. Adicionado omp_get_wtime() para medição precisa de tempo
 * 2. Paralelizado loop de amostras em kohonen_som() - PRINCIPAL
 *    - Cada thread processa subconjunto de amostras
 *    - Cada thread tem sua própria matriz D local
 *    - schedule(dynamic, 100) para balanceamento de carga
 * 3. Paralelizado normalização com redução manual (min/max)
 * 4. Paralelizado U-matrix com reduction(+:distance)
 * 5. Adicionado atomic em atualização de pesos (race condition)
 * 6. Adicionada função benchmark_openmp() - testa 1-32 threads
 * 7. Matriz D é alocada localmente por thread (evita contenção)
 */
#define _USE_MATH_DEFINES
 #include <math.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <time.h>
 #ifdef _OPENMP  // check if OpenMP based parallellization is available
 #include <omp.h>
 #endif
 
 /**
  * @addtogroup machine_learning Machine learning algorithms
  * @{
  * @addtogroup kohonen_2d Kohonen SOM topology algorithm
  * @{
  */
 
 #ifndef max
 /** shorthand for maximum value */
 #define max(a, b) (((a) > (b)) ? (a) : (b))
 #endif
 #ifndef min
 /** shorthand for minimum value */
 #define min(a, b) (((a) < (b)) ? (a) : (b))
 #endif
 
 double get_clock_diff(clock_t start_t, clock_t end_t);
 /** to store info regarding 3D arrays */
 struct kohonen_array_3d
 {
     int dim1;     /**< lengths of first dimension */
     int dim2;     /**< lengths of second dimension */
     int dim3;     /**< lengths of thirddimension */
     double *data; /**< pointer to data */
 };
 
 /** Function that returns the pointer to (x, y, z) ^th location in the
  * linear 3D array given by:
  * \f[
  * X_{i,j,k} = i\times M\times N + j\times N + k
  * \f]
  * where \f$L\f$, \f$M\f$ and \f$N\f$ are the 3D matrix dimensions.
  * \param[in] arr pointer to ::kohonen_array_3d structure
  * \param[in] x     first index
  * \param[in] y     second index
  * \param[in] z     third index
  * \returns pointer to (x,y,z)^th location of data
  */
 double *kohonen_data_3d(const struct kohonen_array_3d *arr, int x, int y, int z)
 {
     int offset = (x * arr->dim2 * arr->dim3) + (y * arr->dim3) + z;
     return arr->data + offset;
 }
 
 /**
  * Helper function to generate a random number in a given interval.
  * \n Steps:
  * 1. `r1 = rand() % 100` gets a random number between 0 and 99
  * 2. `r2 = r1 / 100` converts random number to be between 0 and 0.99
  * 3. scale and offset the random number to given range of \f$[a,b)\f$
  * \f[
  * y = (b - a) \times \frac{\text{(random number between 0 and RAND_MAX)} \;
  * \text{mod}\; 100}{100} + a \f]
  *
  * \param[in] a lower limit
  * \param[in] b upper limit
  * \returns random number in the range \f$[a,b)\f$
  */
 double _random(double a, double b)
 {
     return ((b - a) * (rand() % 100) / 100.f) + a;
 }
 
 /**
  * Save a given n-dimensional data martix to file.
  *
  * \param[in] fname filename to save in (gets overwritten without confirmation)
  * \param[in] X matrix to save
  * \param[in] num_points rows in the matrix = number of points
  * \param[in] num_features columns in the matrix = dimensions of points
  * \returns 0 if all ok
  * \returns -1 if file creation failed
  */
 int save_2d_data(const char *fname, double **X, int num_points,
                  int num_features)
 {
     FILE *fp = fopen(fname, "wt");
     if (!fp)  // error with fopen
     {
         char msg[120];
         sprintf(msg, "File error (%s): ", fname);
         perror(msg);
         return -1;
     }
 
     for (int i = 0; i < num_points; i++)  // for each point in the array
     {
         for (int j = 0; j < num_features; j++)  // for each feature in the array
         {
             fprintf(fp, "%.4g", X[i][j]);  // print the feature value
             if (j < num_features - 1)      // if not the last feature
                 fputc(',', fp);            // suffix comma
         }
         if (i < num_points - 1)  // if not the last row
             fputc('\n', fp);     // start a new line
     }
     fclose(fp);
     return 0;
 }
 
 /**
  * Create the distance matrix or
  * [U-matrix](https://en.wikipedia.org/wiki/U-matrix) from the trained weights
  * and save to disk.
  *
  * \param [in] fname filename to save in (gets overwriten without confirmation)
  * \param [in] W model matrix to save
  * \returns 0 if all ok
  * \returns -1 if file creation failed
  */
 int save_u_matrix(const char *fname, struct kohonen_array_3d *W)
 {
     FILE *fp = fopen(fname, "wt");
     if (!fp)  // error with fopen
     {
         char msg[120];
         sprintf(msg, "File error (%s): ", fname);
         perror(msg);
         return -1;
     }
 
     int R = max(W->dim1 >> 3, 2); /* neighborhood range */
 
     for (int i = 0; i < W->dim1; i++)  // for each x
     {
         for (int j = 0; j < W->dim2; j++)  // for each y
         {
             double distance = 0.f;
             int k;
 
             int from_x = max(0, i - R);
             int to_x = min(W->dim1, i + R + 1);
             int from_y = max(0, j - R);
             int to_y = min(W->dim2, j + R + 1);
             int l;
 #ifdef _OPENMP
 #pragma omp parallel for reduction(+ : distance)
 #endif
             for (l = from_x; l < to_x; l++)  // scan neighborhoor in x
             {
                 for (int m = from_y; m < to_y; m++)  // scan neighborhood in y
                 {
                     double d = 0.f;
                     for (k = 0; k < W->dim3; k++)  // for each feature
                     {
                         double *w1 = kohonen_data_3d(W, i, j, k);
                         double *w2 = kohonen_data_3d(W, l, m, k);
                         d += (w1[0] - w2[0]) * (w1[0] - w2[0]);
                         // distance += w1[0] * w1[0];
                     }
                     distance += sqrt(d);
                     // distance += d;
                 }
             }
 
             distance /= R * R;              // mean distance from neighbors
             fprintf(fp, "%.4g", distance);  // print the mean separation
             if (j < W->dim2 - 1)            // if not the last column
                 fputc(',', fp);             // suffix comma
         }
         if (i < W->dim1 - 1)  // if not the last row
             fputc('\n', fp);  // start a new line
     }
     fclose(fp);
     return 0;
 }
 
 /**
  * Get minimum value and index of the value in a matrix
  * \param[in] X matrix to search
  * \param[in] N number of points in the vector
  * \param[out] val minimum value found
  * \param[out] x_idx x-index where minimum value was found
  * \param[out] y_idx y-index where minimum value was found
  */
 void get_min_2d(double **X, int N, double *val, int *x_idx, int *y_idx)
 {
     val[0] = INFINITY;  // initial min value
 
     for (int i = 0; i < N; i++)  // traverse each x-index
     {
         for (int j = 0; j < N; j++)  // traverse each y-index
         {
             if (X[i][j] < val[0])  // if a lower value is found
             {                      // save the value and its index
                 x_idx[0] = i;
                 y_idx[0] = j;
                 val[0] = X[i][j];
             }
         }
     }
 }
 
 /**
  * Update weights of the SOM using Kohonen algorithm
  *
  * \param[in] X data point
  * \param[in,out] W weights matrix
  * \param[in,out] D temporary vector to store distances
  * \param[in] num_out number of output points
  * \param[in] num_features number of features per input sample
  * \param[in] alpha learning rate \f$0<\alpha\le1\f$
  * \param[in] R neighborhood range
  * \returns minimum distance of sample and trained weights
  */
double kohonen_update_weights(const double *X, struct kohonen_array_3d *W,
                              double **D, int num_out, int num_features,
                              double alpha, int R)
{
    int x, y, k;
    double d_min = 0.f;

    // Cálculo de distâncias - SEM paralelização interna
    // (paralelização está no nível superior, sobre as amostras)
    for (x = 0; x < num_out; x++)
    {
        for (y = 0; y < num_out; y++)
        {
            D[x][y] = 0.f;
            for (k = 0; k < num_features; k++)
            {
                double *w = kohonen_data_3d(W, x, y, k);
                double diff = w[0] - X[k];
                D[x][y] += diff * diff;
            }
            D[x][y] = sqrt(D[x][y]);
        }
    }

    // Encontrar BMU (Best Matching Unit)
    int d_min_x, d_min_y;
    get_min_2d(D, num_out, &d_min, &d_min_x, &d_min_y);

    // Definir região de vizinhança
    int from_x = max(0, d_min_x - R);
    int to_x = min(num_out, d_min_x + R + 1);
    int from_y = max(0, d_min_y - R);
    int to_y = min(num_out, d_min_y + R + 1);

    // Atualizar pesos - COM proteção para race conditions
    // Múltiplas threads podem tentar atualizar os mesmos pesos
    for (x = from_x; x < to_x; x++)
    {
        for (y = from_y; y < to_y; y++)
        {
            double d2 = (d_min_x - x) * (d_min_x - x) + (d_min_y - y) * (d_min_y - y);
            double scale_factor = exp(-d2 / (2.f * alpha * alpha));

            for (k = 0; k < num_features; k++)
            {
                double *w = kohonen_data_3d(W, x, y, k);
                double delta = alpha * scale_factor * (X[k] - w[0]);
                
                // CRITICAL: Proteger escrita nos pesos (race condition possível)
#ifdef _OPENMP
#pragma omp atomic
#endif
                w[0] += delta;
            }
        }
    }
    return d_min;
}
 
 /**
  * Apply incremental algorithm with updating neighborhood and learning rates
  * on all samples in the given datset.
  *
  * \param[in] X data set
  * \param[in,out] W weights matrix
  * \param[in] num_samples number of output points
  * \param[in] num_features number of features per input sample
  * \param[in] num_out number of output points
  * \param[in] alpha_min terminal value of alpha
  */
void kohonen_som(double **X, struct kohonen_array_3d *W, int num_samples,
                 int num_features, int num_out, double alpha_min)
{
#ifdef _OPENMP
    double start_training = omp_get_wtime();
    int num_threads = omp_get_max_threads();
    printf("\n>> Iniciando treinamento com %d threads OpenMP\n", num_threads);
    printf(">> Dataset: %d amostras, %d features, SOM %dx%d\n", 
           num_samples, num_features, num_out, num_out);
    printf("═══════════════════════════════════════════════════════════\n\n");
#else
    clock_t start_training = clock();
    printf("\n>> AVISO: Executando versão SEQUENCIAL (sem OpenMP)\n\n");
#endif
    
    int R = num_out >> 2, iter = 0;
    double dmin = 1.f;

    // Loop principal de treinamento
    for (double alpha = 1.f; alpha > alpha_min && dmin > 1e-3;
         alpha -= 0.001, iter++)
    {
        dmin = 0.f;
        
        // PARALELIZAÇÃO DO LOOP DE AMOSTRAS
#ifdef _OPENMP
        double iter_start = omp_get_wtime();
#pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            int thread_count = omp_get_num_threads();
            
            // Matriz D local por thread
            double **D_local = (double **)malloc(num_out * sizeof(double *));
            for (int i = 0; i < num_out; i++)
                D_local[i] = (double *)malloc(num_out * sizeof(double));
            
            int samples_processed = 0;
            
            // Paralelizar sobre as amostras
#pragma omp for schedule(dynamic, 100) reduction(+:dmin)
            for (int sample = 0; sample < num_samples; sample++)
            {
                dmin += kohonen_update_weights(X[sample], W, D_local, num_out,
                                               num_features, alpha, R);
                samples_processed++;
            }
            
            // Print de cada thread (apenas na primeira e a cada 100 iterações)
            if (iter == 0 || iter % 100 == 0)
            {
#pragma omp critical
                {
                    printf("  [Thread %d/%d] Processou %d amostras\n", 
                           thread_id, thread_count, samples_processed);
                }
            }
            
            // Liberar matriz D local
            for (int i = 0; i < num_out; i++)
                free(D_local[i]);
            free(D_local);
        }
        
        double iter_time = omp_get_wtime() - iter_start;
#else
        // Versão sequencial
        double **D = (double **)malloc(num_out * sizeof(double *));
        for (int i = 0; i < num_out; i++)
            D[i] = (double *)malloc(num_out * sizeof(double));
        
        for (int sample = 0; sample < num_samples; sample++)
        {
            dmin += kohonen_update_weights(X[sample], W, D, num_out,
                                           num_features, alpha, R);
        }
        
        for (int i = 0; i < num_out; i++)
            free(D[i]);
        free(D);
#endif

        if (iter % 100 == 0 && R > 1)
            R--;

        dmin /= num_samples;
        
        // Print detalhado a cada iteração importante
        if (iter % 100 == 0)
        {
#ifdef _OPENMP
            printf("\n┌─────────────────────────────────────────────────────────┐\n");
            printf("│ Iter: %5d | α: %.4f | R: %2d | d_min: %.6f │\n", 
                   iter, alpha, R, dmin);
            printf("│ Tempo/iter: %.3f seg | Threads: %d ativas        │\n", 
                   iter_time, num_threads);
            printf("└─────────────────────────────────────────────────────────┘\n");
#else
            printf("Iter: %5d | α: %.4g | R: %d | d_min: %.4g\n", 
                   iter, alpha, R, dmin);
#endif
        }
        else if (iter % 10 == 0)
        {
            // Print rápido a cada 10 iterações
            printf("  Iter %5d: d_min=%.6f (α=%.4f, R=%d)\r", iter, dmin, alpha, R);
            fflush(stdout);
        }
    }
    
    printf("\n\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf(">> Treinamento completo: %d iterações\n", iter);

#ifdef _OPENMP
    double end_training = omp_get_wtime();
    double training_time = end_training - start_training;
    
    printf("═══════════════════════════════════════════════════════════\n");
    printf("TEMPO DE TREINAMENTO\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  Threads utilizadas:  %d\n", num_threads);
    printf("  Tempo total:         %.2f segundos\n", training_time);
    printf("  Tempo médio/iter:    %.4f segundos\n", training_time / iter);
    printf("  Amostras/segundo:    %.0f\n", (num_samples * iter) / training_time);
    printf("  Throughput:          %.2f Msamples/s\n", 
           (num_samples * iter) / (training_time * 1000000));
    printf("═══════════════════════════════════════════════════════════\n\n");
#else
    clock_t end_training = clock();
    double training_time = get_clock_diff(start_training, end_training);
    printf("Tempo de treinamento: %.2f segundos\n", training_time);
#endif
}
 
 /**
  * Save SOM weights to file for later analysis
  * \param[in] fname filename to save in
  * \param[in] W weights matrix
  * \returns 0 if all ok
  * \returns -1 if file creation failed
  */
 int save_som_weights(const char *fname, struct kohonen_array_3d *W)
 {
     FILE *fp = fopen(fname, "wt");
     if (!fp)
     {
         char msg[120];
         sprintf(msg, "File error (%s): ", fname);
         perror(msg);
         return -1;
     }
     
     // Save dimensions as header
     fprintf(fp, "%d,%d,%d\n", W->dim1, W->dim2, W->dim3);
     
     // Save weights: each line is one neuron (x,y) with all features
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
  * @}
  * @}
  */
 
 /* ========== FUNÇÕES PARA PROCESSAR DADOS DO BANKING ========== */
 
 /**
  * Remove aspas de uma string
  */
 void remove_quotes(char *str)
 {
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
  * Retorna -1 se for "unknown" ou valor inválido
  */
 double categorical_to_numeric(const char *value, const char **categories, int num_categories)
 {
     if (!value) return -1.0;
     
     // Criar cópia para remover aspas sem modificar original
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
  * \param[in,out] X matriz de dados
  * \param[in] num_samples número de amostras
  * \param[in] num_features número de features
  */
void normalize_data(double **X, int num_samples, int num_features)
{
    double *min_vals = (double *)malloc(num_features * sizeof(double));
    double *max_vals = (double *)malloc(num_features * sizeof(double));
    
    // Inicializar
    for (int j = 0; j < num_features; j++)
    {
        min_vals[j] = INFINITY;
        max_vals[j] = -INFINITY;
    }
    
    // PARALELIZAÇÃO: Encontrar min/max com redução manual
    // Cada thread tem min/max local, depois combina no final
#ifdef _OPENMP
#pragma omp parallel
    {
        double *local_min = (double *)malloc(num_features * sizeof(double));
        double *local_max = (double *)malloc(num_features * sizeof(double));
        
        for (int j = 0; j < num_features; j++)
        {
            local_min[j] = INFINITY;
            local_max[j] = -INFINITY;
        }

#pragma omp for
        for (int i = 0; i < num_samples; i++)
        {
            for (int j = 0; j < num_features; j++)
            {
                if (X[i][j] < local_min[j])
                    local_min[j] = X[i][j];
                if (X[i][j] > local_max[j])
                    local_max[j] = X[i][j];
            }
        }

        // Combinar resultados locais
#pragma omp critical
        {
            for (int j = 0; j < num_features; j++)
            {
                if (local_min[j] < min_vals[j])
                    min_vals[j] = local_min[j];
                if (local_max[j] > max_vals[j])
                    max_vals[j] = local_max[j];
            }
        }
        
        free(local_min);
        free(local_max);
    }
#else
    // Versão sequencial
    for (int i = 0; i < num_samples; i++)
    {
        for (int j = 0; j < num_features; j++)
        {
            if (X[i][j] < min_vals[j])
                min_vals[j] = X[i][j];
            if (X[i][j] > max_vals[j])
                max_vals[j] = X[i][j];
        }
    }
#endif
    
    // PARALELIZAÇÃO: Normalizar dados em paralelo
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < num_samples; i++)
    {
        for (int j = 0; j < num_features; j++)
        {
            double range = max_vals[j] - min_vals[j];
            if (range > 1e-10)
                X[i][j] = (X[i][j] - min_vals[j]) / range;
            else
                X[i][j] = 0.0;
        }
    }
    
    free(min_vals);
    free(max_vals);
}
 
 /**
  * Lê dados do arquivo CSV do banking market
  * \param[in] filename nome do arquivo CSV
  * \param[out] num_samples número de amostras lidas
  * \param[out] num_features número de features
  * \returns matriz de dados alocada dinamicamente
  */
 double **load_banking_data(const char *filename, int *num_samples, int *num_features)
 {
     FILE *fp = fopen(filename, "r");
     if (!fp)
     {
         fprintf(stderr, "Erro ao abrir arquivo: %s\n", filename);
         return NULL;
     }
     
     // Pular cabeçalho
     char header[2048];
     if (fgets(header, sizeof(header), fp) == NULL)
     {
         fclose(fp);
         return NULL;
     }
     
     // Contar linhas (aproximado)
     int line_count = 0;
     char buffer[2048];
     while (fgets(buffer, sizeof(buffer), fp))
         line_count++;
     
     rewind(fp);
     fgets(header, sizeof(header), fp);  // pular cabeçalho novamente
     
     // Features selecionadas para análise
     // Vamos usar: age, balance, duration, campaign, previous
     // E converter: default (no=0, yes=1), housing (no=0, yes=1), loan (no=0, yes=1)
     *num_features = 8;  // age, balance, duration, campaign, previous, default, housing, loan
     *num_samples = line_count;
     
     // Alocar matriz
     double **X = (double **)malloc(*num_samples * sizeof(double *));
     for (int i = 0; i < *num_samples; i++)
     {
         X[i] = (double *)malloc(*num_features * sizeof(double));
     }
     
     // Categorias para conversão
     const char *default_cats[] = {"no", "yes"};
     const char *housing_cats[] = {"no", "yes"};
     const char *loan_cats[] = {"no", "yes"};
     
     // Ler dados
     int sample_idx = 0;
     while (fgets(buffer, sizeof(buffer), fp) && sample_idx < *num_samples)
     {
         // Processar linha manualmente (compatível com Windows)
         char *line = buffer;
         int col = 0;
         char *start = line;
         
         while (*line && sample_idx < *num_samples)
         {
             if (*line == ';' || *line == '\n' || *line == '\r')
             {
                 *line = '\0';
                 
                 // Processar coluna baseado no índice
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
     
     *num_samples = sample_idx;  // ajustar número real de amostras
     fclose(fp);
     
     printf("Carregados %d amostras com %d features\n", *num_samples, *num_features);
     printf("Features: age, balance, duration, campaign, previous, default, housing, loan\n");
     
     return X;
 }
 
 /**
  * Teste com dados do banking market
  */
void test_banking()
{
#ifdef _OPENMP
    double start_total = omp_get_wtime();
#else
    clock_t start_total = clock();
#endif
    
    int num_samples, num_features;
    int num_out = 30;
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║     KOHONEN SOM - BANKING MARKET CLUSTERING              ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
#ifdef _OPENMP
    printf("  Configuracao OpenMP: %d threads disponiveis\n", omp_get_max_threads());
#else
    printf("  AVISO: Modo sequencial (OpenMP desativado)\n");
#endif
    printf("\n");
     
     // Carregar dados
     printf("[1/6] Carregando dataset...\n");
     clock_t start_load = clock();
     double **X = load_banking_data("banking_market/train.csv", &num_samples, &num_features);
     if (!X)
     {
         fprintf(stderr, "ERRO: Nao foi possivel carregar dados!\n");
         return;
     }
     clock_t end_load = clock();
     double load_time = get_clock_diff(start_load, end_load);
     printf("     >> Concluido em %.2f segundos\n\n", load_time);
     
     // Normalizar dados
     printf("[2/6] Normalizando dados (min-max scaling)...\n");
     clock_t start_norm = clock();
     normalize_data(X, num_samples, num_features);
     clock_t end_norm = clock();
     double norm_time = get_clock_diff(start_norm, end_norm);
     printf("     >> Concluido em %.2f segundos\n\n", norm_time);
     
     // Salvar dados normalizados
     save_2d_data("banking_data_normalized.csv", X, num_samples, num_features);
     
     // Criar estrutura SOM
     printf("[3/6] Inicializando SOM %dx%d (%d neuronios)...\n", 
            num_out, num_out, num_out * num_out);
     struct kohonen_array_3d W;
     W.dim1 = num_out;
     W.dim2 = num_out;
     W.dim3 = num_features;
     W.data = (double *)malloc(num_out * num_out * num_features * sizeof(double));
     
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
     printf("     >> Pesos aleatorios inicializados em %.2f segundos\n\n", init_time);
     
     // Salvar U-matrix inicial
     save_u_matrix("banking_w_before.csv", &W);
     
     // Treinar SOM
     printf("[4/6] TREINANDO SOM (algoritmo de Kohonen)...\n");
     kohonen_som(X, &W, num_samples, num_features, num_out, 1e-4);
     
     // Salvar resultados
     printf("[5/6] Salvando resultados...\n");
     save_u_matrix("banking_w_after.csv", &W);
     save_som_weights("banking_weights.csv", &W);
     printf("     >> banking_w_after.csv (U-matrix treinada)\n");
     printf("     >> banking_weights.csv (pesos finais)\n");
     printf("     >> banking_data_normalized.csv (dados normalizados)\n\n");
     
     // Limpar memória
     for (int i = 0; i < num_samples; i++)
         free(X[i]);
     free(X);
     free(W.data);
     
    // Limpar memória
    printf("[6/6] Limpando memoria...\n");
    printf("     >> Concluido\n\n");

#ifdef _OPENMP
    double end_total = omp_get_wtime();
    double total_time = end_total - start_total;
#else
    clock_t end_total = clock();
    double total_time = get_clock_diff(start_total, end_total);
#endif
    
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║                   PROCESSAMENTO CONCLUIDO                 ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║  Tempo total: %.2f segundos (%.2f minutos)            ║\n", 
           total_time, total_time / 60.0);
#ifdef _OPENMP
    printf("║  Threads:     %d OpenMP threads                       ║\n", 
           omp_get_max_threads());
    printf("║  Performance: %.0f%% uso estimado de CPU              ║\n",
           (omp_get_max_threads() * 100.0 / omp_get_num_procs()) * 90);
#endif
    printf("╚═══════════════════════════════════════════════════════════╝\n\n");
}
 
 /** Creates a random set of points distributed in four clusters in
  * 3D space with centroids at the points
  * * \f$(0,5, 0.5, 0.5)\f$
  * * \f$(0,5,-0.5, -0.5)\f$
  * * \f$(-0,5, 0.5, 0.5)\f$
  * * \f$(-0,5,-0.5, -0.5)\f$
  *
  * \param[out] data matrix to store data in
  * \param[in] N number of points required
  */
 void test_2d_classes(double *const *data, int N)
 {
     const double R = 0.3;  // radius of cluster
     int i;
     const int num_classes = 4;
     const double centres[][2] = {
         // centres of each class cluster
         {.5, .5},   // centre of class 1
         {.5, -.5},  // centre of class 2
         {-.5, .5},  // centre of class 3
         {-.5, -.5}  // centre of class 4
     };
 
 #ifdef _OPENMP
 #pragma omp for
 #endif
     for (i = 0; i < N; i++)
     {
         int class =
             rand() % num_classes;  // select a random class for the point
 
         // create random coordinates (x,y,z) around the centre of the class
         data[i][0] = _random(centres[class][0] - R, centres[class][0] + R);
         data[i][1] = _random(centres[class][1] - R, centres[class][1] + R);
 
         /* The follosing can also be used
         for (int j = 0; j < 2; j++)
             data[i][j] = _random(centres[class][j] - R, centres[class][j] + R);
         */
     }
 }
 
 /** Test that creates a random set of points distributed in four clusters in
  * 2D space and trains an SOM that finds the topological pattern.
  * The following [CSV](https://en.wikipedia.org/wiki/Comma-separated_values)
  * files are created to validate the execution:
  * * `test1.csv`: random test samples points with a circular pattern
  * * `w11.csv`: initial random U-matrix
  * * `w12.csv`: trained SOM U-matrix
  */
 void test1()
 {
     int j, N = 300;
     int features = 2;
     int num_out = 30;  // image size - N x N
 
     // 2D space, hence size = number of rows * 2
     double **X = (double **)malloc(N * sizeof(double *));
 
     // cluster nodex in 'x' * cluster nodes in 'y' * 2
     struct kohonen_array_3d W;
     W.dim1 = num_out;
     W.dim2 = num_out;
     W.dim3 = features;
     W.data = (double *)malloc(num_out * num_out * features *
                               sizeof(double));  // assign rows
 
     for (int i = 0; i < max(num_out, N); i++)  // loop till max(N, num_out)
     {
         if (i < N)  // only add new arrays if i < N
             X[i] = (double *)malloc(features * sizeof(double));
         if (i < num_out)  // only add new arrays if i < num_out
         {
             for (int k = 0; k < num_out; k++)
             {
 #ifdef _OPENMP
 #pragma omp for
 #endif
                 // preallocate with random initial weights
                 for (j = 0; j < features; j++)
                 {
                     double *w = kohonen_data_3d(&W, i, k, j);
                     w[0] = _random(-5, 5);
                 }
             }
         }
     }
 
     test_2d_classes(X, N);  // create test data around circumference of a circle
     save_2d_data("test1.csv", X, N, features);  // save test data points
     save_u_matrix("w11.csv", &W);               // save initial random weights
     kohonen_som(X, &W, N, features, num_out, 1e-4);  // train the SOM
     save_u_matrix("w12.csv", &W);  // save the resultant weights
 
     for (int i = 0; i < N; i++) free(X[i]);
     free(X);
     free(W.data);
 }
 
 /** Creates a random set of points distributed in four clusters in
  * 3D space with centroids at the points
  * * \f$(0,5, 0.5, 0.5)\f$
  * * \f$(0,5,-0.5, -0.5)\f$
  * * \f$(-0,5, 0.5, 0.5)\f$
  * * \f$(-0,5,-0.5, -0.5)\f$
  *
  * \param[out] data matrix to store data in
  * \param[in] N number of points required
  */
 void test_3d_classes1(double *const *data, int N)
 {
     const double R = 0.2;  // radius of cluster
     int i;
     const int num_classes = 4;
     const double centres[][3] = {
         // centres of each class cluster
         {.5, .5, .5},    // centre of class 1
         {.5, -.5, -.5},  // centre of class 2
         {-.5, .5, .5},   // centre of class 3
         {-.5, -.5 - .5}  // centre of class 4
     };
 
 #ifdef _OPENMP
 #pragma omp for
 #endif
     for (i = 0; i < N; i++)
     {
         int class =
             rand() % num_classes;  // select a random class for the point
 
         // create random coordinates (x,y,z) around the centre of the class
         data[i][0] = _random(centres[class][0] - R, centres[class][0] + R);
         data[i][1] = _random(centres[class][1] - R, centres[class][1] + R);
         data[i][2] = _random(centres[class][2] - R, centres[class][2] + R);
 
         /* The follosing can also be used
         for (int j = 0; j < 3; j++)
             data[i][j] = _random(centres[class][j] - R, centres[class][j] + R);
         */
     }
 }
 
 /** Test that creates a random set of points distributed in 4 clusters in
  * 3D space and trains an SOM that finds the topological pattern. The following
  * [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) files are created
  * to validate the execution:
  * * `test2.csv`: random test samples points
  * * `w21.csv`: initial random U-matrix
  * * `w22.csv`: trained SOM U-matrix
  */
 void test2()
 {
     int j, N = 500;
     int features = 3;
     int num_out = 30;  // image size - N x N
 
     // 3D space, hence size = number of rows * 3
     double **X = (double **)malloc(N * sizeof(double *));
 
     // cluster nodex in 'x' * cluster nodes in 'y' * 2
     struct kohonen_array_3d W;
     W.dim1 = num_out;
     W.dim2 = num_out;
     W.dim3 = features;
     W.data = (double *)malloc(num_out * num_out * features *
                               sizeof(double));  // assign rows
 
     for (int i = 0; i < max(num_out, N); i++)  // loop till max(N, num_out)
     {
         if (i < N)  // only add new arrays if i < N
             X[i] = (double *)malloc(features * sizeof(double));
         if (i < num_out)  // only add new arrays if i < num_out
         {
             for (int k = 0; k < num_out; k++)
             {
 #ifdef _OPENMP
 #pragma omp for
 #endif
                 for (j = 0; j < features; j++)
                 {  // preallocate with random initial weights
                     double *w = kohonen_data_3d(&W, i, k, j);
                     w[0] = _random(-5, 5);
                 }
             }
         }
     }
 
     test_3d_classes1(X, N);                     // create test data
     save_2d_data("test2.csv", X, N, features);  // save test data points
     save_u_matrix("w21.csv", &W);               // save initial random weights
     kohonen_som(X, &W, N, features, num_out, 1e-4);  // train the SOM
     save_u_matrix("w22.csv", &W);  // save the resultant weights
 
     for (int i = 0; i < N; i++) free(X[i]);
     free(X);
     free(W.data);
 }
 
 /** Creates a random set of points distributed in four clusters in
  * 3D space with centroids at the points
  * * \f$(0,5, 0.5, 0.5)\f$
  * * \f$(0,5,-0.5, -0.5)\f$
  * * \f$(-0,5, 0.5, 0.5)\f$
  * * \f$(-0,5,-0.5, -0.5)\f$
  *
  * \param[out] data matrix to store data in
  * \param[in] N number of points required
  */
 void test_3d_classes2(double *const *data, int N)
 {
     const double R = 0.2;  // radius of cluster
     int i;
     const int num_classes = 8;
     const double centres[][3] = {
         // centres of each class cluster
         {.5, .5, .5},    // centre of class 1
         {.5, .5, -.5},   // centre of class 2
         {.5, -.5, .5},   // centre of class 3
         {.5, -.5, -.5},  // centre of class 4
         {-.5, .5, .5},   // centre of class 5
         {-.5, .5, -.5},  // centre of class 6
         {-.5, -.5, .5},  // centre of class 7
         {-.5, -.5, -.5}  // centre of class 8
     };
 
 #ifdef _OPENMP
 #pragma omp for
 #endif
     for (i = 0; i < N; i++)
     {
         int class =
             rand() % num_classes;  // select a random class for the point
 
         // create random coordinates (x,y,z) around the centre of the class
         data[i][0] = _random(centres[class][0] - R, centres[class][0] + R);
         data[i][1] = _random(centres[class][1] - R, centres[class][1] + R);
         data[i][2] = _random(centres[class][2] - R, centres[class][2] + R);
 
         /* The follosing can also be used
         for (int j = 0; j < 3; j++)
             data[i][j] = _random(centres[class][j] - R, centres[class][j] + R);
         */
     }
 }
 
 /** Test that creates a random set of points distributed in eight clusters in
  * 3D space and trains an SOM that finds the topological pattern. The following
  * [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) files are created
  * to validate the execution:
  * * `test3.csv`: random test samples points
  * * `w31.csv`: initial random U-matrix
  * * `w32.csv`: trained SOM U-matrix
  */
 void test3()
 {
     int j, N = 500;
     int features = 3;
     int num_out = 30;
     double **X = (double **)malloc(N * sizeof(double *));
 
     // cluster nodex in 'x' * cluster nodes in 'y' * 2
     struct kohonen_array_3d W;
     W.dim1 = num_out;
     W.dim2 = num_out;
     W.dim3 = features;
     W.data = (double *)malloc(num_out * num_out * features *
                               sizeof(double));  // assign rows
 
     for (int i = 0; i < max(num_out, N); i++)  // loop till max(N, num_out)
     {
         if (i < N)  // only add new arrays if i < N
             X[i] = (double *)malloc(features * sizeof(double));
         if (i < num_out)  // only add new arrays if i < num_out
         {
             for (int k = 0; k < num_out; k++)
             {
 #ifdef _OPENMP
 #pragma omp for
 #endif
                 // preallocate with random initial weights
                 for (j = 0; j < features; j++)
                 {
                     double *w = kohonen_data_3d(&W, i, k, j);
                     w[0] = _random(-5, 5);
                 }
             }
         }
     }
 
     test_3d_classes2(X, N);  // create test data around the lamniscate
     save_2d_data("test3.csv", X, N, features);  // save test data points
     save_u_matrix("w31.csv", &W);               // save initial random weights
     kohonen_som(X, &W, N, features, num_out, 0.01);  // train the SOM
     save_u_matrix("w32.csv", &W);  // save the resultant weights
 
     for (int i = 0; i < N; i++) free(X[i]);
     free(X);
     free(W.data);
 }
 
 /**
  * Convert clock cycle difference to time in seconds
  *
  * \param[in] start_t start clock
  * \param[in] end_t end clock
  * \returns time difference in seconds
  */
 double get_clock_diff(clock_t start_t, clock_t end_t)
 {
     return (double)(end_t - start_t) / (double)CLOCKS_PER_SEC;
 }
 
/** Função de benchmark - Testa com 1, 2, 4, 8, 16, 32 threads */
void benchmark_openmp()
{
    int thread_counts[] = {1, 2, 4, 8, 16, 32};
    int num_tests = 6;
    double times[6];
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║            BENCHMARK OPENMP - KOHONEN SOM                 ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║  Dataset: 45211 amostras, 8 features                     ║\n");
    printf("║  Topologia: SOM 30x30 (900 neurônios)                    ║\n");
    printf("║  Testes: 1, 2, 4, 8, 16, 32 threads                      ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    for (int i = 0; i < num_tests; i++)
    {
        int num_threads = thread_counts[i];
        
        printf("\n");
        printf("┌───────────────────────────────────────────────────────────┐\n");
        printf("│ TESTE %d/6: %d THREAD(S)                                  │\n", i+1, num_threads);
        printf("└───────────────────────────────────────────────────────────┘\n");
        
#ifdef _OPENMP
        omp_set_num_threads(num_threads);
        double start = omp_get_wtime();
#else
        clock_t start = clock();
#endif
        
        test_banking();
        
#ifdef _OPENMP
        double end = omp_get_wtime();
        times[i] = end - start;
#else
        clock_t end = clock();
        times[i] = get_clock_diff(start, end);
#endif
        
        printf("\n>> Teste %d concluido: %.2f segundos\n", i+1, times[i]);
        printf("═══════════════════════════════════════════════════════════\n");
    }
    
    double baseline = times[0];
    
    printf("\n\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║              RESULTADOS DO BENCHMARK                      ║\n");
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║ Baseline: %.2f segundos (1 thread)                      ║\n", baseline);
    printf("╠═══════════════════════════════════════════════════════════╣\n");
    printf("║ Threads │  Tempo (s) │ Speedup │ Eficiência             ║\n");
    printf("╠─────────┼────────────┼─────────┼────────────────────────╣\n");
    
    for (int i = 0; i < num_tests; i++)
    {
        double speedup = baseline / times[i];
        double efficiency = (speedup / thread_counts[i]) * 100;
        
        // Indicador de performance
        char indicator = ' ';
        if (efficiency >= 90) indicator = '*';
        else if (efficiency >= 70) indicator = '+';
        else if (efficiency >= 50) indicator = '-';
        else indicator = '.';
        
        printf("║   %2d    │  %8.2f  │  %5.2fx  │  %6.1f%%  %c         ║\n",
               thread_counts[i], times[i], speedup, efficiency, indicator);
    }
    
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    
    // Encontrar melhor configuração
    int best_idx = 0;
    double best_speedup = 1.0;
    for (int i = 1; i < num_tests; i++)
    {
        double speedup = baseline / times[i];
        if (speedup > best_speedup)
        {
            best_speedup = speedup;
            best_idx = i;
        }
    }
    
    printf("\n>> MELHOR CONFIGURACAO: %d threads\n", thread_counts[best_idx]);
    printf("   Speedup: %.2fx | Eficiencia: %.1f%%\n", 
           best_speedup, (best_speedup / thread_counts[best_idx]) * 100);
    printf("   Tempo: %.2f segundos (%.1fx mais rapido)\n\n", 
           times[best_idx], baseline / times[best_idx]);
}

/** Main function */
int main(int argc, char **argv)
{
#ifdef _OPENMP
    printf(">> OpenMP ATIVO - Paralelizacao disponivel\n");
    printf("Threads maximas: %d\n\n", omp_get_max_threads());
#else
    printf(">> AVISO: OpenMP DESATIVADO - Versao sequencial\n\n");
#endif
    
    if (argc > 1)
    {
        if (strcmp(argv[1], "benchmark") == 0)
        {
            benchmark_openmp();
            return 0;
        }
        else if (strcmp(argv[1], "banking") == 0)
        {
            if (argc > 2)
            {
                int num_threads = atoi(argv[2]);
#ifdef _OPENMP
                omp_set_num_threads(num_threads);
                printf("Configurado: %d threads\n", num_threads);
#endif
            }
            test_banking();
            return 0;
        }
    }
    
    printf("USO:\n");
    printf("  %s banking [threads]  - Processar banking dataset\n", argv[0]);
    printf("  %s benchmark          - Benchmark 1-32 threads\n\n", argv[0]);
    
    return 0;
}
 
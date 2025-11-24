/**
 * \file
 * \brief [Kohonen self organizing
 * map](https://en.wikipedia.org/wiki/Self-organizing_map) (topological map)
 *
 * This example implements a powerful unsupervised learning algorithm called as
 * a self organizing map. The algorithm creates a connected network of weights
 * that closely follows the given data points. This thus creates a topological
 * map of the given data i.e., it maintains the relationship between various
 * data points in a much higher dimensional space by creating an equivalent in a
 * 2-dimensional space.
 * <img alt="Trained topological maps for the test cases in the program"
 * src="https://raw.githubusercontent.com/TheAlgorithms/C/docs/images/machine_learning/kohonen/2D_Kohonen_SOM.svg"
 * />
 * \author [Krishna Vedala](https://github.com/kvedala)
 * \warning MSVC 2019 compiler generates code that does not execute as expected.
 * However, MinGW, Clang for GCC and Clang for MSVC compilers on windows perform
 * as expected. Any insights and suggestions should be directed to the author.
 * \see kohonen_som_trace.c
 */
#define _USE_MATH_DEFINES /**< required for MS Visual C */
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

#ifdef _OPENMP
#pragma omp for
#endif
    // step 1: for each 2D output point
    for (x = 0; x < num_out; x++)
    {
        for (y = 0; y < num_out; y++)
        {
            D[x][y] = 0.f;
            // compute Euclidian distance of each output
            // point from the current sample
            for (k = 0; k < num_features; k++)
            {
                double *w = kohonen_data_3d(W, x, y, k);
                D[x][y] += (w[0] - X[k]) * (w[0] - X[k]);
            }
            D[x][y] = sqrt(D[x][y]);
        }
    }

    // step 2:  get closest node i.e., node with smallest Euclidian distance to
    // the current pattern
    int d_min_x, d_min_y;
    get_min_2d(D, num_out, &d_min, &d_min_x, &d_min_y);

    // step 3a: get the neighborhood range
    int from_x = max(0, d_min_x - R);
    int to_x = min(num_out, d_min_x + R + 1);
    int from_y = max(0, d_min_y - R);
    int to_y = min(num_out, d_min_y + R + 1);

    // step 3b: update the weights of nodes in the
    // neighborhood
#ifdef _OPENMP
#pragma omp for
#endif
    for (x = from_x; x < to_x; x++)
    {
        for (y = from_y; y < to_y; y++)
        {
            /* you can enable the following normalization if needed.
               personally, I found it detrimental to convergence */
            // const double s2pi = sqrt(2.f * M_PI);
            // double normalize = 1.f / (alpha * s2pi);

            /* apply scaling inversely proportional to distance from the
               current node */
            double d2 =
                (d_min_x - x) * (d_min_x - x) + (d_min_y - y) * (d_min_y - y);
            double scale_factor = exp(-d2 / (2.f * alpha * alpha));

            for (k = 0; k < num_features; k++)
            {
                double *w = kohonen_data_3d(W, x, y, k);
                // update weights of nodes in the neighborhood
                w[0] += alpha * scale_factor * (X[k] - w[0]);
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
    clock_t start_training = clock();  // Início do treinamento
    int R = num_out >> 2, iter = 0;
    double **D = (double **)malloc(num_out * sizeof(double *));
    for (int i = 0; i < num_out; i++)
        D[i] = (double *)malloc(num_out * sizeof(double));

    double dmin = 1.f;  // average minimum distance of all samples

    // Loop alpha from 1 to slpha_min
    for (double alpha = 1.f; alpha > alpha_min && dmin > 1e-3;
         alpha -= 0.001, iter++)
    {
        dmin = 0.f;
        // Loop for each sample pattern in the data set
        for (int sample = 0; sample < num_samples; sample++)
        {
            // update weights for the current input pattern sample
            dmin += kohonen_update_weights(X[sample], W, D, num_out,
                                           num_features, alpha, R);
        }

        // every 20th iteration, reduce the neighborhood range
        if (iter % 100 == 0 && R > 1)
            R--;

        dmin /= num_samples;
        printf("iter: %5d\t alpha: %.4g\t R: %d\td_min: %.4g\r", iter, alpha, R,
               dmin);
    }
    putchar('\n');

    clock_t end_training = clock();  // Fim do treinamento
    double training_time = get_clock_diff(start_training, end_training);
    printf("⏱️  Tempo de treinamento: %.2f segundos (%.2f minutos)\n", 
           training_time, training_time / 60.0);

    for (int i = 0; i < num_out; i++) free(D[i]);
    free(D);
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
    // Encontrar min e max para cada feature
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
    
    // Normalizar: (x - min) / (max - min)
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < num_samples; i++)
    {
        for (int j = 0; j < num_features; j++)
        {
            double range = max_vals[j] - min_vals[j];
            if (range > 1e-10)  // evitar divisão por zero
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
    clock_t start_total = clock();  // Início do processamento total
    int num_samples, num_features;
    int num_out = 30;  // tamanho do mapa SOM (30x30)
    
    printf("\n=== Processando dados do Banking Market ===\n");
    
    // Carregar dados
    clock_t start_load = clock();
    double **X = load_banking_data("banking_market/train.csv", &num_samples, &num_features);
    if (!X)
    {
        fprintf(stderr, "Erro ao carregar dados!\n");
        return;
    }
    clock_t end_load = clock();
    double load_time = get_clock_diff(start_load, end_load);
    printf("⏱️  Tempo de carregamento: %.2f segundos\n", load_time);
    
    // Normalizar dados
    printf("Normalizando dados...\n");
    clock_t start_norm = clock();
    normalize_data(X, num_samples, num_features);
    clock_t end_norm = clock();
    double norm_time = get_clock_diff(start_norm, end_norm);
    printf("⏱️  Tempo de normalização: %.2f segundos\n", norm_time);
    
    // Salvar dados normalizados
    save_2d_data("banking_data_normalized.csv", X, num_samples, num_features);
    printf("Dados normalizados salvos em: banking_data_normalized.csv\n");
    
    // Criar estrutura SOM
    struct kohonen_array_3d W;
    W.dim1 = num_out;
    W.dim2 = num_out;
    W.dim3 = num_features;
    W.data = (double *)malloc(num_out * num_out * num_features * sizeof(double));
    
    // Inicializar pesos aleatoriamente
    printf("Inicializando pesos do SOM...\n");
    clock_t start_init = clock();
    for (int i = 0; i < num_out; i++)
    {
        for (int k = 0; k < num_out; k++)
        {
            for (int j = 0; j < num_features; j++)
            {
                double *w = kohonen_data_3d(&W, i, k, j);
                w[0] = _random(0, 1);  // valores normalizados entre 0 e 1
            }
        }
    }
    clock_t end_init = clock();
    double init_time = get_clock_diff(start_init, end_init);
    printf("⏱️  Tempo de inicialização: %.2f segundos\n", init_time);
    
    // Salvar U-matrix inicial
    clock_t start_save1 = clock();
    save_u_matrix("banking_w_before.csv", &W);
    clock_t end_save1 = clock();
    double save1_time = get_clock_diff(start_save1, end_save1);
    printf("U-matrix inicial salva em: banking_w_before.csv\n");
    printf("⏱️  Tempo de salvamento (antes): %.2f segundos\n", save1_time);
    
    // Treinar SOM
    printf("Treinando SOM...\n");
    kohonen_som(X, &W, num_samples, num_features, num_out, 1e-4);
    
    // Salvar U-matrix treinada
    clock_t start_save2 = clock();
    save_u_matrix("banking_w_after.csv", &W);
    clock_t end_save2 = clock();
    double save2_time = get_clock_diff(start_save2, end_save2);
    printf("U-matrix treinada salva em: banking_w_after.csv\n");
    printf("⏱️  Tempo de salvamento (depois): %.2f segundos\n", save2_time);
    
    // Salvar pesos do SOM para análise posterior
    clock_t start_save3 = clock();
    save_som_weights("banking_weights.csv", &W);
    clock_t end_save3 = clock();
    double save3_time = get_clock_diff(start_save3, end_save3);
    printf("Pesos do SOM salvos em: banking_weights.csv\n");
    printf("⏱️  Tempo de salvamento (pesos): %.2f segundos\n", save3_time);
    
    // Limpar memória
    for (int i = 0; i < num_samples; i++)
        free(X[i]);
    free(X);
    free(W.data);
    
    // Tempo total
    clock_t end_total = clock();
    double total_time = get_clock_diff(start_total, end_total);
    
    // Calcular tempo de treinamento (subtrair outros tempos)
    double training_time = total_time - load_time - norm_time - init_time - save1_time - save2_time - save3_time;
    
    printf("\n");
    printf("============================================================\n");
    printf("📊 RESUMO DE TEMPOS DE EXECUÇÃO\n");
    printf("============================================================\n");
    printf("  Carregamento de dados:     %8.2f segundos\n", load_time);
    printf("  Normalização:              %8.2f segundos\n", norm_time);
    printf("  Inicialização de pesos:    %8.2f segundos\n", init_time);
    printf("  Salvamento (antes):        %8.2f segundos\n", save1_time);
    printf("  Treinamento do SOM:        %8.2f segundos (%.2f minutos)\n", 
           training_time, training_time / 60.0);
    printf("  Salvamento (depois):       %8.2f segundos\n", save2_time);
    printf("  Salvamento (pesos):        %8.2f segundos\n", save3_time);
    printf("  ------------------------------------------------------------\n");
    printf("  TEMPO TOTAL:               %8.2f segundos (%.2f minutos)\n", 
           total_time, total_time / 60.0);
    printf("============================================================\n");
    printf("=== Processamento concluído! ===\n\n");
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

/** Main function */
int main(int argc, char **argv)
{
#ifdef _OPENMP
    printf("Using OpenMP based parallelization\n");
#else
    printf("NOT using OpenMP based parallelization\n");
#endif
    
    // Se argumento "banking" for passado, processa dados do banking
    if (argc > 1 && strcmp(argv[1], "banking") == 0)
    {
        clock_t start_clk = clock();
        test_banking();
        clock_t end_clk = clock();
        printf("Banking test completed in %.4g sec\n",
               get_clock_diff(start_clk, end_clk));
        return 0;
    }
    
    // Caso contrário, executa testes originais
    clock_t start_clk, end_clk;

    start_clk = clock();
    test1();
    end_clk = clock();
    printf("Test 1 completed in %.4g sec\n",
           get_clock_diff(start_clk, end_clk));

    start_clk = clock();
    test2();
    end_clk = clock();
    printf("Test 2 completed in %.4g sec\n",
           get_clock_diff(start_clk, end_clk));

    start_clk = clock();
    test3();
    end_clk = clock();
    printf("Test 3 completed in %.4g sec\n",
           get_clock_diff(start_clk, end_clk));

    printf("(Note: Calculated times include: writing files to disk.)\n");
    printf("\nPara processar dados do banking, execute: ./som_topology banking\n\n");
    return 0;
}

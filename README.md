# 🧠 Self-Organizing Map (SOM) - Análise de Clientes Banking

Projeto de **Computação Paralela** que implementa um **Self-Organizing Map (SOM)** de Kohonen para análise de agrupamentos de clientes bancários e identificação de padrões de comportamento relacionados a empréstimos.

## 📋 Sobre o Projeto

Este projeto utiliza algoritmos de aprendizado não-supervisionado para:
- **Agrupar clientes** com características similares
- **Visualizar padrões** em dados de alta dimensionalidade
- **Identificar clusters** de clientes propensos a empréstimos
- **Mapear relações topológicas** entre diferentes perfis de clientes

### O que é um SOM?

O **Self-Organizing Map (SOM)** é uma rede neural artificial que:
- Cria um **mapa topológico 2D** dos dados
- Preserva **relações espaciais** dos dados originais
- Agrupa dados similares em **regiões próximas** do mapa
- Permite **visualização** de dados de alta dimensionalidade

## 🗂️ Estrutura do Projeto

```
trab-paralela/
├── kohonen_som_topology.c               # Implementação sequencial do SOM
├── kohonen_som_topology_paralel_openMP.c # Implementação OpenMP (multicore)
├── kohonen_som_trace.c                  # Implementação alternativa (cadeia 1D)
├── analyze_banking_clusters.py          # Script Python para análise e visualização
├── banking_market/
│   ├── train.csv                        # Dados de treinamento (45.213 clientes)
│   └── test.csv                         # Dados de teste (4.523 clientes)
├── banking_w_before.csv                 # U-Matrix antes do treinamento
├── banking_w_after.csv                  # U-Matrix depois do treinamento ⭐
├── banking_data_normalized.csv          # Dados normalizados
└── README.md                            # Este arquivo
```

## 🚀 Como Executar

### Pré-requisitos

- **Compilador C** com suporte a OpenMP (GCC, Clang, ou MinGW)
- **Python 3** com bibliotecas: `numpy`, `matplotlib`, `pandas`, `scipy` (opcional)

---

## ⚡ VERSÃO PARALELA OPENMP (CPU MULTICORE)

### 1. Compilar a Versão Paralela

```bash
gcc -fopenmp -O3 -march=native kohonen_som_topology_paralel_openMP.c -o som_parallel -lm
```

### 2. Executar com N threads

```bash
./som_parallel banking 8        # Usar 8 threads
./som_parallel banking 4        # Usar 4 threads
```

### 3. Resultados de Performance

**Dataset:** 45.211 amostras, 1000 iterações, SOM 30x30

| Configuração | Tempo (segundos) | Speedup | Eficiência |
|-------------|------------------|---------|------------|
| **Sequencial (sem OpenMP)** | 2351.51 seg | - | - |
| **1 thread (com -O3)** | 489.67 seg | 1.00x | 100.0% |
| **2 threads** | 266.60 seg | 1.84x | 91.9% |
| **4 threads** | 170.19 seg | 2.88x | 71.9% |
| **8 threads** | 129.47 seg | **3.78x** | 47.3% |

**Melhor configuração:** 8 threads com speedup de **3.78x**

**Ganho total:** **18.16x** mais rápido que a versão sequencial (sem otimizações)

### 4. Análise de Performance

-  **Speedup quase linear até 2 threads** (eficiência > 90%)
-  **Boa escalabilidade até 4 threads** (eficiência ~72%)
-  **Speedup continua em 8 threads, mas eficiência cai** devido a:
  - Race conditions protegidas com `#pragma omp atomic`
  - Overhead de sincronização entre threads
  - Contenção de memória e cache

### 5. Mudanças de Paralelização Implementadas

1. **Medição precisa de tempo** com `omp_get_wtime()`
2. **Loop principal paralelizado** em `kohonen_som()`:
   - Cada thread processa subconjunto de amostras
   - Cada thread tem matriz D local (evita contenção)
   - `schedule(dynamic, 100)` para balanceamento de carga
3. **Normalização paralelizada** com redução manual (min/max)
4. **U-matrix paralelizada** com `reduction(+:distance)`
5. **Atualização de pesos protegida** com `#pragma omp atomic`
6. **Função de benchmark** que testa 1-32 threads automaticamente
7. **Prints detalhados** mostrando threads ativas em tempo real

---

## 📘 VERSÃO SEQUENCIAL (Original)

### 1. Compilar o Código Sequencial

#### Linux/Mac:
```bash
gcc kohonen_som_topology.c -o som_topology -lm
```

#### Windows (MinGW):
```bash
gcc kohonen_som_topology.c -o som_topology.exe -lm
```

#### Windows (MSVC):
```bash
cl kohonen_som_topology.c
```

### 2. Executar o Treinamento

#### Com dados do Banking Market:
```bash
./som_topology banking
```

#### Testes originais (dados sintéticos):
```bash
./som_topology
```

### 3. O que acontece durante a execução?

O programa irá:
1. ✅ Carregar dados do arquivo `banking_market/train.csv`
2. ✅ Processar e normalizar os dados (8 features selecionadas)
3. ✅ Inicializar pesos aleatórios do SOM
4. ✅ Treinar o SOM (~800-1000 iterações)
5. ✅ Salvar resultados em arquivos CSV

**Tempo estimado:** 5-15 minutos (com OpenMP) ou 15-45 minutos (sem OpenMP)

### 4. Saída do Console

Durante o treinamento, você verá:
```
=== Processando dados do Banking Market ===
Carregados 45211 amostras com 8 features
Features: age, balance, duration, campaign, previous, default, housing, loan
Normalizando dados...
Inicializando pesos do SOM...
Treinando SOM...
iter:     0  alpha: 1      R: 7  d_min: 2.345
iter:   100  alpha: 0.9   R: 6  d_min: 1.234
iter:   200  alpha: 0.8   R: 5  d_min: 0.567
...
U-matrix treinada salva em: banking_w_after.csv
Pesos do SOM salvos em: banking_weights.csv
```

## 📊 Arquivos Gerados

Após a execução, os seguintes arquivos serão criados:

### Arquivos Principais:

| Arquivo | Descrição |
|---------|-----------|
| `banking_w_before.csv` | U-Matrix inicial (pesos aleatórios) |
| `banking_w_after.csv` | **U-Matrix final (clusters organizados)** ⭐ |
| `banking_data_normalized.csv` | Dados normalizados (8 features) |
| `banking_weights.csv` | Pesos do SOM (para mapeamento de clientes) |

### O que é a U-Matrix?

A **U-Matrix** (Unified Distance Matrix) é um mapa 30x30 que mostra:
- **Valores baixos (áreas escuras)**: Clusters de clientes similares
- **Valores altos (áreas claras)**: Fronteiras entre grupos diferentes

## 📈 Visualizar Resultados

### Opção 1: Script Python (Recomendado)

```bash
python analyze_banking_clusters.py
```

Este script irá:
- ✅ Comparar antes vs depois do treinamento
- ✅ Visualizar U-Matrix como mapa de calor
- ✅ Identificar clusters automaticamente
- ✅ Gerar gráficos PNG para análise

**Arquivos gerados:**
- `banking_comparison.png` - Comparação visual
- `banking_u_matrix_final.png` - Mapa de clusters
- `banking_clusters_identified.png` - Clusters destacados

### Opção 2: Visualização Manual

```python
import numpy as np
import matplotlib.pyplot as plt

# Carregar U-matrix
u_matrix = np.loadtxt('banking_w_after.csv', delimiter=',')

# Visualizar
plt.figure(figsize=(12, 10))
plt.imshow(u_matrix, cmap='hot', interpolation='nearest', origin='lower')
plt.colorbar(label='Distância Média')
plt.title('U-Matrix - Agrupamentos de Clientes')
plt.xlabel('Índice X do Neurônio')
plt.ylabel('Índice Y do Neurônio')
plt.savefig('u_matrix.png', dpi=300)
plt.show()
```

## 🔍 Interpretação dos Resultados

### Analisando `banking_w_after.csv`:

#### Áreas Escuras (Valores < 1.0) = Clusters
```
Exemplo: Linha 1, Colunas 1-15
Valores: 0.379 - 1.555
```
- ✅ Grupos de clientes com características similares
- ✅ Clientes agrupados por perfil similar

#### Áreas Claras (Valores > 3.0) = Fronteiras
```
Exemplo: Linha 10, Coluna 22
Valor: 4.387
```
- ⚠️ Separação entre grupos distintos
- ⚠️ Mudança brusca de características

### Features Utilizadas:

O SOM foi treinado com 8 features:
1. **age** - Idade do cliente
2. **balance** - Saldo da conta
3. **duration** - Duração da última chamada
4. **campaign** - Número de contatos na campanha
5. **previous** - Número de contatos anteriores
6. **default** - Tem crédito em default? (0=no, 1=yes)
7. **housing** - Tem empréstimo habitacional? (0=no, 1=yes)
8. **loan** - Tem empréstimo pessoal? (0=no, 1=yes)

## 🎯 Próximos Passos

### Para identificar clientes propensos a empréstimos:

1. **Mapear clientes aos clusters**
   - Para cada cliente, encontrar seu neurônio vencedor (BMU)
   - Identificar em qual cluster ele está

2. **Calcular taxa de aceitação por cluster**
   - Usar a coluna "y" do arquivo original
   - Calcular: `aceitos / total` por cluster

3. **Identificar clusters de alto valor**
   - Clusters com alta taxa = clientes propensos
   - Analisar características desses clusters

## ⚙️ Configurações

### Ajustar tamanho do mapa SOM:

No arquivo `kohonen_som_topology.c`, função `test_banking()`:
```c
int num_out = 30;  // Mude para 20, 40, 50, etc.
```

### Ajustar número de iterações:

Na função `kohonen_som()`:
```c
alpha -= 0.001  // Mais iterações (mais lento, mais preciso)
alpha -= 0.01   // Menos iterações (mais rápido)
```

## 🐛 Troubleshooting

### Erro: "Erro ao abrir arquivo"
- Verifique se `banking_market/train.csv` existe
- Verifique o caminho relativo

### Erro: "Carregados 0 amostras"
- Verifique o formato do CSV (deve usar `;` como separador)
- Verifique se o cabeçalho está correto

### Erro ao importar matplotlib
```bash
pip install matplotlib numpy pandas
```

### Compilação sem OpenMP
Se OpenMP não estiver disponível, o código ainda funciona (mais lento):
```bash
gcc kohonen_som_topology.c -o som_topology -lm
```

## 📚 Referências

- [Self-Organizing Map - Wikipedia](https://en.wikipedia.org/wiki/Self-organizing_map)
- [U-Matrix - Wikipedia](https://en.wikipedia.org/wiki/U-matrix)
- [Kohonen Networks](https://en.wikipedia.org/wiki/Kohonen_network)

## 👥 Autores

Projeto desenvolvido para a disciplina de **Computação Paralela**.

## 📝 Licença

Este projeto é para fins educacionais.

---

## 🚀 Quick Start

### Versão Paralela (Recomendada - 18x mais rápida!)

```bash
# 1. Compilar com otimizações
gcc -fopenmp -O3 -march=native kohonen_som_topology_paralel_openMP.c -o som_parallel -lm

# 2. Executar com 8 threads
./som_parallel banking 8

# 4. Visualizar resultados
python analyze_banking_clusters.py
```

### Versão Sequencial

```bash
# 1. Compilar
gcc kohonen_som_topology.c -o som_topology -lm

# 2. Executar
./som_topology banking

# 3. Visualizar
python analyze_banking_clusters.py
```

**Pronto!** Os resultados estarão nos arquivos CSV e PNG gerados.

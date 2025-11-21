"""
Script para analisar os clusters do SOM e identificar
quais grupos de clientes tendem mais a pegar empréstimos
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def load_u_matrix(filename):
    """Carrega U-matrix do arquivo CSV"""
    if not os.path.exists(filename):
        print(f"Arquivo {filename} não encontrado!")
        return None
    
    data = []
    with open(filename, 'r') as f:
        for line in f:
            row = [float(x) for x in line.strip().split(',')]
            data.append(row)
    return np.array(data)

def visualize_u_matrix(u_matrix, title="U-Matrix", save_name=None):
    """Visualiza U-matrix como mapa de calor"""
    plt.figure(figsize=(14, 12))
    im = plt.imshow(u_matrix, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar(im, label='Distância Média', fraction=0.046)
    plt.xlabel('Índice X do Neurônio', fontsize=12)
    plt.ylabel('Índice Y do Neurônio', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico salvo em: {save_name}")
    
    return plt.gcf()

def load_banking_data_with_target(csv_file):
    """Carrega dados do banking incluindo a coluna target 'y'"""
    print(f"Carregando dados de {csv_file}...")
    
    # Ler CSV com ponto e vírgula
    df = pd.read_csv(csv_file, sep=';', quotechar='"')
    
    # Extrair target (coluna 'y')
    target = df['y'].map({'yes': 1, 'no': 0}).values
    
    # Carregar dados normalizados (já processados pelo C)
    normalized_file = 'banking_data_normalized.csv'
    if os.path.exists(normalized_file):
        data = np.loadtxt(normalized_file, delimiter=',')
        print(f"✅ Carregados {len(data)} amostras com {data.shape[1]} features")
        return data, target
    else:
        print(f"⚠️ Arquivo {normalized_file} não encontrado!")
        return None, None

def find_bmu(data_point, som_weights):
    """
    Encontra o Best Matching Unit (BMU) para um ponto de dados
    som_weights: array 3D (x, y, features)
    """
    min_dist = float('inf')
    bmu_x, bmu_y = 0, 0
    
    for x in range(som_weights.shape[0]):
        for y in range(som_weights.shape[1]):
            # Calcular distância euclidiana
            dist = np.sqrt(np.sum((data_point - som_weights[x, y, :]) ** 2))
            if dist < min_dist:
                min_dist = dist
                bmu_x, bmu_y = x, y
    
    return bmu_x, bmu_y, min_dist

def load_som_weights():
    """Carrega os pesos do SOM do arquivo de dados normalizados e U-matrix"""
    # Os pesos do SOM não estão salvos diretamente, mas podemos usar a U-matrix
    # Para análise completa, precisaríamos dos pesos, mas vamos trabalhar com o que temos
    print("⚠️ Nota: Para mapeamento completo, seria necessário salvar os pesos do SOM.")
    print("   Vamos analisar a estrutura dos clusters pela U-matrix.")
    return None

def analyze_clusters_from_u_matrix(u_matrix, data, target):
    """
    Analisa clusters baseado na U-matrix
    Identifica áreas de baixa distância (clusters) e mapeia clientes
    """
    print("\n" + "="*60)
    print("ANÁLISE DE CLUSTERS")
    print("="*60)
    
    # Identificar clusters (áreas com valores baixos na U-matrix)
    threshold = np.percentile(u_matrix, 30)  # 30% mais baixos = clusters
    
    # Criar máscara de clusters
    cluster_mask = u_matrix < threshold
    
    # Contar clusters
    from scipy import ndimage
    labeled, num_clusters = ndimage.label(cluster_mask)
    
    print(f"\n📊 Número de clusters identificados: {num_clusters}")
    print(f"   Threshold usado: {threshold:.4f}")
    print(f"   (Valores abaixo deste são considerados clusters)")
    
    # Visualizar clusters identificados
    plt.figure(figsize=(14, 12))
    plt.imshow(u_matrix, cmap='hot', interpolation='nearest', origin='lower')
    plt.contour(labeled, levels=range(1, num_clusters+1), colors='cyan', linewidths=2)
    plt.colorbar(label='Distância Média')
    plt.xlabel('Índice X do Neurônio', fontsize=12)
    plt.ylabel('Índice Y do Neurônio', fontsize=12)
    plt.title('U-Matrix com Clusters Identificados (linhas ciano)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('banking_clusters_identified.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Gráfico de clusters salvo em: banking_clusters_identified.png")
    
    return labeled, num_clusters

def compare_before_after():
    """Compara U-matrix antes e depois do treinamento"""
    print("\n" + "="*60)
    print("COMPARAÇÃO: ANTES vs DEPOIS DO TREINAMENTO")
    print("="*60)
    
    u_before = load_u_matrix('banking_w_before.csv')
    u_after = load_u_matrix('banking_w_after.csv')
    
    if u_before is None or u_after is None:
        print("⚠️ Arquivos não encontrados!")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Antes
    im1 = axes[0].imshow(u_before, cmap='hot', interpolation='nearest', origin='lower')
    axes[0].set_title('ANTES do Treinamento\n(Pesos Aleatórios)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Índice X')
    axes[0].set_ylabel('Índice Y')
    plt.colorbar(im1, ax=axes[0], label='Distância')
    
    # Depois
    im2 = axes[1].imshow(u_after, cmap='hot', interpolation='nearest', origin='lower')
    axes[1].set_title('DEPOIS do Treinamento\n(Clusters Organizados)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Índice X')
    axes[1].set_ylabel('Índice Y')
    plt.colorbar(im2, ax=axes[1], label='Distância')
    
    plt.tight_layout()
    plt.savefig('banking_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Comparação salva em: banking_comparison.png")
    
    # Estatísticas
    print(f"\n📊 Estatísticas:")
    print(f"   Antes - Média: {np.mean(u_before):.4f}, Std: {np.std(u_before):.4f}")
    print(f"   Depois - Média: {np.mean(u_after):.4f}, Std: {np.std(u_after):.4f}")
    print(f"   Redução na variância: {((np.std(u_before) - np.std(u_after)) / np.std(u_before) * 100):.1f}%")

def main():
    """Função principal"""
    print("="*60)
    print("ANÁLISE DE CLUSTERS - BANKING MARKET")
    print("="*60)
    
    # 1. Comparar antes e depois
    compare_before_after()
    
    # 2. Visualizar U-matrix final
    u_after = load_u_matrix('banking_w_after.csv')
    if u_after is not None:
        visualize_u_matrix(u_after, 
                          title="U-Matrix Final - Agrupamentos de Clientes",
                          save_name='banking_u_matrix_final.png')
    
    # 3. Tentar carregar dados para análise mais profunda
    data, target = load_banking_data_with_target('banking_market/train.csv')
    
    if data is not None and target is not None:
        # Analisar clusters
        try:
            labeled, num_clusters = analyze_clusters_from_u_matrix(u_after, data, target)
            print(f"\n✅ Análise completa!")
        except ImportError:
            print("\n⚠️ scipy não encontrado. Instale com: pip install scipy")
            print("   Continuando com análise básica...")
    
    # 4. Interpretação
    print("\n" + "="*60)
    print("📋 INTERPRETAÇÃO DOS RESULTADOS")
    print("="*60)
    print("""
    A U-Matrix mostra:
    
    🔴 ÁREAS ESCURAS (valores baixos):
       → Clusters de clientes com características similares
       → Neurônios próximos = dados similares agrupados
    
    🟡 ÁREAS CLARAS (valores altos):
       → Fronteiras entre clusters diferentes
       → Separação entre grupos distintos de clientes
    
    📊 PRÓXIMOS PASSOS:
       1. Identifique os clusters escuros no mapa
       2. Para cada cluster, analise as características dos clientes
       3. Compare com a taxa de aceitação de empréstimos (coluna 'y')
       4. Clusters com alta taxa = clientes propensos a empréstimos
    
    💡 DICA:
       Use os gráficos gerados para identificar visualmente
       quais regiões do mapa correspondem a grupos específicos
       de clientes.
    """)
    
    print("\n✅ Análise concluída! Verifique os arquivos PNG gerados.")
    plt.show()

if __name__ == "__main__":
    main()


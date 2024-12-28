# !pip install sentence-transformers # run in Kaggle with GPU
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import sys
import datasets
from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances

dataset_path = "/kaggle/input/alpaca-language-instruction-training/train.csv"
alpaca_dataset = load_dataset('csv', data_files = dataset_path)
# print(dataset.keys()) # 只包含'train'
alpaca_data = [
    f"Instruction: {row['instruction']} Input: {row.get('input', '')} Output: {row.get('output', '')}"
    for row in alpaca_dataset['train']
]

eval_dataset1_path = "/kaggle/input/d/lizhecheng/mmlu-dataset/data/test/college_physics_test.csv"
eval_dataset2_path = "/kaggle/input/d/lizhecheng/mmlu-dataset/data/test/abstract_algebra_test.csv"
eval_dataset3_path = "/kaggle/input/d/lizhecheng/mmlu-dataset/data/test/econometrics_test.csv"

mmlu_dataset1 = load_dataset('csv', data_files = eval_dataset1_path)
mmlu_data1 = mmlu_dataset1['train'].map(lambda example: {'concatenated': ' '.join([f"{key}: {str(value)}" for key, value in example.items()])})
mmlu_data1_list = mmlu_data1['concatenated']
# print(mmlu_dataset1['train'][0])

mmlu_dataset2 = load_dataset('csv', data_files = eval_dataset2_path)
mmlu_data2 = mmlu_dataset2['train'].map(lambda example: {'concatenated': ' '.join([f"{key}: {str(value)}" for key, value in example.items()])})
mmlu_data2_list = mmlu_data2['concatenated']

mmlu_dataset3 = load_dataset('csv', data_files = eval_dataset3_path)
mmlu_data3 = mmlu_dataset3['train'].map(lambda example: {'concatenated': ' '.join([f"{key}: {str(value)}" for key, value in example.items()])})
mmlu_data3_list = mmlu_data3['concatenated']

# 1. 加载 SentenceTransformer 模型, all-distilroberta-v1
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

alpaca_embeddings = model.encode(alpaca_data)

mmlu1_emdedding = model.encode(mmlu_data1_list)
eval_data_center1 = np.mean(mmlu1_emdedding, axis=0)

mmlu2_emdedding = model.encode(mmlu_data2_list)
eval_data_center2 = np.mean(mmlu2_emdedding, axis=0)

mmlu3_emdedding = model.encode(mmlu_data3_list)
eval_data_center3 = np.mean(mmlu3_emdedding, axis=0)

# # 检查 alpaca_embeddings 的形状
# print(alpaca_embeddings.shape)

n_clusters = 5 
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(alpaca_embeddings)

# 计算 kmeans 聚类中心
kmeans_centers = kmeans.cluster_centers_

# 定义评测数据集中心
eval_data_centers = [eval_data_center1, eval_data_center2, eval_data_center3]

# 计算每个评测数据中心到所有 kmeans 聚类中心的距离
for i, eval_center in enumerate(eval_data_centers):
    # 计算与所有kmeans中心的欧几里得距离
    distances = euclidean_distances([eval_center], kmeans_centers)
    
    # 找出距离最小的kmeans中心
    closest_center_idx = np.argmin(distances)
    closest_center = kmeans_centers[closest_center_idx]
    closest_distance = distances[0][closest_center_idx]
    print(f"Distances for Eval Data Center {i+1}:")
    
    # 输出每个eval_center到所有kmeans中心的距离
    for j, dist in enumerate(distances[0]):
        print(f"  To KMeans Center {j+1}: {dist:.4f}")
    
    print(f"Eval Data Center {i+1} is closest to KMeans Center {closest_center_idx+1} with distance: {closest_distance:.4f}")
    print()

# # 获取每个句子的聚类标签
alpaca_labels = kmeans.labels_
# # # print(alpaca_labels)

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=42)
# reduced_embeddings = tsne.fit_transform(alpaca_embeddings)

# 评测数据集的聚类中心
all_centers = np.vstack([eval_data_center1, eval_data_center2, eval_data_center3])
all_data = np.vstack([alpaca_embeddings, all_centers])

reduced_all_data = tsne.fit_transform(all_data)

# 将降维后的数据重新拆分
reduced_embeddings = reduced_all_data[:len(alpaca_embeddings)]
reduced_centers = reduced_all_data[len(alpaca_embeddings):]

# 可视化所有数据点和评测数据集的聚类中心
plt.figure(figsize=(10, 8))

# 绘制 Alpaca 数据点，并使用颜色表示聚类标签
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=alpaca_labels, cmap='plasma', alpha=0.7)
plt.colorbar()  # 显示颜色条，表示每个类别的标签

# 绘制评测数据集的聚类中心，使用不同颜色和标记
for i, center in enumerate(reduced_centers):
    # 对center1、center2、center3使用不同的颜色和加粗
    if i == 0:  # center1
        color = 'red'
        linewidth = 2
    elif i == 1:  # center2
        color = 'black'
        linewidth = 2
    elif i == 2:  # center3
        color = 'green'
        linewidth = 2
    # 绘制中心点，使用 'x' 标记
    plt.scatter(center[0], center[1], marker='x', s=200, color=color, linewidths=linewidth)
    plt.text(center[0], center[1], f'Center {i+1}', fontsize=12, ha='right', color=color)

# 添加标签和标题
plt.title('t-SNE Clustering of Alpaca Dateset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.show()

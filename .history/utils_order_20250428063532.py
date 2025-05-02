import heapq

def _top_k_nodes_by_free(cluster_state, k):
    """返回空闲 GPU 数最多的 k 个节点下标（按剩余 GPU 降序）"""
    # (-free, idx) 做最大堆
    h = [(-free, idx) for idx, free in enumerate(cluster_state)]
    heapq.heapify(h)
    return [heapq.heappop(h)[1] for _ in range(min(k, len(h)))]

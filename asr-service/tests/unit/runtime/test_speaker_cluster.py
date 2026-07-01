"""app/runtime/speaker_cluster.py 单元测试（合成 embedding，不触模型）。

在线：归簇分支（建簇/挂靠/短段/上限）；离线：AHC/谱聚类分支、阈值语义（spike 勘误锁定）、
小簇并入、近簇合并、首现重排。只测逻辑，不测真实音频精度（实施方案 §4 铁律）。
"""
import numpy as np
import pytest

from app.runtime.speaker_cluster import (
    CLUSTER_LINE,
    OnlineSpeakerClusterer,
    _merge_close_clusters,
    cluster_offline,
    speaker_label,
)

RNG = np.random.default_rng(42)
DIM = 192


def unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def around(center: np.ndarray, scale: float = 0.01) -> np.ndarray:
    v = center + RNG.normal(scale=scale, size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def mix(a: np.ndarray, b: np.ndarray, wa: float, wb: float) -> np.ndarray:
    v = wa * a + wb * b
    return v / np.linalg.norm(v)


# ─── speaker_label ───

def test_speaker_label_alphabet_and_overflow():
    assert speaker_label(0) == "A"
    assert speaker_label(25) == "Z"
    assert speaker_label(26) == "Z1"
    assert speaker_label(27) == "Z2"


# ─── OnlineSpeakerClusterer ───

def test_online_first_segment_creates_a():
    oc = OnlineSpeakerClusterer(threshold=0.5)
    assert oc.assign(unit(0), 3000) == "A"
    assert len(oc.centroids) == 1


def test_online_similar_joins_and_updates_centroid():
    oc = OnlineSpeakerClusterer(threshold=0.5)
    oc.assign(unit(0), 3000)
    before = oc.centroid_of("A").copy()
    assert oc.assign(around(unit(0)), 3000) == "A"
    assert len(oc.centroids) == 1
    assert not np.allclose(oc.centroid_of("A"), before)  # 质心已计数加权更新
    assert np.isclose(np.linalg.norm(oc.centroid_of("A")), 1.0, atol=1e-5)


def test_online_dissimilar_creates_new_cluster():
    oc = OnlineSpeakerClusterer(threshold=0.5)
    oc.assign(unit(0), 3000)
    assert oc.assign(unit(1), 3000) == "B"
    assert len(oc.centroids) == 2


def test_online_short_segment_attaches_without_update():
    oc = OnlineSpeakerClusterer(threshold=0.5, min_seg_ms=1500)
    assert oc.assign(unit(0), 500) is None  # 无簇可挂靠 → None，且不建簇
    assert len(oc.centroids) == 0
    oc.assign(unit(0), 3000)
    before = oc.centroid_of("A").copy()
    assert oc.assign(around(unit(0)), 500) == "A"  # 挂靠成功
    assert np.allclose(oc.centroid_of("A"), before)  # 质心不更新
    assert oc.assign(unit(1), 500) is None  # 相似度不够 → 不建簇返回 None
    assert len(oc.centroids) == 1


def test_online_max_speakers_assigns_nearest():
    oc = OnlineSpeakerClusterer(threshold=0.5, max_speakers=2)
    oc.assign(unit(0), 3000)
    oc.assign(unit(1), 3000)
    label = oc.assign(mix(unit(0), unit(2), 0.4, 0.9), 3000)  # 与 A 最近但低于 τ
    assert label == "A"
    assert len(oc.centroids) == 2  # 不再建簇


def test_online_centroid_of_unknown_label():
    oc = OnlineSpeakerClusterer()
    oc.assign(unit(0), 3000)
    assert oc.centroid_of("B") is None
    assert oc.centroid_of("?") is None


# ─── cluster_offline：分支与边界 ───

def test_offline_empty_and_single():
    assert cluster_offline(np.zeros((0, DIM))).shape == (0,)
    assert cluster_offline(unit(0)[np.newaxis]).tolist() == [0]


def test_offline_ahc_three_clusters_first_appearance_order():
    embs = np.stack(
        [around(unit(0)) for _ in range(6)]
        + [around(unit(1)) for _ in range(6)]
        + [around(unit(2)) for _ in range(5)]
    )
    assert len(embs) < CLUSTER_LINE  # AHC 分支
    labels = cluster_offline(embs)
    assert labels.tolist() == [0] * 6 + [1] * 6 + [2] * 5  # 首现重排：A=先开口


def test_offline_interleaved_speakers():
    embs = np.stack([around(unit(i % 2)) for i in range(12)])
    labels = cluster_offline(embs)
    assert labels.tolist() == [i % 2 for i in range(12)]


def test_offline_ahc_threshold_semantics():
    # spike 勘误锁定：相似度 ≥0.40 并簇，<0.40 分簇（非"距离 0.4=相似度 0.6"）
    close = np.stack([around(unit(0)) for _ in range(6)]
                     + [around(mix(unit(0), unit(1), 1.0, 1.0)) for _ in range(6)])
    assert len(np.unique(cluster_offline(close))) == 1  # 簇间相似度 ≈0.71 → 并

    far = np.stack([around(unit(0)) for _ in range(6)]
                   + [around(mix(unit(0), unit(1), 0.2, 0.98)) for _ in range(6)])
    assert len(np.unique(cluster_offline(far))) == 2  # 簇间相似度 ≈0.2 → 分


def test_offline_minor_cluster_absorbed():
    embs = np.stack(
        [around(unit(0)) for _ in range(10)]
        + [around(unit(1)) for _ in range(10)]
        + [around(unit(2)) for _ in range(3)]  # 小簇（≤4 窗）
    )
    labels = cluster_offline(embs)
    assert len(np.unique(labels)) == 2
    assert set(labels[20:]) <= {0, 1}  # 小簇成员并入大簇


def test_offline_spectral_branch_two_clusters():
    embs = np.stack(
        [around(unit(0)) for _ in range(30)] + [around(unit(1)) for _ in range(30)]
    )
    assert len(embs) >= CLUSTER_LINE  # 谱聚类分支
    labels = cluster_offline(embs)
    assert labels.tolist() == [0] * 30 + [1] * 30


def test_offline_spectral_single_speaker():
    embs = np.stack([around(unit(0)) for _ in range(45)])
    labels = cluster_offline(embs)
    assert labels.tolist() == [0] * 45


def test_offline_spectral_arpack_failure_degrades_single(monkeypatch):
    """ARPACK 不收敛（病态拉普拉斯）→ 降级单说话人，不向上抛异常吞掉整文件标签。"""
    import scipy.sparse.linalg as sla

    def boom(*args, **kwargs):
        raise sla.ArpackNoConvergence("ARPACK 未收敛", np.array([]), np.array([]))

    monkeypatch.setattr(sla, "eigsh", boom)
    embs = np.stack([around(unit(0)) for _ in range(45)])
    labels = cluster_offline(embs)
    assert labels.tolist() == [0] * 45


def test_offline_spectral_pruning_neighbors_capped_and_scale_invariant(monkeypatch):
    """p-pruning 保留邻居数封顶且不随 N 增长——超长多人音频塌成单人的根因回归。

    未封顶时保留邻居数 = n·pval 随 N 线性增长，把不同说话人的窗过度连通，抬高拉普拉斯
    第二特征值使 eigen-gap 误判为单说话人（实测 45min 多人访谈整段 N=2781 塌成 1 人，
    截取 24min 段 N=1541 正常分出多人）。此处拦截 eigsh，验证传入拉普拉斯每行非对角
    非零数（对称化后保留邻居的并集）被 SPECTRAL_MAX_PNUM 封顶、且 N 翻倍不再增长。
    """
    import scipy.sparse.linalg as sla

    from app.runtime.speaker_cluster import SPECTRAL_MAX_PNUM, _spectral

    orig = sla.eigsh

    def measure(n, max_pnum):
        # 独立种子生成器：不依赖模块级共享 RNG，结果与测试执行顺序无关
        rng = np.random.default_rng(n)
        embs = rng.normal(size=(n, DIM)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        seen = {}

        def spy(lap, *a, **k):
            off = np.array(lap)
            np.fill_diagonal(off, 0.0)
            seen["max_nnz"] = int((np.abs(off) > 1e-12).sum(axis=1).max())
            return orig(lap, *a, **k)

        monkeypatch.setattr(sla, "eigsh", spy)
        _spectral(embs, max_spks=8, max_pnum=max_pnum)
        return seen["max_nnz"]

    # 封顶后每行邻居数被硬约束（对称化最多翻倍），N 翻倍不放大
    capped_2k = measure(2000, SPECTRAL_MAX_PNUM)
    capped_4k = measure(4000, SPECTRAL_MAX_PNUM)
    assert capped_4k <= 2 * SPECTRAL_MAX_PNUM
    assert capped_4k <= capped_2k + SPECTRAL_MAX_PNUM   # 不随 N 线性增长

    # 反证 bug：不封顶时每行邻居数随 N 显著增长（≈翻倍）
    uncapped_2k = measure(2000, 10**9)
    uncapped_4k = measure(4000, 10**9)
    assert uncapped_4k > uncapped_2k + 10
    assert capped_4k < uncapped_4k                      # 封顶确实收紧了 N=4000 的连通度


# ─── 后处理 ───

def test_merge_close_clusters_by_centroid_similarity():
    a = [around(unit(0)) for _ in range(5)]
    b = [around(mix(unit(0), unit(1), 0.97, 0.24)) for _ in range(5)]  # 质心相似度 ≈0.97
    labels = _merge_close_clusters(np.array([0] * 5 + [1] * 5), np.stack(a + b))
    assert len(np.unique(labels)) == 1

    c = [around(unit(1)) for _ in range(5)]  # 质心相似度 ≈0
    labels2 = _merge_close_clusters(np.array([0] * 5 + [1] * 5), np.stack(a + c))
    assert len(np.unique(labels2)) == 2

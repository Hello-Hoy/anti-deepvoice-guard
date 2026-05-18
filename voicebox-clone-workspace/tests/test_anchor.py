import numpy as np

from jhm_extract.anchor import nearest_distinct_files, pick_target_cluster


def test_pick_cluster_by_file_coverage_not_window_count():
    # 클러스터1이 윈도우는 3개로 최다지만 파일은 1개. 클러스터0이 파일 2개 → 0 선택.
    labels = np.array([0, 0, 1, 1, 1, 2])
    file_ids = np.array([0, 1, 2, 2, 2, 3])
    assert pick_target_cluster(labels, file_ids) == 0


def test_pick_cluster_tiebreak_by_window_count():
    # 0과 1 모두 파일 2개 커버 → 윈도우 더 많은 1 선택
    labels = np.array([0, 0, 1, 1, 1])
    file_ids = np.array([0, 1, 0, 1, 1])
    assert pick_target_cluster(labels, file_ids) == 1


def test_nearest_distinct_files_picks_one_per_file():
    centroid = np.array([1.0, 0.0], dtype=np.float32)
    embs = np.array(
        [[0.99, 0.14], [0.95, 0.31], [0.20, 0.98], [0.90, 0.44]],
        dtype=np.float32,
    )
    file_ids = np.array([10, 10, 11, 12])
    picks = nearest_distinct_files(embs, file_ids, centroid, n=2)
    assert len(picks) == 2
    assert len({fid for fid, _ in picks}) == 2  # 서로 다른 파일
    assert picks[0] == (10, 0)  # centroid에 가장 가까운 윈도우
    assert picks[1] == (12, 3)  # file 11(sim 0.20) 건너뛰고 file 12(sim 0.90) 선택


from jhm_extract.anchor import rank_clusters


def test_rank_clusters_orders_by_file_then_window():
    labels = np.array([0, 0, 1, 1, 1, 2])
    file_ids = np.array([0, 1, 2, 2, 2, 3])
    # cluster0: 2 files,2 win ; cluster1: 1 file,3 win ; cluster2: 1 file,1 win
    assert rank_clusters(labels, file_ids) == [0, 1, 2]


def test_rank_clusters_first_equals_pick_target():
    labels = np.array([0, 0, 1, 1, 1])
    file_ids = np.array([0, 1, 0, 1, 1])
    r = rank_clusters(labels, file_ids)
    assert r[0] == pick_target_cluster(labels, file_ids) == 1

from jhm.device import pick_device


def test_prefers_cuda_when_available():
    avail = {"cuda": True, "mps": True, "cpu": True}
    assert pick_device(available=avail) == "cuda"


def test_falls_back_to_mps_without_cuda():
    avail = {"cuda": False, "mps": True, "cpu": True}
    assert pick_device(available=avail) == "mps"


def test_falls_back_to_cpu():
    avail = {"cuda": False, "mps": False, "cpu": True}
    assert pick_device(available=avail) == "cpu"


def test_prefer_honored_when_available():
    avail = {"cuda": True, "mps": True, "cpu": True}
    assert pick_device(prefer="mps", available=avail) == "mps"


def test_prefer_ignored_when_unavailable():
    avail = {"cuda": True, "mps": False, "cpu": True}
    assert pick_device(prefer="mps", available=avail) == "cuda"

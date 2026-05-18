from jhm_extract.cleanliness import cleanliness_gate

NF = 1e-3  # 테스트용 고정 노이즈 플로어


def test_clean_speech_passes(clean_speech):
    m = cleanliness_gate(clean_speech, noise_floor=NF)
    assert m.passed, m.reasons
    assert m.gap_rms_ratio < 4.0
    assert m.voiced_ratio >= 0.7


def test_music_bed_rejected_by_gap_energy(speech_with_music_bed):
    m = cleanliness_gate(speech_with_music_bed, noise_floor=NF)
    assert not m.passed
    assert any("gap_energy" in r for r in m.reasons)


def test_clipping_rejected(clipped):
    m = cleanliness_gate(clipped, noise_floor=NF)
    assert not m.passed
    assert any("clipping" in r for r in m.reasons)


def test_silence_rejected_low_voiced_ratio(mostly_silence):
    m = cleanliness_gate(mostly_silence, noise_floor=NF)
    assert not m.passed
    assert any("voiced_ratio" in r for r in m.reasons)


def test_continuous_tone_rejected_no_pauses(continuous_tone):
    m = cleanliness_gate(continuous_tone, noise_floor=NF)
    assert not m.passed
    assert any("no_pauses" in r for r in m.reasons)


def test_metrics_serializable(clean_speech):
    m = cleanliness_gate(clean_speech, noise_floor=NF)
    d = m.as_dict()
    assert isinstance(d, dict)
    assert set(d) >= {"voiced_ratio", "gap_rms_ratio", "pause_ratio",
                      "spectral_flatness", "peak", "longest_sil_s",
                      "passed", "reasons"}

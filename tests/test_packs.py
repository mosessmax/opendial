import pytest

from opendial.packs import AccentPack, LinePack, accent, line, packs, resolve_audio
from opendial.scenario import Audio, Noise


def test_builtin_packs_load():
    found = packs()
    assert isinstance(found["ng-pidgin"], AccentPack)
    assert isinstance(found["ng-yoruba-english"], AccentPack)
    for pack_id in ("ng-mobile-3g", "ng-market-call", "ng-poor-signal"):
        assert isinstance(found[pack_id], LinePack)


def test_lookup_checks_kind():
    assert accent("ng-pidgin").tts_locale == "en-NG"
    with pytest.raises(KeyError):
        accent("ng-mobile-3g")
    with pytest.raises(KeyError):
        line("nope")


def test_resolve_audio_lets_scenario_override_pack():
    audio = resolve_audio(Audio(line="ng-market-call", noise=Noise(kind="babble", snr_db=3)))
    assert audio.codec == "gsm-like"
    assert audio.packet_loss.rate == 0.02
    assert audio.noise == Noise(kind="babble", snr_db=3)


def test_resolve_without_line_is_identity():
    audio = Audio(narrowband=True)
    assert resolve_audio(audio) is audio

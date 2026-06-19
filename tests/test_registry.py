"""Tests for FlashTrack registry."""

from flashtrack.registry import BACKBONES, ENCODERS, HEADS, LOSSES, TRACKERS, Registry


def test_registry_register_and_build():
    """Test basic register and build."""
    reg = Registry("test")

    @reg.register("TestClass")
    class TestClass:
        def __init__(self, value=42):
            self.value = value

    assert "TestClass" in reg
    obj = reg.build("TestClass", value=99)
    assert obj.value == 99


def test_registry_auto_name():
    """Test registration without explicit name."""
    reg = Registry("test")

    @reg.register()
    class AutoNamed:
        pass

    assert "AutoNamed" in reg


def test_registry_list():
    """Test listing registered names."""
    reg = Registry("test")

    @reg.register("A")
    class A:
        pass

    @reg.register("B")
    class B:
        pass

    names = reg.list()
    assert names == ["A", "B"]


def test_registry_duplicate():
    """Test duplicate registration raises error."""
    reg = Registry("test")

    @reg.register("Dup")
    class Dup1:
        pass

    import pytest
    with pytest.raises(KeyError):
        @reg.register("Dup")
        class Dup2:
            pass


def test_global_registries_exist():
    """Verify global registries are defined."""
    assert BACKBONES.name == "backbones"
    assert ENCODERS.name == "encoders"
    assert HEADS.name == "heads"
    assert LOSSES.name == "losses"
    assert TRACKERS.name == "trackers"

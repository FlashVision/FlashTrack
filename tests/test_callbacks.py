"""Tests for FlashTrack callbacks."""

from flashtrack.engine.callbacks import Callback, CallbackList, EarlyStopping


def test_callback_list():
    """Test CallbackList fires events."""
    results = []

    class TestCallback(Callback):
        def on_train_start(self, trainer):
            results.append("started")

        def on_epoch_end(self, trainer, epoch, metrics):
            results.append(f"epoch_{epoch}")

    cb_list = CallbackList([TestCallback()])
    cb_list.fire("on_train_start", None)
    cb_list.fire("on_epoch_end", None, 1, {"loss": 0.5})

    assert results == ["started", "epoch_1"]


def test_early_stopping():
    """Test EarlyStopping callback."""
    es = EarlyStopping(patience=3, metric="val_loss", mode="min")

    # Improving
    es.on_epoch_end(None, 1, {"val_loss": 1.0})
    assert not es.should_stop

    es.on_epoch_end(None, 2, {"val_loss": 0.9})
    assert not es.should_stop

    # Not improving
    es.on_epoch_end(None, 3, {"val_loss": 0.95})
    es.on_epoch_end(None, 4, {"val_loss": 0.95})
    es.on_epoch_end(None, 5, {"val_loss": 0.95})

    assert es.should_stop


def test_callback_add():
    """Test adding callbacks to CallbackList."""
    cb_list = CallbackList()
    assert len(cb_list.callbacks) == 0

    cb_list.add(Callback())
    assert len(cb_list.callbacks) == 1

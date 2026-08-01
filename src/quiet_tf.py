"""Silence TensorFlow's startup noise so the captured logs show only real output.

**Import this BEFORE tensorflow.** Setting TF_CPP_MIN_LOG_LEVEL after the C++ runtime
has initialised has no effect, so import order matters:

    import quiet_tf            # noqa: F401  — must come first
    import tensorflow as tf

There are four separate noise sources. This module fixes three of them:

1. `WARNING: All log messages before absl::InitializeLog() ...`
   `I0000 ... port.cc:153] oneDNN custom operations are on ...`
       **NOT fixable from Python.** These are written directly to file descriptor 2 by
       TensorFlow's C++ layer *before* absl logging initialises — which is exactly what
       the first line says. TF_CPP_MIN_LOG_LEVEL is read by absl after init, so it
       cannot gate messages emitted before init, and replacing `sys.stderr` does not
       help either because the writes bypass Python entirely. Verified: setting
       TF_CPP_MIN_LOG_LEVEL=3 leaves both lines in place.
       These are stripped at the stream level by `scripts/run_all.py` instead.
2. `WARNING:tensorflow:TensorFlow GPU support is not available on native Windows ...`
       Python `tensorflow` logger  ->  tf.get_logger().setLevel()
3. `WARNING:tensorflow:From ...global_state.py:82: tf.reset_default_graph is deprecated`
       tf.compat.v1 deprecation logger  ->  same logger, set before keras_tuner touches it
4. `saveable.load_own_variables(store)`
       a keras UserWarning during get_best_models()  ->  warnings.filterwarnings

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
The oneDNN message suggests `TF_ENABLE_ONEDNN_OPTS=0` to make it go away. That is not
used here: turning oneDNN off changes the floating-point computation order, which would
change results and break the seeded reproducibility the repo depends on. oneDNN stays
on; only the message is silenced.

This module suppresses *log chatter only*. Real errors, exceptions and Python-level
warnings from this repo's own code still surface.
"""

from __future__ import annotations

import logging
import os
import warnings

# 1. C++ / absl layer. Must be set before tensorflow is imported.
#    0=all, 1=no INFO, 2=no WARNING, 3=no ERROR (FATAL still shown).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Keras 3 prints its own startup banner on some versions.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

# 4. keras saving emits a UserWarning when keras-tuner reloads a trial's best model.
warnings.filterwarnings("ignore", category=UserWarning, module=r"keras\..*")
warnings.filterwarnings("ignore", message=r".*load_own_variables.*")
warnings.filterwarnings("ignore", message=r".*Skipping variable loading.*")

import tensorflow as tf  # noqa: E402  — deliberately after the env vars above

# 2 + 3. Python-side loggers.
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
try:  # present on TF 2.x, but do not hard-fail if the shim ever goes away
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception:  # noqa: BLE001
    pass

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

__all__ = ["tf"]

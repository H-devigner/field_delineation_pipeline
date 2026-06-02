import numpy as np


def test_integer_masked_arrays_can_be_filled_with_nan_after_float_cast():
    data = np.ma.array(
        [[[1, 2], [3, 4]]],
        mask=[[[False, True], [False, False]]],
        dtype=np.int16,
    )

    filled = data.astype("float32").filled(np.nan)

    assert filled.dtype == np.float32
    assert np.isnan(filled[0, 0, 1])
    assert filled[0, 1, 0] == 3

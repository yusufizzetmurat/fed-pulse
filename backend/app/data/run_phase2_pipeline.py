from __future__ import annotations

import warnings

from app.data.pipeline_data_prep import main


if __name__ == "__main__":
    warnings.warn(
        "app.data.run_phase2_pipeline is deprecated. Use app.data.pipeline_data_prep instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    raise SystemExit(main())


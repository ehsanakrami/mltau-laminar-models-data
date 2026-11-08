from pathlib import Path
import json
import numpy as np
from mltau_models import predict
root=Path(__file__).resolve().parents[1]
model=json.loads((root/"models/pressure45/selected_ridge.json").read_text())
with np.load(root/"tests/fixtures/pressure45_forward.npz",allow_pickle=False) as z:
    output=predict(model,z["features"],"continuous_metric24")
    np.testing.assert_allclose(output,z[model["name"]],rtol=2e-13,atol=1e-14)
print(json.dumps({"model":model["name"],"rows":len(output),"coefficient_min":float(output.min()),"coefficient_max":float(output.max()),"fixture_agrees":True},indent=2))

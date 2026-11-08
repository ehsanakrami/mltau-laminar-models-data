from pathlib import Path
import json,unittest
import numpy as np
from mltau_models import predict,constant_coefficient
ROOT=Path(__file__).resolve().parents[1]
class InferenceTests(unittest.TestCase):
 def setUp(self):
  self.models=json.loads((ROOT/'models/pressure45/MODELS.json').read_text())['models']
  with np.load(ROOT/'tests/fixtures/pressure45_forward.npz',allow_pickle=False) as z:self.arrays={k:z[k] for k in z.files}
 def test_frozen_physical_feature_predictions(self):
  for m in self.models:np.testing.assert_allclose(predict(m,self.arrays['features'],m['feature_variant']),self.arrays[m['name']],rtol=2e-13,atol=1e-14)
 def test_affine_independent_matrix_formula(self):
  m=next(x for x in self.models if x['kind']=='affine');x=self.arrays['features'];a=np.clip((np.arcsinh(x)-m['transform']['mean'])/m['transform']['scale'],-8,8)
  expected=np.exp(np.clip(a@np.asarray(m['parameters'][0])+m['parameters'][1],np.log(.004),np.log(30.)))
  np.testing.assert_allclose(predict(m,x,m['feature_variant']),expected,rtol=2e-13,atol=1e-14)
 def test_contract_and_nonfinite_inputs_refused(self):
  m=self.models[1];x=self.arrays['features'].copy()
  for bad,contract in [(x[:,:23],m['feature_variant']),(x,'wrong_contract'),(x*np.nan,m['feature_variant'])]:
   with self.assertRaises(ValueError):predict(m,bad,contract)
 def test_constant_needs_no_computed_predictor(self):
  m=next(x for x in self.models if x['kind']=='constant');self.assertAlmostEqual(constant_coefficient(m),float(predict(m,np.zeros((1,24)),m['feature_variant'])[0]))
if __name__=='__main__':unittest.main()

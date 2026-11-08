from pathlib import Path
import sys,json,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis/rectangular_features'))
try:
 import rect_fe,continuous_metric
 AVAILABLE=True
except ImportError:AVAILABLE=False
@unittest.skipUnless(AVAILABLE,'optional netCDF4 dependency unavailable')
class SavedFieldTests(unittest.TestCase):
 def test_native_predictor_feature_parity(self):
  expected=json.loads((ROOT/'tests/fixtures/predictor_expected.json').read_text());mesh=rect_fe.load(ROOT/'tests/fixtures/predictor_q1_mms_n10_nu006.e');ids,features=continuous_metric.augment(mesh,expected['viscosity'])
  self.assertEqual(len(ids),100);np.testing.assert_allclose(features,expected['features'],rtol=1e-10,atol=1e-11)

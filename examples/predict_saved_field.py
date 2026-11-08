"""Infer coefficients from a saved rectangular Q1 computed SH predictor."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from mltau_models import predict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analysis/rectangular_features'))
import rect_fe,continuous_metric
p=argparse.ArgumentParser(description=__doc__);p.add_argument('field',type=Path);p.add_argument('--nu',type=float,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.nu<=0 or not np.isfinite(a.nu):p.error('viscosity must be finite and positive')
model=json.loads((ROOT/'models/pressure45/selected_ridge.json').read_text());mesh=rect_fe.load(a.field);ids,features=continuous_metric.augment(mesh,a.nu);coefficients=predict(model,features,'continuous_metric24')
np.savetxt(a.output,np.column_stack([ids,coefficients]),delimiter=',',header='elem_id,coefficient',comments='',fmt=['%d','%.17g'])
print(json.dumps({'field':a.field.name,'rows':len(ids),'model':model['name'],'output':str(a.output)}))

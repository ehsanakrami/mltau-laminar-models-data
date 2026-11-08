"""Truth-free features for axis-aligned rectangular Q1 elements only.

Native Exodus element maps are checked against solver-written centroid IDs.
There is deliberately no analytic-solution import or missing-data fallback.
"""
import numpy as np
from netCDF4 import Dataset

def decode_names(raw):
    # Exodus C names end at the first NUL. Some local writer builds leave
    # nontext padding after that terminator; it is not part of the name.
    names=[np.ma.filled(row,b'\0').tobytes().split(b'\0',1)[0].decode('ascii') for row in raw]
    if any(not n for n in names) or len(names)!=len(set(names)):
        raise ValueError('Empty or duplicate Exodus variable name')
    return names

FEATS = ['umag', 'Pe', 'Pe_long', 'Pe_short', 'Pe_aniso', 'Snorm',
         'Wnorm', 'Q', 'divu', 'gradnorm', 'vsr', 'h', 'aspect', 'align',
         'xi_coth', 'xi_long', 'xi_short']


def coth_xi(pe):
    return pe / 3 if pe < 1e-4 else 1 / np.tanh(pe) - 1 / pe


def corners_features(c, hx, hy, nu):
    if min(hx, hy, nu) <= 0 or not np.isfinite(c).all():
        raise ValueError('Invalid geometry, viscosity or velocity')
    c00, c10, c11, c01 = c
    u = np.mean(c, axis=0)
    dx = (c10 + c11 - c00 - c01) / (2 * hx)
    dy = (c01 + c11 - c00 - c10) / (2 * hy)
    g = np.column_stack((dx, dy))
    sn = np.linalg.norm((g + g.T) / 2)
    wn = np.linalg.norm((g - g.T) / 2)
    speed = np.linalg.norm(u)
    h = np.sqrt(hx * hy)
    pe = speed * h / (2 * nu)
    hl, hs, ul, us = ((hx, hy, abs(u[0]), abs(u[1])) if hx >= hy
                      else (hy, hx, abs(u[1]), abs(u[0])))
    pl, ps = ul * hl / (2 * nu), us * hs / (2 * nu)
    return np.array([speed, pe, pl, ps,
                     min(max(pl, ps) / max(min(pl, ps), 1e-6), 100),
                     sn, wn, (wn**2 - sn**2) / 2, abs(np.trace(g)),
                     np.linalg.norm(g), wn / max(sn, 1e-12), h, hl / hs,
                     ul / max(speed, 1e-12), coth_xi(pe), coth_xi(pl), coth_xi(ps)])


def read_state(path):
    with Dataset(path) as d:
        names = decode_names(d['name_nod_var'][:])
        fields = {n: np.asarray(d[f'vals_nod_var{i+1}'][:])
                  for i, n in enumerate(names)}
        return (np.column_stack((d['coordx'][:], d['coordy'][:])),
                np.asarray(d['connect1'][:], dtype=int) - 1,
                np.asarray(d['elem_num_map'][:], dtype=int) - 1,
                np.asarray(d['time_whole'][:]), fields)


def extract(exodus, sens, nu):
    xy, conn, ids, times, fields = read_state(exodus)
    if len(set(ids)) != len(ids) or conn.shape != (len(ids), 4):
        raise ValueError('Expected unique mapped QUAD4 elements')
    reference = np.atleast_1d(np.genfromtxt(sens, delimiter=',', names=True))
    expected_ids = reference['elem_id'].astype(int)
    if len(set(expected_ids)) != len(ids) or set(expected_ids) != set(ids):
        raise ValueError('Exodus and solver element IDs disagree')
    centroids = {int(r['elem_id']): np.array([r['xc'], r['yc']]) for r in reference}
    velocity = np.column_stack((fields['u'][-1], fields['v'][-1]))
    rows = []
    for eid, nodes in zip(ids, conn):
        points = xy[nodes]
        lo, hi = points.min(axis=0), points.max(axis=0)
        hx, hy = hi - lo
        wanted = np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]])
        distance = np.linalg.norm(wanted[:, None] - points[None, :], axis=2)
        order = distance.argmin(axis=1)
        if len(set(order)) != 4 or distance[np.arange(4), order].max() > 1e-10:
            raise ValueError('Nonrectangular or ambiguous Q1 element')
        # Solver CSV prints six significant digits; only validate identity with
        # it. Geometry/features themselves use full-precision Exodus coordinates.
        if not np.allclose(points.mean(axis=0), centroids[eid], atol=5.1e-7, rtol=0):
            raise ValueError('Element map centroid mismatch')
        rows.append(corners_features(velocity[nodes[order]], hx, hy, nu))
    result = np.asarray(rows)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite feature')
    return ids, result


def predict(model, x):
    if model['features'] != FEATS or model['activation'] != 'tanh':
        raise ValueError('Unsupported model contract')
    z = (x - np.array(model['normalization']['mu'])) / np.array(model['normalization']['sd'])
    predictions = []
    for member in model['ensemble']:
        a = z
        for i, (w, b) in enumerate(zip(member['W'], member['b'])):
            a = a @ np.asarray(w) + np.asarray(b)
            if i + 1 < len(member['W']):
                a = np.tanh(a)
        predictions.append(a[:, 0])
    c = np.clip(np.maximum(x[:, FEATS.index('xi_coth')], 1e-6) *
                np.exp(np.mean(predictions, axis=0)), model['cmin'], model['cmax'])
    if not np.isfinite(c).all():
        raise ValueError('Nonfinite prediction')
    return c

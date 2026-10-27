"""
grasp.solve compiled with numba: the same damped Gauss-Newton, the same forward kinematics (world.World.frame for the
gripper part G), the same residuals (grasp.errors), only without Python and tiny-numpy overhead on every step.

Profiled, a from-scratch plan_best for a 272 mm block was 85 s on the laptop (139 s on the Pi), 96% of it
inside solve() -- 943,000 forward-kinematics calls on 3x3 arrays. numba is on the Pi (0.63, system Python); without
it, or for a world this does not cover (a tower lean), grasp.solve stays on its Python path. tests/test_world.py
checks this against the Python path.
"""
import math

import numpy as np

try:
    import warnings
    with warnings.catch_warnings():               # numba pokes at scipy on import; the Pi's cannot load (numpy 1.24)
        warnings.simplefilter("ignore")           #   and says so -- harmless here, nothing below uses scipy
        import numba
    _njit = numba.njit(cache=True)
except Exception:                                   # no numba: grasp.solve uses its Python path
    numba = None
    _njit = None

DEG = math.pi / 180.0


def params(w):
    """The numbers the compiled kinematics need, from a World (kept on it), or None when it cannot do this world."""
    if numba is None or getattr(w, "lean", None) is not None or not hasattr(w, "_rotate_axis"):
        return None
    got = w.__dict__.get("_fastik")
    a = w.arm
    key = (tuple(sorted(a.items())), tuple(w.ref), w.gain, w.base_zero, tuple(w.axis), tuple(w.tip_in_G))
    if got is not None and got[0] == key:
        return got[1]
    n, pa = w._rotate_axis()
    p = np.array([a["th1"], a["k1"], a["th2"], a["k2"], a["th3"], a["k3"], a["th4"], a["k4"], a["k4b"],
                  a["L1"], a["L2"], w.ref[0], w.ref[1], w.ref[2], w.ref[3], w.gain, w.base_zero,
                  w.axis[0], w.axis[1], w.axis[2], n[0], n[1], n[2], pa[0], pa[1], pa[2]], float)
    w.__dict__["_fastik"] = (key, p)
    return p


if numba is not None:
    # No @ and no np.linalg in here: numba hands those to BLAS/LAPACK through scipy, and the Pi's scipy cannot load
    # (it wants numpy 1.26.4+, the Pi's system Python has 1.24.2 and is shared with other projects). 3x3 products and
    # a 5x5 Gaussian elimination by hand are all this needs, and plain loops are what numba compiles best anyway.
    @_njit
    def _mm(A, B):
        C = np.zeros((A.shape[0], B.shape[1]))
        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                s = 0.0
                for k in range(A.shape[1]):
                    s += A[i, k] * B[k, j]
                C[i, j] = s
        return C

    @_njit
    def _mv(A, v):
        out = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            s = 0.0
            for k in range(A.shape[1]):
                s += A[i, k] * v[k]
            out[i] = s
        return out

    @_njit
    def _lin_solve(A, b):
        """A x = b, Gaussian elimination with partial pivoting (what LAPACK's gesv does)."""
        n = b.shape[0]
        M = A.copy()
        x = b.copy()
        for c in range(n):
            piv = c
            for r in range(c + 1, n):
                if abs(M[r, c]) > abs(M[piv, c]):
                    piv = r
            if piv != c:
                for k in range(n):
                    M[c, k], M[piv, k] = M[piv, k], M[c, k]
                x[c], x[piv] = x[piv], x[c]
            for r in range(c + 1, n):
                f = M[r, c] / M[c, c]
                for k in range(c, n):
                    M[r, k] -= f * M[c, k]
                x[r] -= f * x[c]
        for c in range(n - 1, -1, -1):
            s = x[c]
            for k in range(c + 1, n):
                s -= M[c, k] * x[k]
            x[c] = s / M[c, c]
        return x

    @_njit
    def _rotvec(rx, ry, rz):
        th = math.sqrt(rx * rx + ry * ry + rz * rz)
        R = np.eye(3)
        if th < 1e-12:
            return R
        kx, ky, kz = rx / th, ry / th, rz / th
        K = np.zeros((3, 3))
        K[0, 1], K[0, 2] = -kz, ky
        K[1, 0], K[1, 2] = kz, -kx
        K[2, 0], K[2, 1] = -ky, kx
        return R + math.sin(th) * K + (1.0 - math.cos(th)) * _mm(K, K)

    @_njit
    def _hand(p, tipG, q):
        """(tip, R of G) in table mm: World.frame(G) and grasp.hand, base_offset 0, no slack."""
        d0, d1, d2, d3 = q[0] - p[11], q[1] - p[12], q[2] - p[13], q[3] - p[14]
        a1 = (p[0] + p[1] * d0) * DEG
        a2 = a1 + (p[2] + p[3] * d1) * DEG
        a3 = a2 + (p[4] + p[5] * d2) * DEG
        a4 = (p[6] + p[7] * d3 + p[8] * d3 * d3) * DEG
        ang = -(p[15] * (q[5] + 0.0 + p[16] + 0.0)) * DEG
        c, s = math.cos(ang), math.sin(ang)
        Rw = np.zeros((3, 3))
        Rw[0, 0], Rw[0, 1], Rw[1, 0], Rw[1, 1], Rw[2, 2] = c, -s, s, c, 1.0
        ax = np.array([p[17], p[18], p[19]])
        oT0 = ax - _mv(Rw, ax)
        c3, s3 = math.cos(a3), math.sin(a3)
        RT3 = np.zeros((3, 3))
        RT3[0, 0], RT3[0, 2], RT3[1, 1], RT3[2, 0], RT3[2, 2] = c3, -s3, 1.0, s3, c3
        oT3 = np.array([p[9] * math.cos(a1) + p[10] * math.cos(a2), 0.0,
                        p[9] * math.sin(a1) + p[10] * math.sin(a2)])
        Rr = _rotvec(a4 * p[20], a4 * p[21], a4 * p[22])
        pa = np.array([p[23], p[24], p[25]])
        RT5 = _mm(RT3, Rr)
        oT5 = oT3 + _mv(RT3, pa - _mv(Rr, pa))
        RwT = Rw.T.copy()
        R = _mm(RwT, RT5)
        o = _mv(RwT, oT5 - oT0)
        return _mv(R, tipG) + o, R

    @_njit
    def _errors(p, tipG, q, target, pitch_deg, face, wp, ws):
        tip, R = _hand(p, tipG, q)
        out = np.empty(5)
        out[0], out[1], out[2] = tip[0] - target[0], tip[1] - target[1], tip[2] - target[2]
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
        y0, y1 = R[0, 1], R[1, 1]
        nrm = max(1e-9, math.sqrt(y0 * y0 + y1 * y1))
        sq = math.degrees(math.asin(max(-1.0, min(1.0, y0 / nrm * math.sin(face) - y1 / nrm * math.cos(face)))))
        out[3], out[4] = wp * (pitch - pitch_deg), ws * sq
        return out


def solve(w, target, pitch_deg, face, grip, start, iters, wp, ws, lim):
    """grasp.solve's answer, compiled; None when it cannot do this world (no numba, a tower lean)."""
    p = params(w)
    if p is None:
        return None
    tipG = np.asarray(w.tip_in_G + w.tip_shift(grip), float)
    st = np.asarray(list(start[:4]) + [grip, start[5]], float)
    q, r = _solve2(p, tipG, np.asarray(target, float), float(pitch_deg), float(face), float(grip), st, int(iters),
                   float(wp), float(ws), lim)
    return [float(v) for v in q], r


if numba is not None:
    @_njit
    def _gaps(X, plane, bias, ax0, ax1, ply, binw, mode, c_rlo, c_rhi, c_mid, c_h, angle_bin):
        """tablecheck.TableCheck.gaps after points(): (gap, low) for the arm points X, the same arithmetic in loops.
        mode 0: the spun plywood profile only; 1: a bearing map with no cells (low, low); 2: the bearing map."""
        n = X.shape[0]
        z = np.empty(n)
        low = 1e300
        for i in range(n):
            z[i] = X[i, 2] - (plane[0] * X[i, 0] + plane[1] * X[i, 1] + plane[2]) - bias
            low = min(low, z[i])
        top = ply[0]
        for k in range(ply.shape[0]):
            top = max(top, ply[k])
        gap = low
        nb = ply.shape[0]
        idx = np.empty(n, np.int64)
        m = 0
        for i in range(n):
            if z[i] < top + 150.0:
                idx[m] = i
                m += 1
        if m == 0:
            return gap, low
        if mode == 1:
            return low, low
        rn = np.empty(m)
        lb = np.empty(m)
        for j in range(m):
            i = idx[j]
            rn[j] = math.sqrt((X[i, 0] - ax0) ** 2 + (X[i, 1] - ax1) ** 2)
            best = 1e300
            for k in range(nb):
                r_lo = k * binw
                dr = max(max(r_lo - rn[j], rn[j] - (r_lo + binw)), 0.0)
                dz = max(z[i] - ply[k], 0.0)
                best = min(best, math.sqrt(dr * dr + dz * dz))
            lb[j] = best
        if mode == 0:
            for j in range(m):
                gap = min(gap, lb[j])
            return min(gap, low), low
        order = np.argsort(lb)
        half = angle_bin / 2.0
        for jj in range(m):
            j = order[jj]
            if lb[j] >= gap:
                break
            i = idx[j]
            th = math.degrees(math.atan2(X[i, 1] - ax1, X[i, 0] - ax0)) % 360.0
            for c in range(c_rlo.shape[0]):
                d_out = max(max(c_rlo[c] - rn[j], rn[j] - c_rhi[c]), 0.0)
                dth = max(abs((th - c_mid[c] + 180.0) % 360.0 - 180.0) - half, 0.0)
                d_round = math.radians(dth) * rn[j]
                d_up = max(z[i] - c_h[c], 0.0)
                gap = min(gap, math.sqrt(d_out * d_out + d_round * d_round + d_up * d_up))
        return gap, low


def gaps(check, X):
    """TableCheck.gaps' answer for the arm points X, compiled; None without numba."""
    if numba is None:
        return None
    if check.ply2 is None:
        mode, cells = 0, None
    elif check._cells is None:
        mode, cells = 1, None
    else:
        mode, cells = 2, check._cells
    e = np.zeros(1)
    c_rlo, c_rhi, c_mid, c_h = (np.ascontiguousarray(v, float) for v in cells) if cells is not None else (e, e, e, e)
    return _gaps(np.ascontiguousarray(X, float), np.asarray(check.plane, float), float(check.bias),
                 float(check.axis[0]), float(check.axis[1]), np.ascontiguousarray(check.ply, float), float(check.bin),
                 mode, c_rlo, c_rhi, c_mid, c_h, float(check.angle_bin or 0.0))


if numba is not None:
    @_njit
    def _rot_xz(a):
        c, s = math.cos(a), math.sin(a)
        m = np.zeros((3, 3))
        m[0, 0], m[0, 2], m[1, 1], m[2, 0], m[2, 2] = c, -s, 1.0, s, c
        return m

    @_njit
    def _parts(p, q, base):
        """World.parts (no slack, no base offset, no lean): every part's (R, o) in table mm, as Rs (6,3,3), os (6,3)."""
        d0, d1, d2, d3 = q[0] - p[11], q[1] - p[12], q[2] - p[13], q[3] - p[14]
        a1 = (p[0] + p[1] * d0) * DEG
        a2 = a1 + (p[2] + p[3] * d1) * DEG
        a3 = a2 + (p[4] + p[5] * d2) * DEG
        a4 = (p[6] + p[7] * d3 + p[8] * d3 * d3) * DEG
        ang = -(p[15] * (base + 0.0 + p[16] + 0.0)) * DEG
        c, s = math.cos(ang), math.sin(ang)
        Rw = np.zeros((3, 3))
        Rw[0, 0], Rw[0, 1], Rw[1, 0], Rw[1, 1], Rw[2, 2] = c, -s, s, c, 1.0
        ax = np.array([p[17], p[18], p[19]])
        RT = np.zeros((6, 3, 3))
        oT = np.zeros((6, 3))
        RT[0] = Rw
        oT[0] = ax - _mv(Rw, ax)
        RT[1] = _rot_xz(a1)
        RT[2] = _rot_xz(a2)
        oT[2, 0], oT[2, 2] = p[9] * math.cos(a1), p[9] * math.sin(a1)
        RT[3] = _rot_xz(a3)
        oT[3, 0], oT[3, 2] = oT[2, 0] + p[10] * math.cos(a2), oT[2, 2] + p[10] * math.sin(a2)
        RT[4] = np.eye(3)
        Rr = _rotvec(a4 * p[20], a4 * p[21], a4 * p[22])
        pa = np.array([p[23], p[24], p[25]])
        RT[5] = _mm(RT[3], Rr)
        oT[5] = oT[3] + _mv(RT[3], pa - _mv(Rr, pa))
        RwT = Rw.T.copy()
        Rs = np.zeros((6, 3, 3))
        os_ = np.zeros((6, 3))
        for i in range(6):
            Rs[i] = _mm(RwT, RT[i])
            os_[i] = _mv(RwT, oT[i] - oT[0])
        return Rs, os_

    @_njit
    def _points(p, q, base, local, part):
        """tablecheck.TableCheck.points: each local point (in its part's frame) carried into table mm."""
        Rs, os_ = _parts(p, q, base)
        X = np.empty((local.shape[0], 3))
        for i in range(local.shape[0]):
            k = part[i]
            for r in range(3):
                X[i, r] = Rs[k, r, 0] * local[i, 0] + Rs[k, r, 1] * local[i, 1] + Rs[k, r, 2] * local[i, 2] + os_[k, r]
        return X


def points(check, joints, base):
    """TableCheck.points' answer, compiled; None when it cannot do this world."""
    p = params(check.w)
    if p is None or len(joints) < 5:
        return None
    from .world import PARTS
    g = round(float(joints[4]), 6)
    keep = check.__dict__.setdefault("_fast_local", {})
    got = keep.get(g)
    if got is None:                                  # the local points at this gripper value (the fingers slide)
        iG = PARTS.index("G")
        loc, part = [check.hand["G"]], [np.full(len(check.hand["G"]), iG)]
        for s in ("L", "R"):
            shift = check.w.finger_offset_side(s, g) - check.w.finger_offset_side(s, check.ref_grip)
            loc.append(check.hand[s] + shift)
            part.append(np.full(len(check.hand[s]), iG))
        for f, pts in check.links.items():
            loc.append(pts)
            part.append(np.full(len(pts), PARTS.index(f)))
        got = (np.ascontiguousarray(np.vstack(loc), float), np.concatenate(part).astype(np.int64))
        if len(keep) > 64:
            keep.clear()
        keep[g] = got
    q = np.array([float(v) for v in joints[:4]], float)
    return _points(p, q, float(base), got[0], got[1])


if numba is not None:
    @_njit
    def _err_s(p, tipG, q0, q1, q2, q3, q5, target, pitch_deg, face, wp, ws, out):
        """_errors with scalars only (no arrays made): about a million calls a hard plan, so allocation was the cost."""
        d0, d1, d2, d3 = q0 - p[11], q1 - p[12], q2 - p[13], q3 - p[14]
        a1 = (p[0] + p[1] * d0) * DEG
        a2 = a1 + (p[2] + p[3] * d1) * DEG
        a3 = a2 + (p[4] + p[5] * d2) * DEG
        a4 = (p[6] + p[7] * d3 + p[8] * d3 * d3) * DEG
        ang = -(p[15] * (q5 + 0.0 + p[16] + 0.0)) * DEG
        c, s = math.cos(ang), math.sin(ang)
        # Rodrigues for a4 * n
        rx, ry, rz = a4 * p[20], a4 * p[21], a4 * p[22]
        th = math.sqrt(rx * rx + ry * ry + rz * rz)
        if th < 1e-12:
            r00, r01, r02, r10, r11, r12, r20, r21, r22 = 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0
        else:
            kx, ky, kz = rx / th, ry / th, rz / th
            st, vt = math.sin(th), 1.0 - math.cos(th)
            # I + st K + vt K K, with K = [[0,-kz,ky],[kz,0,-kx],[-ky,kx,0]]
            r00 = 1.0 + vt * (-kz * kz - ky * ky)
            r01 = -st * kz + vt * (ky * kx)
            r02 = st * ky + vt * (kz * kx)
            r10 = st * kz + vt * (kx * ky)
            r11 = 1.0 + vt * (-kz * kz - kx * kx)
            r12 = -st * kx + vt * (kz * ky)
            r20 = -st * ky + vt * (kx * kz)
            r21 = st * kx + vt * (ky * kz)
            r22 = 1.0 + vt * (-ky * ky - kx * kx)
        c3, s3 = math.cos(a3), math.sin(a3)
        # M = rot_xz(a3) @ Rr, rot_xz = [[c3,0,-s3],[0,1,0],[s3,0,c3]]
        m00, m01, m02 = c3 * r00 - s3 * r20, c3 * r01 - s3 * r21, c3 * r02 - s3 * r22
        m10, m11, m12 = r10, r11, r12
        m20, m21, m22 = s3 * r00 + c3 * r20, s3 * r01 + c3 * r21, s3 * r02 + c3 * r22
        # oT5 = oT3 + rot_xz(a3) @ (pa - Rr @ pa)
        pa0, pa1, pa2 = p[23], p[24], p[25]
        v0 = pa0 - (r00 * pa0 + r01 * pa1 + r02 * pa2)
        v1 = pa1 - (r10 * pa0 + r11 * pa1 + r12 * pa2)
        v2 = pa2 - (r20 * pa0 + r21 * pa1 + r22 * pa2)
        o30 = p[9] * math.cos(a1) + p[10] * math.cos(a2)
        o32 = p[9] * math.sin(a1) + p[10] * math.sin(a2)
        t0 = o30 + c3 * v0 - s3 * v2
        t1 = v1
        t2 = o32 + s3 * v0 + c3 * v2
        # oT0 = ax - Rw @ ax, Rw = [[c,-s,0],[s,c,0],[0,0,1]]
        ax0, ax1, ax2 = p[17], p[18], p[19]
        u0 = t0 - (ax0 - (c * ax0 - s * ax1))
        u1 = t1 - (ax1 - (s * ax0 + c * ax1))
        u2 = t2 - (ax2 - ax2)
        # R = Rw^T @ M, o = Rw^T @ u, Rw^T = [[c,s,0],[-s,c,0],[0,0,1]]
        R00, R01, R02 = c * m00 + s * m10, c * m01 + s * m11, c * m02 + s * m12
        R10, R11, R12 = -s * m00 + c * m10, -s * m01 + c * m11, -s * m02 + c * m12
        R20, R21, R22 = m20, m21, m22
        o0, o1, o2 = c * u0 + s * u1, -s * u0 + c * u1, u2
        g0, g1, g2 = tipG[0], tipG[1], tipG[2]
        out[0] = R00 * g0 + R01 * g1 + R02 * g2 + o0 - target[0]
        out[1] = R10 * g0 + R11 * g1 + R12 * g2 + o1 - target[1]
        out[2] = R20 * g0 + R21 * g1 + R22 * g2 + o2 - target[2]
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R20))))
        nrm = max(1e-9, math.sqrt(R01 * R01 + R11 * R11))
        sq = math.degrees(math.asin(max(-1.0, min(1.0, R01 / nrm * math.sin(face) - R11 / nrm * math.cos(face)))))
        out[3] = wp * (pitch - pitch_deg)
        out[4] = ws * sq

    @_njit
    def _clamp_s(v, i, lim):
        if i < 3:
            return min(lim[3 + i], max(lim[i], v))
        if i == 3:
            return min(lim[7], max(lim[6], v))
        if i == 5:
            return min(lim[9], max(lim[8], v))
        return v

    @_njit
    def _solve2(p, tipG, target, pitch_deg, face, grip, start, iters, wp, ws, lim):
        """Damped Gauss-Newton on grasp.errors, without making arrays inside the loop (same steps as grasp.solve)."""
        q = start.copy()
        q[4] = grip
        for i in (0, 1, 2, 3, 5):
            q[i] = _clamp_s(q[i], i, lim)
        idx = (0, 1, 2, 3, 5)
        lam = 1e-2
        r = np.empty(5)
        rp = np.empty(5)
        rc = np.empty(5)
        J = np.empty((5, 5))
        A = np.empty((5, 5))
        g = np.empty(5)
        cand = np.empty(6)
        _err_s(p, tipG, q[0], q[1], q[2], q[3], q[5], target, pitch_deg, face, wp, ws, r)
        for _ in range(iters):
            for k in range(5):
                i = idx[k]
                keep = q[i]
                q[i] = keep + 0.05
                _err_s(p, tipG, q[0], q[1], q[2], q[3], q[5], target, pitch_deg, face, wp, ws, rp)
                q[i] = keep
                for m in range(5):
                    J[m, k] = (rp[m] - r[m]) / 0.05
            for a in range(5):
                s_ = 0.0
                for m in range(5):
                    s_ += J[m, a] * r[m]
                g[a] = -s_
                for b in range(5):
                    s2 = 0.0
                    for m in range(5):
                        s2 += J[m, a] * J[m, b]
                    A[a, b] = s2 + (lam if a == b else 0.0)
            step = _lin_solve(A, g)
            for m in range(6):
                cand[m] = q[m]
            for k in range(5):
                i = idx[k]
                cand[i] = _clamp_s(cand[i] + min(15.0, max(-15.0, step[k])), i, lim)
            _err_s(p, tipG, cand[0], cand[1], cand[2], cand[3], cand[5], target, pitch_deg, face, wp, ws, rc)
            nr, nc = 0.0, 0.0
            for m in range(5):
                nr += r[m] * r[m]
                nc += rc[m] * rc[m]
            if math.sqrt(nc) < math.sqrt(nr) - 1e-9:
                for m in range(6):
                    q[m] = cand[m]
                for m in range(5):
                    r[m] = rc[m]
                lam = max(1e-4, lam / 3)
            else:
                lam *= 5
                if lam > 1e4:
                    break
        return q, r

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from orbitforge.core.constants import R_EARTH_EQUATOR_KM
from orbitforge.core.errors import ConvergenceError
from orbitforge.core.state import CartesianState
from orbitforge.core.vector import Vec3
from .perturbations import gravity_j2

Accel = Callable[[Vec3, Vec3], Vec3]
EventFn = Callable[[float, Vec3, Vec3], float]

_EPS = 2.220446049250313e-16


def _j2(r: Vec3, v: Vec3) -> Vec3:
    return gravity_j2(r)


# Dormand-Prince RK5(4) tableau; stage 7 is the FSAL evaluation at the new point.
_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_B = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
_E = (71 / 57600, 0.0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40)


def _err_norm(er: Vec3, ev: Vec3, r0: Vec3, r1: Vec3, v0: Vec3, v1: Vec3, tol: float) -> float:
    sr = tol * max(r0.norm(), r1.norm(), 1e-3)
    sv = tol * max(v0.norm(), v1.norm(), 1e-6)
    return math.sqrt((er.norm2() / (sr * sr) + ev.norm2() / (sv * sv)) / 6.0)


@dataclass(frozen=True)
class _DenseStep:
    """One accepted step with quintic Hermite dense output in (r, v, a)."""
    t0: float
    r0: Vec3
    v0: Vec3
    a0: Vec3
    t1: float
    r1: Vec3
    v1: Vec3
    a1: Vec3

    def state_at(self, t: float):
        h = self.t1 - self.t0
        u = (t - self.t0) / h
        u2, u3 = u * u, u * u * u
        u4, u5 = u3 * u, u3 * u * u
        h0 = 1.0 - 10.0 * u3 + 15.0 * u4 - 6.0 * u5
        h1 = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
        h2 = u - 6.0 * u3 + 8.0 * u4 - 3.0 * u5
        h3 = -4.0 * u3 + 7.0 * u4 - 3.0 * u5
        h4 = 0.5 * u2 - 1.5 * u3 + 1.5 * u4 - 0.5 * u5
        h5 = 0.5 * u3 - u4 + 0.5 * u5
        r = (self.r0 * h0 + self.r1 * h1 + self.v0 * (h2 * h) + self.v1 * (h3 * h)
             + self.a0 * (h4 * h * h) + self.a1 * (h5 * h * h))
        d0 = (-30.0 * u2 + 60.0 * u3 - 30.0 * u4) / h
        d1 = (30.0 * u2 - 60.0 * u3 + 30.0 * u4) / h
        d2 = 1.0 - 18.0 * u2 + 32.0 * u3 - 15.0 * u4
        d3 = -12.0 * u2 + 28.0 * u3 - 15.0 * u4
        d4 = (u - 4.5 * u2 + 6.0 * u3 - 2.5 * u4) * h
        d5 = (1.5 * u2 - 4.0 * u3 + 2.5 * u4) * h
        v = (self.r0 * d0 + self.r1 * d1 + self.v0 * d2 + self.v1 * d3
             + self.a0 * d4 + self.a1 * d5)
        return r, v


def _steps(state: CartesianState, dt_s: float, h_max: float, tol: float,
           accel: Accel, h_min: float):
    """Yield _DenseStep for each accepted step from t to t + dt_s (either sign)."""
    t_end = state.epoch_tai_s + dt_s
    sign = 1.0 if dt_s > 0 else -1.0
    t = state.epoch_tai_s
    r, v = state.position_km, state.velocity_km_s
    a = accel(r, v)
    h = sign * min(h_max, abs(dt_s))
    while (t_end - t) * sign > 0.0:
        last = abs(h) >= abs(t_end - t)
        if last:
            h = t_end - t
        kr: list[Vec3] = [None] * 7  # type: ignore[list-item]
        kv: list[Vec3] = [None] * 7  # type: ignore[list-item]
        kr[0], kv[0] = v, a
        for i in range(1, 7):
            ri, vi = r, v
            for j, aij in enumerate(_A[i]):
                if aij:
                    ri = ri + kr[j] * (aij * h)
                    vi = vi + kv[j] * (aij * h)
            kr[i], kv[i] = vi, accel(ri, vi)
        r1, v1 = r, v
        er = Vec3(0.0, 0.0, 0.0)
        ev = Vec3(0.0, 0.0, 0.0)
        for i in range(7):
            bi, ei = _B[i], _E[i]
            if bi:
                r1 = r1 + kr[i] * (bi * h)
                v1 = v1 + kv[i] * (bi * h)
            if ei:
                er = er + kr[i] * (ei * h)
                ev = ev + kv[i] * (ei * h)
        err = _err_norm(er, ev, r, r1, v, v1, tol)
        if err <= 1.0:
            t1 = t_end if last else t + h
            yield _DenseStep(t, r, v, a, t1, r1, v1, kv[6])
            t, r, v, a = t1, r1, v1, kv[6]
            fac = 10.0 if err == 0.0 else min(10.0, max(0.2, 0.9 * err ** -0.2))
        else:
            fac = max(0.2, 0.9 * err ** -0.2)
        h *= fac
        if abs(h) > h_max:
            h = math.copysign(h_max, h)
        if abs(h) < h_min and abs(t_end - t) > h_min:
            raise ConvergenceError('cowell: step size underflow (singular dynamics?)')


def _brent(f, a: float, b: float, fa: float, fb: float, tol: float) -> float:
    """Root of f bracketed in [a, b] (fa*fb <= 0), converged to ~tol in t."""
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    c, fc = a, fa
    d = e = b - a
    for _ in range(100):
        if (fb > 0.0) == (fc > 0.0):
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2.0 * _EPS * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0.0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                rq = fb / fc
                p = s * (2.0 * xm * q * (q - rq) - (b - a) * (rq - 1.0))
                q = (q - 1.0) * (rq - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = xm
        else:
            d = e = xm
        a, fa = b, fb
        if abs(d) > tol1:
            b += d
        else:
            b += tol1 if xm > 0.0 else -tol1
        fb = f(b)
    return b


@dataclass(frozen=True)
class EventSpec:
    """Zero-crossing event g(t, r, v) == 0.

    direction: +1 rising, -1 falling, 0 both -- always with respect to
    increasing time, independent of the propagation direction.  A terminal
    event truncates the propagation at the event epoch.
    """
    name: str
    g: EventFn
    direction: int = 0
    terminal: bool = False


@dataclass(frozen=True)
class OrbitEvent:
    name: str
    epoch_tai_s: float
    state: CartesianState
    direction: int
    value: float


def ascending_node(terminal: bool = False) -> EventSpec:
    return EventSpec('ascending_node', lambda t, r, v: r.z, +1, terminal)


def descending_node(terminal: bool = False) -> EventSpec:
    return EventSpec('descending_node', lambda t, r, v: r.z, -1, terminal)


def perigee(terminal: bool = False) -> EventSpec:
    return EventSpec('perigee', lambda t, r, v: r.dot(v), +1, terminal)


def apogee(terminal: bool = False) -> EventSpec:
    return EventSpec('apogee', lambda t, r, v: r.dot(v), -1, terminal)


def altitude_crossing(alt_km: float, direction: int = 0, terminal: bool = False) -> EventSpec:
    target = R_EARTH_EQUATOR_KM + alt_km
    return EventSpec(f'altitude_{alt_km:g}km',
                     lambda t, r, v: r.norm() - target, direction, terminal)


def _direction(spec: EventSpec, step: _DenseStep, t_star: float, t_tol: float) -> int:
    lo, hi = min(step.t0, step.t1), max(step.t0, step.t1)
    delta = max(1e-4 * (hi - lo), 10.0 * t_tol)
    ta = max(lo, t_star - delta)
    tb = min(hi, t_star + delta)
    ga = spec.g(ta, *step.state_at(ta))
    gb = spec.g(tb, *step.state_at(tb))
    return 1 if gb >= ga else -1


def propagate_cowell(state: CartesianState, dt_s: float, step_s: float = 300.0,
                     tol: float = 1e-10, accel: Optional[Accel] = None,
                     h_min: float = 1e-9) -> CartesianState:
    """Adaptive Cowell propagation with local error control (Dormand-Prince 5(4)).

    step_s is the maximum step magnitude; the integrator picks smaller steps
    to meet tol.  Works forwards (dt_s > 0) and backwards (dt_s < 0).
    """
    accel = accel or _j2
    final = state
    for step in _steps(state, dt_s, abs(step_s), tol, accel, h_min):
        final = CartesianState(step.t1, step.r1, step.v1, state.frame)
    return final


def propagate_cowell_events(state: CartesianState, dt_s: float,
                            events: Sequence[EventSpec], step_s: float = 300.0,
                            tol: float = 1e-10, accel: Optional[Accel] = None,
                            t_tol: float = 1e-6, h_min: float = 1e-9):
    """Propagate and locate all requested events in a single pass.

    Events are bracketed inside accepted steps and refined with Brent's
    method on the step's dense output, so event epochs are not constrained
    to step boundaries.  Directions are reported with respect to increasing
    time and the returned list is sorted by epoch, giving consistent
    results for forward and backward propagation of the same arc.  A
    terminal event truncates the propagation at the event epoch.
    """
    accel = accel or _j2
    specs = list(events)
    sign = 1.0 if dt_s >= 0.0 else -1.0
    found: list[OrbitEvent] = []
    g_prev = [sp.g(state.epoch_tai_s, state.position_km, state.velocity_km_s)
              for sp in specs]
    first = True
    final = state
    for step in _steps(state, dt_s, abs(step_s), tol, accel, h_min):
        hits: list[tuple[EventSpec, OrbitEvent]] = []
        for idx, sp in enumerate(specs):
            g0 = g_prev[idx]
            g1 = sp.g(step.t1, step.r1, step.v1)
            roots = []
            if first and g0 == 0.0:
                roots.append(step.t0)
            if g0 != 0.0 and (g1 == 0.0 or g0 * g1 < 0.0):
                roots.append(_brent(lambda tt: sp.g(tt, *step.state_at(tt)),
                                    step.t0, step.t1, g0, g1, t_tol))
            for t_star in roots:
                d = _direction(sp, step, t_star, t_tol)
                if sp.direction and sp.direction != d:
                    continue
                r_ev, v_ev = step.state_at(t_star)
                hits.append((sp, OrbitEvent(sp.name, t_star,
                                            CartesianState(t_star, r_ev, v_ev, state.frame),
                                            d, sp.g(t_star, r_ev, v_ev))))
            g_prev[idx] = g1
        first = False
        terminal = [(sp, h) for sp, h in hits if sp.terminal]
        if terminal:
            cut_sp, cut = min(terminal, key=lambda sh: (sh[1].epoch_tai_s - step.t0) * sign)
            found.extend(h for _, h in hits
                         if (h.epoch_tai_s - cut.epoch_tai_s) * sign <= 0.0)
            found.sort(key=lambda e: e.epoch_tai_s)
            return cut.state, found
        found.extend(h for _, h in hits)
        final = CartesianState(step.t1, step.r1, step.v1, state.frame)
    found.sort(key=lambda e: e.epoch_tai_s)
    return final, found

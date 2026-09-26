from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
from orbitforge.core.state import CartesianState
from orbitforge.core.vector import Vec3
from orbitforge.core.constants import R_EARTH_EQUATOR_KM
from .perturbations import gravity_j2

# Dormand-Prince DOPRI5(4) tableau (FSAL: 7th stage = next step's 1st).
_A = (
    (),
    (1.0 / 5.0,),
    (3.0 / 40.0, 9.0 / 40.0),
    (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
    (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
    (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0),
    (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0),
)
_B5 = (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0)
_B4 = (5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0,
       -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0)

_SAFETY = 0.9
_FACTOR_MIN = 0.2
_FACTOR_MAX = 5.0
_MIN_STEP_S = 1e-9
_ROOT_TOL_S = 1e-9
_ROOT_ITERS = 80
_EVENT_TIME_EPS = 1e-7

AccelFn = Callable[[Vec3], Vec3]
EventFn = Callable[[Vec3, Vec3], float]


@dataclass(frozen=True)
class EventSpec:
    """g(r, v) = 0 defines an event.

    direction filters crossings in physical time: +1 = g goes negative ->
    positive, -1 = positive -> negative, 0 = either direction (tangencies
    are reported with direction 0).
    """
    name: str
    g: EventFn
    direction: int = 0


@dataclass(frozen=True)
class OrbitEvent:
    name: str
    t_tai_s: float
    value: float
    direction: int
    state: CartesianState


@dataclass(frozen=True)
class CowellResult:
    state: CartesianState
    events: tuple[OrbitEvent, ...]
    steps_accepted: int
    steps_rejected: int


def event_ascending_node(direction: int = 1) -> EventSpec:
    # z = 0 crossing; ascending = z increasing in physical time.
    return EventSpec('ascending_node', lambda r, v: r.z, direction)


def event_descending_node() -> EventSpec:
    return EventSpec('descending_node', lambda r, v: r.z, -1)


def event_periapsis() -> EventSpec:
    # r.v = 0 going negative -> positive is periapsis.
    return EventSpec('periapsis', lambda r, v: r.dot(v), 1)


def event_apoapsis() -> EventSpec:
    # r.v = 0 going positive -> negative is apoapsis.
    return EventSpec('apoapsis', lambda r, v: r.dot(v), -1)


def event_altitude_threshold(alt_km: float, direction: int = 0,
                             radius_km: float = R_EARTH_EQUATOR_KM) -> EventSpec:
    target = radius_km + alt_km
    return EventSpec(f'altitude_{alt_km:g}_km',
                     lambda r, v: r.norm() - target, direction)


def _deriv(y: tuple[float, ...], accel: AccelFn) -> tuple[float, ...]:
    a = accel(Vec3(y[0], y[1], y[2]))
    return (y[3], y[4], y[5], a.x, a.y, a.z)


def _stage(y, h, k, row):
    out = list(y)
    for aij, ki in zip(row, k):
        if aij != 0.0:
            for j in range(6):
                out[j] += h * aij * ki[j]
    return tuple(out)


def _short_integrate(y0, ds, accel, rtol=1e-11, atol=1e-11):
    """Adaptive DOPRI5 over a short (sub-step) signed interval.

    Bisection probes sit inside one already accepted step; taking a single
    full step there would re-introduce that step's O(h^6) local error into
    every g() evaluation, which biases the root. Integrating adaptively to
    a tighter tolerance instead makes the probe values consistent with the
    main trajectory to well below its own error.
    """
    if ds == 0.0:
        return y0
    direction = 1.0 if ds > 0 else -1.0
    y = y0
    remain = ds
    h = direction * min(abs(ds), max(abs(ds) / 8.0, 1e-6))
    while abs(remain) > 1e-13:
        if abs(h) > abs(remain):
            h = remain
        k = [_deriv(y, accel)]
        for row in _A[1:]:
            k.append(_deriv(_stage(y, h, k, row), accel))
        y5 = tuple(y[i] + h * sum(_B5[m] * k[m][i] for m in range(7)) for i in range(6))
        y4 = tuple(y[i] + h * sum(_B4[m] * k[m][i] for m in range(7)) for i in range(6))
        err = _error_norm(y5, y4, y, atol, rtol)
        if err <= 1.0:
            y = y5
            remain -= h
            factor = _FACTOR_MAX if err == 0.0 else min(
                _FACTOR_MAX, max(_FACTOR_MIN, _SAFETY * err ** -0.2))
            h *= factor
        else:
            h *= max(_FACTOR_MIN, _SAFETY * err ** -0.2)
    return y


def _error_norm(ynew, yhat, yold, atol, rtol):
    total = 0.0
    for i in range(6):
        scale = atol + rtol * max(abs(yold[i]), abs(ynew[i]))
        e = (ynew[i] - yhat[i]) / scale
        total += e * e
    return (total / 6.0) ** 0.5


def _initial_step(y, f0, accel, atol, rtol, direction):
    """Hairer/Wanner/Wittum automatic initial step selection."""
    sc = tuple(atol + rtol * abs(v) for v in y)
    d0 = sum((y[i] / sc[i]) ** 2 for i in range(6)) ** 0.5
    d1 = sum((f0[i] / sc[i]) ** 2 for i in range(6)) ** 0.5
    h0 = 1e-6 if d0 < 1e-5 or d1 < 1e-5 else 0.01 * d0 / d1
    y1 = tuple(y[i] + h0 * f0[i] for i in range(6))
    f1 = _deriv(y1, accel)
    d2 = sum(((f1[i] - f0[i]) / sc[i]) ** 2 for i in range(6)) ** 0.5 / h0
    if max(d1, d2) <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** 0.25
    return direction * min(100.0 * h0, h1)


def _state_from_y(t, y, frame):
    return CartesianState(t, Vec3(y[0], y[1], y[2]),
                          Vec3(y[3], y[4], y[5]), frame)


def interpolate_step(state0: CartesianState, state1: CartesianState,
                     theta: float, accel_fn: AccelFn | None = None) -> CartesianState:
    """Cubic Hermite state at fraction theta in [0, 1] of one step.

    C1-continuous and identical for forward and backward steps; convenient
    for dense sampling/plotting. Event root finding does not rely on this
    cubic (its O(h^3) interior error can mis-bracket roots) and instead
    re-integrates, so prefer propagate_cowell events for precise timing.
    """
    if not 0.0 <= theta <= 1.0:
        raise ValueError('theta must be in [0, 1]')
    accel = accel_fn or gravity_j2
    y0 = (state0.position_km.x, state0.position_km.y, state0.position_km.z,
          state0.velocity_km_s.x, state0.velocity_km_s.y, state0.velocity_km_s.z)
    y1 = (state1.position_km.x, state1.position_km.y, state1.position_km.z,
          state1.velocity_km_s.x, state1.velocity_km_s.y, state1.velocity_km_s.z)
    h = state1.epoch_tai_s - state0.epoch_tai_s
    f0 = _deriv(y0, accel)
    f1 = _deriv(y1, accel)
    u = 1.0 - theta
    yy = tuple(
        u * y0[i] + theta * y1[i]
        + theta * u * ((1.0 - 2.0 * theta) * (y1[i] - y0[i])
                       + (theta - 1.0) * h * f0[i]
                       + theta * h * f1[i])
        for i in range(6))
    return _state_from_y(state0.epoch_tai_s + h * theta, yy, state0.frame)


def _g_value(y, spec):
    return spec.g(Vec3(y[0], y[1], y[2]), Vec3(y[3], y[4], y[5]))


def _detect_events(specs, t0, h, y0, y1, accel, frame,
                   include_origin, prior):
    """Locate every spec event inside one accepted step.

    Event functions are sampled at theta = 0, .25, .5, .75, 1 on the true
    flow (short adaptive integrations from the step origin); the dense
    Hermite cubic alone is not accurate enough to guarantee that its
    sign-change brackets contain the real root. Quarter spacing keeps even
    two roots in a single step from hiding. Sign changes are bisected on
    the true flow to ~1 ns, so event times are independent of the accepted
    step grid. Zero-valued probes are handled as intervals so exact-root
    steps are not double-counted. Direction is measured in physical time,
    so forward and backward steps behave identically.
    """
    hsign = 1.0 if h > 0 else -1.0
    thetas = (0.0, 0.25, 0.5, 0.75, 1.0)
    # One set of true-flow probe states per step, shared by every spec.
    probes = [y0,
              _short_integrate(y0, h * 0.25, accel),
              _short_integrate(y0, h * 0.5, accel),
              _short_integrate(y0, h * 0.75, accel),
              y1]
    out = []

    def already_reported(tev, name):
        return any(abs(e.t_tai_s - tev) < _EVENT_TIME_EPS and e.name == name
                   for e in (*prior, *out))

    def refine(spec, j_lo, gs):
        """Bisect a genuine sign bracket between probes j_lo and j_lo+1."""
        lo, hi = h * thetas[j_lo], h * thetas[j_lo + 1]
        flo, fhi = gs[j_lo], gs[j_lo + 1]
        for _ in range(_ROOT_ITERS):
            mid = 0.5 * (lo + hi)
            fm = _g_value(_short_integrate(y0, mid, accel), spec)
            if flo * fm <= 0.0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
            if abs(hi - lo) < _ROOT_TOL_S:
                break
        ds = 0.5 * (lo + hi)
        yy = _short_integrate(y0, ds, accel)
        t = t0 + ds
        direction = hsign * (1 if fhi > flo else -1)
        return OrbitEvent(spec.name, t, _g_value(yy, spec), int(direction),
                          _state_from_y(t, yy, frame))

    for spec in specs:
        gs = [_g_value(y, spec) for y in probes]
        j, n = 0, len(thetas)
        while j < n:
            event = None
            if gs[j] == 0.0:
                j0 = j
                while j < n and gs[j] == 0.0:
                    j += 1
                if j0 == 0 and not include_origin:
                    continue  # reported as the previous step's end point
                gl = gs[j0 - 1] if j0 > 0 else None
                gr = gs[j] if j < n else None
                if gl is not None and gr is not None:
                    crossing = gl * gr < 0.0
                    direction = hsign * (1 if gr > gl else -1)
                elif gl is not None:
                    crossing, direction = True, hsign * (1 if gl < 0.0 else -1)
                elif gr is not None:
                    crossing, direction = True, hsign * (1 if gr > 0.0 else -1)
                else:
                    crossing, direction = True, 0
                theta = 0.5 * (thetas[j0] + thetas[j - 1])
                if crossing or direction == 0:
                    ds = h * theta
                    yy = _short_integrate(y0, ds, accel)
                    event = OrbitEvent(spec.name, t0 + ds, _g_value(yy, spec),
                                       int(direction),
                                       _state_from_y(t0 + ds, yy, frame))
            else:
                if j + 1 < n and gs[j + 1] != 0.0 and gs[j] * gs[j + 1] < 0.0:
                    event = refine(spec, j, gs)
                j += 1

            if event is None:
                continue
            if spec.direction != 0 and spec.direction != event.direction:
                continue
            if not already_reported(event.t_tai_s, event.name):
                out.append(event)
    return out


def propagate_cowell(state: CartesianState, dt_s: float, step_s: float | None = 20.0, *,
                     rtol: float = 1e-9, atol: float = 1e-9,
                     events: Sequence[EventSpec] | None = None,
                     stop_at_event: bool = False,
                     max_step_s: float | None = None,
                     accel_fn: AccelFn | None = None):
    """Adaptive Cowell propagation with DOPRI5(4) local error control.

    step_s is only the initial trial step (s); None selects one
    automatically. Every subsequent step shrinks/grows to meet the
    rtol/atol scaled error norm. Forward (dt_s > 0) and backward
    (dt_s < 0) integration share the same dense output and root
    refinement, so event times are identical regardless of propagation
    direction and are not quantized to integration steps.

    With events (or stop_at_event) returns a CowellResult; otherwise a
    CartesianState, matching the original interface.
    """
    accel = accel_fn or gravity_j2
    specs = tuple(events) if events else ()
    want_result = bool(specs) or stop_at_event

    y = (state.position_km.x, state.position_km.y, state.position_km.z,
         state.velocity_km_s.x, state.velocity_km_s.y, state.velocity_km_s.z)
    t = state.epoch_tai_s
    remain = dt_s
    direction = 1 if dt_s >= 0 else -1

    f0 = _deriv(y, accel)
    if step_s is None:
        h = _initial_step(y, f0, accel, atol, rtol, direction)
    else:
        h = direction * abs(step_s)
    if dt_s != 0.0 and abs(h) > abs(dt_s):
        h = dt_s

    found: list[OrbitEvent] = []
    accepted = 0
    rejected = 0
    is_first_step = True

    while abs(remain) > 1e-12:
        if abs(h) > abs(remain):
            h = remain
        if abs(h) < _MIN_STEP_S:
            raise RuntimeError(f'cowell step below {_MIN_STEP_S}s; tolerance unreachable')

        k = [f0]
        for row in _A[1:]:
            k.append(_deriv(_stage(y, h, k, row), accel))
        y5 = tuple(y[i] + h * sum(_B5[m] * k[m][i] for m in range(7)) for i in range(6))
        y4 = tuple(y[i] + h * sum(_B4[m] * k[m][i] for m in range(7)) for i in range(6))
        err = _error_norm(y5, y4, y, atol, rtol)

        if err <= 1.0:
            t_new = t + h
            f_new = k[6]  # FSAL: k7 is the derivative at y5
            step_events = _detect_events(
                specs, t, h, y, y5, accel, state.frame,
                is_first_step, found) if specs else []
            accepted += 1
            t, y = t_new, y5
            remain -= h
            f0 = f_new
            is_first_step = False

            stop_event = None
            if step_events and stop_at_event:
                # First event along the direction of travel.
                stop_event = (min(step_events, key=lambda e: e.t_tai_s) if direction > 0
                              else max(step_events, key=lambda e: e.t_tai_s))
                if direction > 0:
                    step_events = [e for e in step_events
                                   if e.t_tai_s <= stop_event.t_tai_s + _EVENT_TIME_EPS]
                else:
                    step_events = [e for e in step_events
                                   if e.t_tai_s >= stop_event.t_tai_s - _EVENT_TIME_EPS]
            if step_events:
                found.extend(step_events)
            if stop_event is not None:
                t = stop_event.t_tai_s
                sy = stop_event.state
                y = (sy.position_km.x, sy.position_km.y, sy.position_km.z,
                     sy.velocity_km_s.x, sy.velocity_km_s.y, sy.velocity_km_s.z)
                remain = 0.0
                break

            factor = _FACTOR_MAX if err == 0.0 else min(
                _FACTOR_MAX, max(_FACTOR_MIN, _SAFETY * err ** -0.2))
            h *= factor
            if max_step_s is not None:
                h = direction * min(abs(h), abs(max_step_s))
        else:
            rejected += 1
            h *= max(_FACTOR_MIN, _SAFETY * err ** -0.2)

    final_state = _state_from_y(t, y, state.frame)
    if not want_result:
        return final_state

    found.sort(key=lambda e: e.t_tai_s)
    return CowellResult(final_state, tuple(found), accepted, rejected)

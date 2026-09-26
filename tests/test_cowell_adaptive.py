import math
from orbitforge.core.state import CartesianState
from orbitforge.core.vector import Vec3
from orbitforge.core.constants import MU_EARTH_KM3_S2, R_EARTH_EQUATOR_KM
from orbitforge.orbits.elements import KeplerianElements, elements_to_state
from orbitforge.orbits.periods import orbital_period_s
from orbitforge.orbits.universal import propagate_universal
from orbitforge.orbits import cowell


def _two_body(r):
    d = r.norm()
    return r * (-MU_EARTH_KM3_S2 / d ** 3)


def _fixture(e=0.05, inc_deg=30.0, nu=2.0):
    a = 7000.0
    r, v = elements_to_state(
        KeplerianElements(a, e, math.radians(inc_deg), 0.0, 0.0, nu))
    return CartesianState(0.0, r, v), orbital_period_s(a)


def _exact_state(s0, t):
    r, v = propagate_universal(s0.position_km, s0.velocity_km_s, t)
    return CartesianState(t, r, v, s0.frame)


def _exact_root(s0, g, lo, hi):
    flo = g(_exact_state(s0, lo))
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fm = g(_exact_state(s0, mid))
        if hi - lo < 1e-11:
            break
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def test_backward_compatible_interface():
    s0, T = _fixture()
    out = cowell.propagate_cowell(s0, 600.0, accel_fn=_two_body)
    assert isinstance(out, CartesianState)
    ref = _exact_state(s0, 600.0)
    assert (out.position_km - ref.position_km).norm() < 1e-3


def test_adaptive_local_error_control_converges():
    s0, T = _fixture()
    errors = []
    for tol in (1e-6, 1e-9, 1e-12):
        out = cowell.propagate_cowell(s0, T, accel_fn=_two_body,
                                      rtol=tol, atol=tol, step_s=None)
        errors.append((out.position_km - s0.position_km).norm())
    assert errors[0] > 1e-2
    assert errors[1] < 1e-3
    assert errors[2] < 1e-5
    assert errors[1] < errors[0] / 100
    assert errors[2] < errors[1] / 100


def test_one_period_accuracy():
    s0, T = _fixture()
    out = cowell.propagate_cowell(s0, T, accel_fn=_two_body)
    assert (out.position_km - s0.position_km).norm() < 1e-3


def test_event_times_match_exact_two_body():
    s0, T = _fixture()
    alt = R_EARTH_EQUATOR_KM + 600.0
    specs = [cowell.event_ascending_node(),
             cowell.event_periapsis(),
             cowell.event_altitude_threshold(600.0)]
    res = cowell.propagate_cowell(s0, T, events=specs, accel_fn=_two_body,
                                  rtol=1e-10, atol=1e-10)
    refs = {
        'ascending_node': _exact_root(s0, lambda s: s.position_km.z, 0.6 * T, 0.8 * T),
        'periapsis': _exact_root(s0, lambda s: s.position_km.dot(s.velocity_km_s),
                                 0.6 * T, 0.8 * T),
        'altitude_600_km': _exact_root(s0, lambda s: s.position_km.norm() - alt,
                                       0.4 * T, 0.6 * T),
    }
    # ascending node and periapsis coincide in this geometry, plus two
    # altitude crossings: expect each reference event to appear at least once
    assert len(res.events) >= 3
    found = {}
    for ev in res.events:
        if ev.name in refs and ev.name not in found:
            found[ev.name] = ev
    assert set(found) == set(refs)
    for name, ev in found.items():
        assert abs(ev.t_tai_s - refs[name]) < 1e-3
        assert abs(ev.value) < 1e-3


def test_all_event_types_over_a_period():
    s0, T = _fixture()
    specs = [cowell.event_ascending_node(), cowell.event_descending_node(),
             cowell.event_periapsis(), cowell.event_apoapsis(),
             cowell.event_altitude_threshold(600.0)]
    res = cowell.propagate_cowell(s0, T, events=specs, accel_fn=_two_body,
                                  rtol=1e-10, atol=1e-10)
    names = [ev.name for ev in res.events]
    assert names.count('ascending_node') == 1
    assert names.count('descending_node') == 1
    assert names.count('periapsis') == 1
    assert names.count('apoapsis') == 1
    assert names.count('altitude_600_km') == 2
    by_name = {ev.name: ev for ev in res.events}
    assert by_name['ascending_node'].direction == 1
    assert by_name['descending_node'].direction == -1
    assert by_name['periapsis'].direction == 1
    assert by_name['apoapsis'].direction == -1
    # event state is the instantaneous state at the crossing
    per = by_name['periapsis']
    assert abs(per.state.position_km.dot(per.state.velocity_km_s)) < 1e-6


def test_event_times_not_quantized_to_step_grid():
    s0, T = _fixture()
    specs = [cowell.event_ascending_node(),
             cowell.event_altitude_threshold(600.0)]
    times = []
    for step in (0.5, 20.0, 500.0):
        res = cowell.propagate_cowell(s0, T, step_s=step, events=specs,
                                      accel_fn=_two_body, rtol=1e-10, atol=1e-10)
        times.append([round(e.t_tai_s, 6) for e in res.events])
    for other in times[1:]:
        for a, b in zip(times[0], other):
            assert abs(a - b) < 1e-3
    # and never land exactly on a step boundary grid
    res = cowell.propagate_cowell(s0, T, step_s=20.0, events=specs,
                                  accel_fn=_two_body)
    for ev in res.events:
        assert abs(ev.t_tai_s / 20.0 - round(ev.t_tai_s / 20.0)) > 1e-3


def test_event_times_forward_and_backward_consistent():
    s0, T = _fixture()
    specs = [cowell.event_ascending_node(), cowell.event_descending_node(),
             cowell.event_periapsis(), cowell.event_apoapsis(),
             cowell.event_altitude_threshold(600.0)]
    fwd = cowell.propagate_cowell(s0, T, events=specs, accel_fn=_two_body,
                                  rtol=1e-11, atol=1e-11)
    bwd = cowell.propagate_cowell(fwd.state, -T, events=specs,
                                  accel_fn=_two_body, rtol=1e-11, atol=1e-11)
    for fe in fwd.events:
        match = [be for be in bwd.events
                 if be.name == fe.name and be.direction == fe.direction]
        assert match, f'missing {fe.name} backwards'
        assert min(abs(be.t_tai_s - fe.t_tai_s) for be in match) < 1e-3
    # backward propagation returns to the origin
    assert (bwd.state.position_km - s0.position_km).norm() < 1e-4


def test_direction_filter():
    s1, T1 = _fixture()
    ups = cowell.propagate_cowell(
        s1, T1, events=[cowell.event_altitude_threshold(600.0, direction=1)],
        accel_fn=_two_body)
    downs = cowell.propagate_cowell(
        s1, T1, events=[cowell.event_altitude_threshold(600.0, direction=-1)],
        accel_fn=_two_body)
    assert len(ups.events) == 1 and len(downs.events) == 1
    assert ups.events[0].direction == 1 and downs.events[0].direction == -1
    assert ups.events[0].t_tai_s > downs.events[0].t_tai_s


def test_stop_at_first_event():
    s0, T = _fixture()
    spec = cowell.event_ascending_node()
    fwd = cowell.propagate_cowell(s0, T, events=[spec], stop_at_event=True,
                                  accel_fn=_two_body, rtol=1e-11, atol=1e-11)
    assert abs(fwd.state.position_km.z) < 1e-6
    bwd = cowell.propagate_cowell(_exact_state(s0, T), -T, events=[spec],
                                  stop_at_event=True, accel_fn=_two_body,
                                  rtol=1e-11, atol=1e-11)
    assert abs(fwd.state.epoch_tai_s - bwd.state.epoch_tai_s) < 1e-4
    # the stopped state is the same event, reached from either direction
    assert abs(fwd.state.position_km.z - bwd.state.position_km.z) < 1e-6


def test_boundary_event_reported_once():
    # state starting exactly on the ascending node
    a = 7000.0
    r, v = elements_to_state(
        KeplerianElements(a, 0.0, math.radians(40), 0.0, 0.0, 0.0))
    s0 = CartesianState(0.0, r, v)
    T = orbital_period_s(a)
    res = cowell.propagate_cowell(
        s0, T, events=[cowell.event_ascending_node(),
                       cowell.event_descending_node()],
        accel_fn=_two_body, rtol=1e-11, atol=1e-11)
    asc = [e for e in res.events if e.name == 'ascending_node']
    desc = [e for e in res.events if e.name == 'descending_node']
    assert len(asc) == 1 and abs(asc[0].t_tai_s) < 1e-9
    assert len(desc) == 1
    assert 0.0 < desc[0].t_tai_s < T


def test_step_rejections_happen_and_recover():
    # absurd initial guess forces rejection before the first acceptance
    s0, T = _fixture()
    res = cowell.propagate_cowell(s0, T, step_s=1.0e6,
                                  events=[cowell.event_periapsis()],
                                  accel_fn=_two_body)
    assert res.steps_rejected > 0
    assert res.steps_accepted > 10
    assert (res.state.position_km - _exact_state(s0, T).position_km).norm() < 1e-2


def test_j2_events_forward_backward():
    s0, T = _fixture()
    specs = [cowell.event_ascending_node(), cowell.event_periapsis(),
             cowell.event_altitude_threshold(400.0)]
    fwd = cowell.propagate_cowell(s0, T, events=specs, rtol=1e-11, atol=1e-11)
    bwd = cowell.propagate_cowell(fwd.state, -T, events=specs,
                                  rtol=1e-11, atol=1e-11)
    for fe in fwd.events:
        match = [be for be in bwd.events
                 if be.name == fe.name and be.direction == fe.direction]
        assert match
        assert min(abs(be.t_tai_s - fe.t_tai_s) for be in match) < 1e-3


def test_interpolate_step_endpoints():
    s0, T = _fixture()
    s1 = cowell.propagate_cowell(s0, 60.0, accel_fn=_two_body)
    a = cowell.interpolate_step(s0, s1, 0.0, accel_fn=_two_body)
    b = cowell.interpolate_step(s0, s1, 1.0, accel_fn=_two_body)
    assert (a.position_km - s0.position_km).norm() < 1e-12
    assert (b.position_km - s1.position_km).norm() < 1e-12
    mid = cowell.interpolate_step(s0, s1, 0.5, accel_fn=_two_body)
    exact = _exact_state(s0, 30.0)
    # cubic Hermite is a local interpolant (O(h^3)); a few km over a 60 s
    # LEO step is expected; precise timing uses event re-integration instead
    assert (mid.position_km - exact.position_km).norm() < 20.0


def test_zero_and_negative_dt():
    s0, _ = _fixture()
    assert cowell.propagate_cowell(s0, 0.0) == s0
    fwd = cowell.propagate_cowell(s0, 300.0, accel_fn=_two_body)
    back = cowell.propagate_cowell(fwd, -300.0, accel_fn=_two_body,
                                   rtol=1e-11, atol=1e-11)
    assert (back.position_km - s0.position_km).norm() < 1e-5

import math

from orbitforge.core.constants import MU_EARTH_KM3_S2, R_EARTH_EQUATOR_KM
from orbitforge.core.state import CartesianState
from orbitforge.core.vector import Vec3
from orbitforge.orbits.cowell import (
    apogee, ascending_node, altitude_crossing, perigee,
    propagate_cowell, propagate_cowell_events,
)
from orbitforge.orbits.elements import KeplerianElements, elements_to_state
from orbitforge.orbits.universal import propagate_universal


def _kepler_accel(r: Vec3, v: Vec3) -> Vec3:
    return r * (-MU_EARTH_KM3_S2 / r.norm() ** 3)


def _state(a, e, i, raan=0.4, argp=0.5, nu=1.0):
    r, v = elements_to_state(KeplerianElements(a, e, i, raan, argp, nu))
    return CartesianState(0.0, r, v)


def _period(a):
    return 2.0 * math.pi * math.sqrt(a ** 3 / MU_EARTH_KM3_S2)


def test_adaptive_matches_two_body_solution():
    s0 = _state(7000.0, 0.02, 0.6)
    dt = 1.5 * _period(7000.0)
    out = propagate_cowell(s0, dt, tol=1e-12, accel=_kepler_accel)
    r_ref, v_ref = propagate_universal(s0.position_km, s0.velocity_km_s, dt)
    assert (out.position_km - r_ref).norm() < 1e-6
    assert (out.velocity_km_s - v_ref).norm() < 1e-9
    assert abs(out.epoch_tai_s - dt) < 1e-9


def test_backward_propagation_recovers_initial_state():
    s0 = _state(8000.0, 0.1, 0.3)
    dt = 2.0 * _period(8000.0)
    fwd = propagate_cowell(s0, dt, tol=1e-12)
    back = propagate_cowell(fwd, -dt, tol=1e-12)
    assert (back.position_km - s0.position_km).norm() < 1e-5
    assert (back.velocity_km_s - s0.velocity_km_s).norm() < 1e-8
    assert abs(back.epoch_tai_s) < 1e-9


def test_ascending_nodes_refined_off_step_grid():
    a = 7000.0
    s0 = _state(a, 0.01, 0.5)
    dt = 3.5 * _period(a)
    _, events = propagate_cowell_events(s0, dt, [ascending_node()],
                                        step_s=2000.0, tol=1e-12)
    assert len(events) == 3
    for ev in events:
        assert ev.direction == 1
        assert abs(ev.state.position_km.z) < 1e-6
        assert ev.state.velocity_km_s.z > 0.0
        # event epochs are refined, not pinned to multiples of the max step
        rem = ev.epoch_tai_s % 2000.0
        assert min(rem, 2000.0 - rem) > 1e-3
    gaps = [b.epoch_tai_s - a_.epoch_tai_s for a_, b in zip(events, events[1:])]
    for gap in gaps:
        assert abs(gap - _period(a)) < 0.02 * _period(a)


def test_apsides_events():
    a, e = 10000.0, 0.3
    s0 = _state(a, e, 0.3)
    dt = 2.2 * _period(a)
    _, events = propagate_cowell_events(s0, dt, [perigee(), apogee()], tol=1e-12)
    per = [ev for ev in events if ev.name == 'perigee']
    apo = [ev for ev in events if ev.name == 'apogee']
    assert len(per) == 2 and len(apo) == 2
    for ev in per + apo:
        assert abs(ev.state.position_km.dot(ev.state.velocity_km_s)) < 1e-3
    for ev in per:
        assert ev.direction == 1
        assert abs(ev.state.position_km.norm() - a * (1 - e)) < 0.02 * a
    for ev in apo:
        assert ev.direction == -1
        assert abs(ev.state.position_km.norm() - a * (1 + e)) < 0.02 * a


def test_altitude_threshold_crossings():
    a, e = 8000.0, 0.1
    s0 = _state(a, e, 0.3)
    dt = 2.2 * _period(a)
    _, events = propagate_cowell_events(s0, dt, [altitude_crossing(1000.0)],
                                        tol=1e-12)
    assert len(events) == 4
    target = R_EARTH_EQUATOR_KM + 1000.0
    for ev in events:
        assert abs(ev.state.position_km.norm() - target) < 1e-6
    dirs = [ev.direction for ev in events]
    assert dirs in ([1, -1, 1, -1], [-1, 1, -1, 1])


def test_event_times_independent_of_step_size():
    s0 = _state(7000.0, 0.05, 0.5)
    dt = 2.2 * _period(7000.0)
    specs = [ascending_node(), perigee(), apogee()]
    _, e1 = propagate_cowell_events(s0, dt, specs, step_s=137.0, tol=1e-12)
    _, e2 = propagate_cowell_events(s0, dt, specs, step_s=450.0, tol=1e-12)
    assert [e.name for e in e1] == [e.name for e in e2]
    for a_, b_ in zip(e1, e2):
        assert a_.direction == b_.direction
        assert abs(a_.epoch_tai_s - b_.epoch_tai_s) < 1e-3


def test_forward_backward_events_consistent():
    s0 = _state(8000.0, 0.1, 0.5)
    dt = 2.2 * _period(8000.0)
    specs = [ascending_node(), perigee(), apogee(), altitude_crossing(1000.0)]
    fwd, e_fwd = propagate_cowell_events(s0, dt, specs, tol=1e-12)
    _, e_back = propagate_cowell_events(fwd, -dt, specs, tol=1e-12)
    assert [e.name for e in e_fwd] == [e.name for e in e_back]
    assert [e.direction for e in e_fwd] == [e.direction for e in e_back]
    for a_, b_ in zip(e_fwd, e_back):
        assert abs(a_.epoch_tai_s - b_.epoch_tai_s) < 1e-3


def test_terminal_event_stops_propagation():
    a, e = 8000.0, 0.3  # perigee 5600 km, below the surface
    s0 = _state(a, e, 0.3, nu=math.pi)
    dt = _period(a)
    final, events = propagate_cowell_events(
        s0, dt, [altitude_crossing(0.0, direction=-1, terminal=True)], tol=1e-12)
    assert len(events) == 1
    assert events[0].direction == -1
    assert abs(final.position_km.norm() - R_EARTH_EQUATOR_KM) < 1e-6
    assert final.epoch_tai_s == events[0].epoch_tai_s
    assert 0.0 < final.epoch_tai_s < dt

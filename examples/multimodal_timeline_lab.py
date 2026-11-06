"""Offline multimodal event-timeline alignment with evidence provenance.

All clips are synthetic and no media is decoded. Time calibration, interval
joins, scope checks and contradiction detection are independently tested.
Run: python3 examples/multimodal_timeline_lab.py
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from statistics import median


@dataclass(frozen=True)
class ClockCalibration:
    sensor: str
    reference_at_zero_ms: float
    drift_ppm: float

    def global_time(self, local_ms: float):
        """Map a device clock to a common monotonic time domain."""
        if abs(self.drift_ppm) > 100000:
            raise ValueError("implausibly high drift")
        return (self.reference_at_zero_ms
                + local_ms * (1 + self.drift_ppm * 1e-6))


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    tenant: str
    sensor: str
    local_start: float
    local_end: float
    kind: str
    event: str
    confidence: float
    source_sha: str

    def validate(self):
        if not self.evidence_id or not self.sensor or not self.tenant:
            raise ValueError("evidence identity required")
        if self.local_start < 0 or self.local_end < self.local_start:
            raise ValueError("invalid evidence time interval")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence out of range")
        if not self.source_sha:
            raise ValueError("missing source digest")


@dataclass(frozen=True)
class Aligned:
    evidence: Evidence
    global_start: float
    global_end: float

    @property
    def duration(self):
        return self.global_end - self.global_start


@dataclass(frozen=True)
class Incident:
    start: float
    end: float
    evidence_ids: tuple[str, ...]
    categories: tuple[str, ...]
    event_names: tuple[str, ...]
    confidence_min: float


class Timeline:
    def __init__(self, calibration: list[ClockCalibration]):
        if len({c.sensor for c in calibration}) != len(calibration):
            raise ValueError("duplicate sensor calibration")
        self.calibration = {c.sensor: c for c in calibration}
        self.evidence: dict[str, Evidence] = {}
        self.retracted: set[str] = set()

    def ingest(self, item: Evidence):
        item.validate()
        if item.sensor not in self.calibration:
            raise ValueError("no clock calibration")
        existing = self.evidence.get(item.evidence_id)
        if existing is not None:
            if existing != item:
                raise ValueError("same ID with changed content")
            return
        self.evidence[item.evidence_id] = item

    def retract(self, evidence_id: str):
        if evidence_id not in self.evidence:
            raise KeyError("unknown evidence ID")
        self.retracted.add(evidence_id)

    def align(self, tenant: str, min_confidence=0.0):
        aligned = []
        for e in self.evidence.values():
            if e.tenant != tenant or e.evidence_id in self.retracted:
                continue
            if e.confidence < min_confidence:
                continue
            calibration = self.calibration[e.sensor]
            start = calibration.global_time(e.local_start)
            end = calibration.global_time(e.local_end)
            if end < start:
                raise AssertionError("time mapping inverted interval")
            aligned.append(Aligned(e, start, end))
        return sorted(
            aligned,
            key=lambda x: (x.global_start, x.global_end,
                           x.evidence.evidence_id)
        )

    def fuse(self, tenant: str, max_gap_ms=150.0,
             min_confidence=0.0):
        """Connected interval components, not a semantic event classifier."""
        if max_gap_ms < 0:
            raise ValueError("negative gap")
        events = self.align(tenant, min_confidence)
        if not events:
            return []
        clusters = []
        group = [events[0]]
        current_end = events[0].global_end
        for item in events[1:]:
            if item.global_start <= current_end + max_gap_ms:
                group.append(item)
                current_end = max(current_end, item.global_end)
            else:
                clusters.append(group)
                group = [item]
                current_end = item.global_end
        clusters.append(group)
        result = []
        for cluster in clusters:
            result.append(Incident(
                min(e.global_start for e in cluster),
                max(e.global_end for e in cluster),
                tuple(e.evidence.evidence_id for e in cluster),
                tuple(sorted({e.evidence.kind for e in cluster})),
                tuple(sorted({e.evidence.event for e in cluster})),
                min(e.evidence.confidence for e in cluster),
            ))
        return result

    def reference(self, tenant, evidence_id):
        e = self.evidence.get(evidence_id)
        if e is None or e.tenant != tenant or evidence_id in self.retracted:
            raise PermissionError("evidence unavailable")
        c = self.calibration[e.sensor]
        return {
            "evidence_id": e.evidence_id,
            "sensor": e.sensor,
            "source_sha": e.source_sha,
            "global_interval_ms": [
                c.global_time(e.local_start),
                c.global_time(e.local_end)
            ],
        }


def interval_overlap(a: Aligned, b: Aligned):
    return max(0.0, min(a.global_end, b.global_end)
               - max(a.global_start, b.global_start))


def intersection_over_union(a: Aligned, b: Aligned):
    intersection = interval_overlap(a, b)
    union = max(a.global_end, b.global_end) - min(
        a.global_start, b.global_start
    )
    return intersection/union if union > 0 else 0.0


def temporal_consistency(alignments):
    """Identify mutually exclusive tags during overlapping intervals.

    A deliberately simple pedagogical rule, not video/audio semantics.
    """
    conflicts = []
    for i, a in enumerate(alignments):
        for b in alignments[i+1:]:
            if (a.evidence.kind == b.evidence.kind
                    and interval_overlap(a, b) > 0
                    and a.evidence.event != b.evidence.event):
                conflicts.append((
                    a.evidence.evidence_id, b.evidence.evidence_id
                ))
    return conflicts


def sha(text: str):
    return sha256(text.encode("utf-8")).hexdigest()


def synthetic_timeline():
    timeline = Timeline([
        ClockCalibration("video-1", reference_at_zero_ms=100., drift_ppm=0),
        ClockCalibration("audio-1", reference_at_zero_ms=0., drift_ppm=100.),
        ClockCalibration("logs", reference_at_zero_ms=0., drift_ppm=0),
    ])
    records = [
        Evidence("vid-1", "A", "video-1", 900, 1100,
                 "visual", "door opens", 0.95, sha("video-frame-1")),
        Evidence("aud-1", "A", "audio-1", 1010, 1120,
                 "audio", "metal impact", 0.87, sha("audio-clip")),
        Evidence("log-1", "A", "logs", 1000, 1080,
                 "system", "door sensor", 0.99, sha("syslog")),
        Evidence("vid-2", "A", "video-1", 2600, 2800,
                 "visual", "door closes", 0.91, sha("video-frame-2")),
        Evidence("private", "B", "video-1", 900, 950,
                 "visual", "restricted tenant", 1., sha("tenant-b")),
        Evidence("weak", "A", "audio-1", 1010, 1090,
                 "audio", "uncertain noise", 0.1, sha("lowconfidence")),
    ]
    for record in records:
        timeline.ingest(record)
    return timeline


def expect(error, fn):
    try:
        fn()
    except error:
        return
    raise AssertionError("expected " + error.__name__)


def test_calibration():
    c = ClockCalibration("test", 100, 100)
    assert abs(c.global_time(100000) - 100110) < 1e-8
    t = synthetic_timeline()
    items = t.align("A", min_confidence=0.5)
    assert len(items) == 4
    assert set(x.evidence.evidence_id for x in items) == {
        "vid-1", "aud-1", "log-1", "vid-2"
    }
    assert items[0].global_start <= items[1].global_start
    print("PASS common clock alignment and tenant + confidence filters")


def test_fusion():
    t = synthetic_timeline()
    groups = t.fuse("A", max_gap_ms=150, min_confidence=0.5)
    assert len(groups) == 2, groups
    assert set(groups[0].evidence_ids) == {
        "vid-1", "aud-1", "log-1"
    }
    assert groups[1].evidence_ids == ("vid-2",)
    assert len(groups[0].categories) == 3
    assert t.reference("A", "vid-1")["source_sha"] == sha("video-frame-1")
    expect(PermissionError, lambda: t.reference("A", "private"))
    expect(PermissionError, lambda: t.reference("B", "vid-1"))
    t.retract("aud-1")
    assert "aud-1" not in t.fuse(
        "A", 150, 0.5
    )[0].evidence_ids
    expect(PermissionError, lambda: t.reference("A", "aud-1"))
    print("PASS temporal clusters, provenance and revocation")


def test_conflicts():
    t = synthetic_timeline()
    c = t.calibration["video-1"]
    new = Evidence(
        "vid-contradiction", "A", "video-1", 930, 1020,
        "visual", "door stays closed", 0.9, sha("contradiction")
    )
    t.ingest(new)
    records = t.align("A", min_confidence=0.5)
    conflicts = temporal_consistency(records)
    assert ("vid-1", "vid-contradiction") in conflicts
    assert len(conflicts) >= 1
    expect(ValueError, lambda: t.ingest(Evidence(
        "vid-1", "A", "video-1", 900, 1100,
        "visual", "changed", 0.95, sha("different")
    )))
    print("PASS interval overlaps, contradictory observations and immutable IDs")


if __name__ == "__main__":
    test_calibration()
    test_fusion()
    test_conflicts()
    print("NOTE: observations are synthetic; no vision/audio recognition occurs.")

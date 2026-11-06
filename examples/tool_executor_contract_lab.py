"""Minimal authorization, CAS, uncertain remote outcome and dedup contract."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Identity:
    tenant: str
    scopes: frozenset[str]


class Gateway:
    def __init__(self):
        self.version=1
        self.payload="old"
        self.effects=0
        self.completed={}

    def submit(self, who, operation_key, expected_version, text, simulate_lost_ack=False):
        if not isinstance(who,Identity) or "write" not in who.scopes:
            raise PermissionError("write scope denied")
        if who.tenant!="allowed":
            raise PermissionError("tenant denied")
        if not isinstance(text,str) or not text:
            raise ValueError("empty content")
        compound=(who.tenant,operation_key)
        if compound in self.completed:
            old_text, result=self.completed[compound]
            if text!=old_text:
                raise ValueError("same idempotency key, different body")
            return result
        if expected_version!=self.version:
            raise RuntimeError("compare-and-swap version mismatch")
        self.payload=text
        self.version+=1
        self.effects+=1
        result=f"operation-{self.effects}"
        self.completed[compound]=(text,result)
        if simulate_lost_ack:
            raise TimeoutError("committed, but response was lost")
        return result

    def query(self,who,key):
        if who.tenant!="allowed":
            raise PermissionError
        return self.completed.get((who.tenant,key))


def main():
    g=Gateway()
    writer=Identity("allowed",frozenset({"write"}))
    reader=Identity("allowed",frozenset({"read"}))
    try:
        g.submit(reader,"a",1,"payload")
    except PermissionError:
        pass
    else:
        raise AssertionError("read-only user mutated state")
    try:
        g.submit(writer,"a",1,"payload",simulate_lost_ack=True)
    except TimeoutError:
        pass
    else:
        raise AssertionError("expected uncertain response")
    assert g.query(writer,"a") == ("payload","operation-1")
    assert g.submit(writer,"a",1,"payload") == "operation-1"
    assert g.effects == 1
    assert g.version == 2
    try:
        g.submit(writer,"a",1,"different")
    except ValueError:
        pass
    else:
        raise AssertionError("idempotency collision accepted")
    try:
        g.submit(writer,"b",1,"payload2")
    except RuntimeError:
        pass
    else:
        raise AssertionError("stale CAS accepted")
    print("PASS strict scope, tenant, version and idempotency")
    print("PASS lost acknowledgement handled by query; effects=1")
    print("NOTE: in-memory simulation does not prove crash durability.")


if __name__ == "__main__":
    main()

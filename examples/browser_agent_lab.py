"""Deterministic browser state transitions; no real browser/network."""
from dataclasses import dataclass


@dataclass
class Page:
    signed_in: bool = True
    pending: bool = True
    committed: bool = False
    writes: int = 0
    operation_keys: set = None

    def __post_init__(self):
        if self.operation_keys is None:
            self.operation_keys=set()

    def submit(self,user_scopes,key,simulate_lost_response=False):
        if not self.signed_in or "submit" not in user_scopes:
            raise PermissionError("submission denied")
        if key in self.operation_keys:
            return "already committed"
        self.operation_keys.add(key)
        self.pending=False
        self.committed=True
        self.writes+=1
        if simulate_lost_response:
            raise TimeoutError("response lost after commit")
        return "ok"

    def status(self):
        return "COMMITTED" if self.committed else "PENDING"


def main():
    page=Page()
    try:
        page.submit(frozenset({"read"}),"K")
    except PermissionError:
        pass
    else:
        raise AssertionError("unauthorized submit allowed")
    try:
        page.submit(frozenset({"submit"}),"K",simulate_lost_response=True)
    except TimeoutError:
        pass
    else:
        raise AssertionError("expected lost response")
    assert page.status()=="COMMITTED"
    assert page.submit(frozenset({"submit"}),"K")=="already committed"
    assert page.writes==1
    print("PASS browser operation authorized and commit independently verified")
    print("PASS lost-response retry did not produce duplicate submit")
    print("NOTE: isolated state-machine exercise, not live browser testing.")


if __name__=="__main__":
    main()

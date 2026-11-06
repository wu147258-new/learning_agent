"""Versioned long-term memory filters: explicit revocation overrides rank."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Memory:
    key: str
    tenant: str
    version: int
    expires_at: int
    status: str
    similarity: float


def eligible(items,tenant,now):
    newest={}
    for m in items:
        if m.key not in newest or m.version > newest[m.key].version:
            newest[m.key]=m
    return sorted((m for m in newest.values() if
                   m.tenant==tenant and m.status=="ACTIVE" and now<m.expires_at),
                  key=lambda m:-m.similarity)


def decay(days,half_life):
    if days<0 or half_life<=0:
        raise ValueError("invalid decay")
    return math.exp(-math.log(2)*days/half_life)


def main():
    memories=[
      Memory("root","a",1,999,"ACTIVE",0.99),
      Memory("root","a",2,999,"REVOKED",0.99),
      Memory("guide","a",1,110,"ACTIVE",0.70),
      Memory("other","b",1,999,"ACTIVE",1.0),
      Memory("expired","a",1,90,"ACTIVE",0.95),
    ]
    out=eligible(memories,"a",100)
    assert [m.key for m in out]==["guide"]
    assert math.isclose(decay(20,10),0.25)
    assert math.isclose(decay(0,10),1)
    print("PASS revoked high-score memory cannot reappear via older version")
    print("PASS tenant and expiry filters, half-life decay")


if __name__=="__main__":
    main()

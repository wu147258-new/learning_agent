"""Deterministic KV page admission simulation without model execution."""
from __future__ import annotations


class Pool:
    def __init__(self, pages, block):
        if pages <= 0 or block <= 0:
            raise ValueError
        self.capacity=pages
        self.block=block
        self.active={}

    def needed(self,tokens):
        if tokens < 0:
            raise ValueError
        return (tokens+self.block-1)//self.block

    def admit(self,key,tokens):
        if key in self.active:
            raise ValueError("duplicate key")
        n=self.needed(tokens)
        if n+self.used > self.capacity:
            return False
        self.active[key]=n
        return True

    @property
    def used(self):
        return sum(self.active.values())

    def release(self,key):
        return self.active.pop(key)

def main():
    pool=Pool(5,16)
    assert pool.needed(35)==3
    assert pool.admit("request-A",35)
    assert pool.used==3
    assert not pool.admit("request-B",48)
    assert pool.used==3
    try:
        pool.admit("request-A",15)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate allocation")
    assert pool.release("request-A")==3
    assert pool.used==0
    assert pool.admit("request-B",48)
    assert pool.used==3
    print("PASS KV ceil-page accounting and bounded admission")
    print("PASS exhausted pool, duplicate request, release and readmission")
    print("NOTE: abstract page scheduler, no GPU benchmark.")


if __name__=="__main__":
    main()

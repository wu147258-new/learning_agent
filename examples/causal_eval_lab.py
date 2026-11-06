"""Synthetic Simpson's paradox in Agent success-rate comparisons."""
from math import isclose


def rate(success,total):
    if total<=0 or success<0 or success>total:
        raise ValueError
    return success/total


def main():
    new_easy=rate(81,90)
    old_easy=rate(9,10)
    new_hard=rate(2,10)
    old_hard=rate(18,90)
    assert isclose(new_easy,old_easy)
    assert isclose(new_hard,old_hard)
    new_global=rate(83,100)
    old_global=rate(27,100)
    assert isclose(new_global-old_global,.56)
    standardized_new=.5*new_easy+.5*new_hard
    standardized_old=.5*old_easy+.5*old_hard
    assert isclose(standardized_new,standardized_old)
    assert isclose(standardized_new,.55)
    print("PASS within-stratum equality and 56-point aggregate selection bias")
    print("PASS common target mix restores equal 55% standardized success")
    print("NOTE: constructed example, no real causal effect estimated.")


if __name__=="__main__":
    main()

"""Exact two-candidate DPO loss, no model training or benchmark."""
import math


def dpo_loss(preferred, rejected, ref_preferred, ref_rejected, beta):
    inputs=[preferred,rejected,ref_preferred,ref_rejected]
    if beta<=0 or not all(0 < x <=1 for x in inputs):
        raise ValueError("invalid probabilities or beta")
    if not math.isclose(preferred+rejected,1):
        raise ValueError("policy probabilities not normalized")
    if not math.isclose(ref_preferred+ref_rejected,1):
        raise ValueError("reference probabilities not normalized")
    delta=(math.log(preferred)-math.log(ref_preferred)
           -math.log(rejected)+math.log(ref_rejected))
    z=beta*delta
    # stable -log(sigmoid(z))
    loss=max(0,-z)+math.log1p(math.exp(-abs(z)))
    return loss, delta


def main():
    baseline,_=dpo_loss(.5,.5,.5,.5,1)
    better,delta=dpo_loss(.8,.2,.5,.5,1)
    assert math.isclose(baseline,math.log(2))
    assert math.isclose(delta,math.log(4))
    assert math.isclose(better,-math.log(.8))
    assert better < baseline
    for args in ((0,1,.5,.5,1),(.8,.2,1,0,1),(.8,.2,.5,.5,0)):
        try:
            dpo_loss(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid probability accepted")
    print("PASS reference-corrected DPO: ln2 -> -ln0.8")
    print("PASS invalid support, normalization and beta rejected")
    print("NOTE: simple categorical example, no actual SFT/DPO training.")


if __name__=="__main__":
    main()

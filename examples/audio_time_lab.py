"""Audio timing arithmetic and a tiny exact CTC path aggregation."""
from itertools import product
from math import isclose


def samples(seconds,rate):
    if seconds<0 or rate<=0:
        raise ValueError("invalid audio metadata")
    return round(seconds*rate)


def ctc_collapse(path,blank="_"):
    result=[]
    previous=None
    for item in path:
        if item!=previous and item!=blank:
            result.append(item)
        previous=item
    return "".join(result)


def ctc_probability(frame_distributions,target,alphabet=("_","A")):
    total=0.0
    for path in product(alphabet,repeat=len(frame_distributions)):
        if ctc_collapse(path)!=target:
            continue
        probability=1.0
        for dist,symbol in zip(frame_distributions,path):
            probability*=dist[symbol]
        total+=probability
    return total


def main():
    assert samples(10,16000)==160000
    assert samples(5,16000)*2==160000
    assert isclose(160/16000,.01)
    assert isclose(400/16000,.025)
    assert ctc_collapse(("A","A","_","A"))=="AA"
    assert ctc_collapse(("A","A"))=="A"
    # Two frames: A-blank, blank-A, A-A => target "A".
    p=ctc_probability([{"_":.5,"A":.5},{"_":.5,"A":.5}],"A")
    assert isclose(p,.75),p
    assert isclose(300*160/16000,3.0)
    print("PASS audio sample count, STFT timing and CTC exact path sum=0.75")
    print("NOTE: symbolic toy, no audio transcription model.")


if __name__=="__main__":
    main()

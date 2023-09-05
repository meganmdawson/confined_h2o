import numpy as np

def nmTransform(toNM, toBead, vs, qs):
    # switch positions to normal mode and back. then, check if the 
    # output is equal to the input.

    bVs = vs.astype('float16')
    bQs = qs.astype('float16')

    nmVs = toNM(bVs)
    nmQs = toNM(bQs)

    bVs2 = toBead(nmVs)
    bQs2 = toBead(nmQs)

    bVs2 = bVs2.astype('float16')
    bQs2 = bQs2.astype('float16')

    if np.all(np.equal(bVs, bVs2)):
        print("Velocities are the same after transformation")
    else:
        print(f"Velocities are not equal:\n{bVs}\n{bVs2}")

    if np.all(np.equal(bQs, bQs2)):
        print("Positions are the same after transformation")
    else:
        print(f"Positions are not equal:\n{bQs}\n{bQs2}")
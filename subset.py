import numpy as np

(xlen, x, ylen, y) = (np.load("testxlen.npy"), np.load("testx.npy"), np.load("testylen.npy"), np.load("testy.npy"))

np.save("subxlen.npy", xlen[-100:])
np.save("subx.npy", x[-100:])
np.save("subylen.npy", ylen[-100:])
np.save("suby.npy", y[-100:])

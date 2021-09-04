import numpy as np

rng = np.random.default_rng()


def closure_phase(i1, i2, i3):
    I12 = i1 * i2.conjugate()
    I23 = i2 * i3.conjugate()
    I31 = i3 * i1.conjugate()
    phi12 = np.angle(I12)
    phi23 = np.angle(I23)
    phi31 = np.angle(I31)
    print(phi12, phi23, phi31)
    return phi12 + phi23 + phi31


def cpxrand(n=1):
    return rng.normal(size=n) + 1j * rng.normal(size=n)


# Equation 7
phi1 = 0.05
phi2 = 0.15
phi3 = 0.35
a = cpxrand()
b = cpxrand()
# a = 0
i1 = a + b * np.exp(1j * phi1)
# a = cpxrand()
# b = cpxrand()
i2 = a + b * np.exp(1j * phi2)
# a = cpxrand()
# b = cpxrand()
i3 = a + b * np.exp(1j * phi3)



print(closure_phase(i1, i2, i3))

w1, w2, w3 = 1.0, 2.0, 3.0
a = cpxrand()
b = cpxrand()
i1 = a + b * w1
# a = cpxrand()
# b = cpxrand()
i2 = a + b * w2
# a = cpxrand()
# b = cpxrand()
i3 = a + b * w3
print(closure_phase(i1, i2, i3))


siga = 0.5
# sigb = 1.0 # this gives non-zero
sigb = siga # This will give 0 closure phase

phi1 = 0.00
phi2 = 0.75
phi3 = 1.45
I12 = siga + sigb * np.exp(1j * (phi2 - phi1))
I23 = siga + sigb * np.exp(1j * (phi3 - phi2))
I31 = siga + sigb * np.exp(1j * (phi1 - phi3))
phi12 = np.angle(I12)
phi23 = np.angle(I23)
phi31 = np.angle(I31)
print(phi12, phi23, phi31)
print(phi12 + phi23 + phi31)
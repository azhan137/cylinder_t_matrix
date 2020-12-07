from smuthi.utility import math as sf

#utility class for computing spherical Bessel and Hankel functions. Mostly serves as a wrapper for
#existing methods in smuthi.spherical_functions


#compute spherical Bessel or Hankel function. b_l(Z)
#nu = 1: Bessel
#nu = 3: Hankel
#l = degree (integer)
#Z = arguments (numpy array)
def sph_bessel(nu, l, Z):
    if nu == 1:
        return sf.spherical_bessel(l, Z)
    elif nu == 3:
        return sf.spherical_hankel(l, Z)
    else:
        return 0

#compute derivative of Bessel function d/dZ (Z * b_l(Z))
#nu = 1: Bessel
#nu = 3: Hankel
def d1Z_Z_sph_bessel(nu, l, Z):
    if nu == 1:
        return sf.dx_xj(l, Z)
    elif nu == 3:
        return sf.dx_xh(l, Z)
    else:
        return 0

#compute derivative of Bessel function d/dZ (b_l(Z))
#nu = 1: Bessel
#nu = 3: Hankel
def d1Z_sph_bessel(nu, l, Z):
        return l/Z*sph_bessel(nu, l, Z)-sph_bessel(nu, l+1, Z)

#compute second derivative of Bessel function d^2/dZ^2 (Z b_l(Z))
#nu = 1: Bessel
#nu = 3: Hankel
def d2Z_Z_sph_bessel(nu, l, Z):
        return (l+l**2-Z**2)/Z*sph_bessel(nu, l, Z)

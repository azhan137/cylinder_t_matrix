import numpy as np
from numpy.polynomial import legendre
import scipy.io as sio

#need to import spherical angular functions from SMUTHI by Amos Egel
from smuthi import spherical_functions as sf

#wrapper functions for spherical Bessel and Hankel functions found in SMUTHI
import bessel_functions as bf


#main method to be called to compute the T matrix of a scatterer and its derivatives. Takes in:
#   lmax: maximum angular momentum expansion order: integer
#   Ntheta: number of points to use for Gaussian-Legendre quadrature: integer
#   geometric_params: geometry dependent parameters as numpy array: float vector
#       cylinder: [0]: a, radius, [1] h, height
#   n0: refractive index of the medium: float
#   ns: refractive index of the scatterer: float
#   wavelength: excitation wavelength of the calculation: float
#   type of scatterer (cylinder, sphere, ellipsoid, etc): string

#returns: T and dT
#   T: nmax x nmax T-matrix
#   dT: nmax x nmax x num_geometric params dT-matrix
#       dimension depends on type of particle being computed
def compute_T(lmax, Ntheta, geometric_params, n0, ns, wavelength, particle_type):
    #compute Q matrices
    [Q, dQ] = compute_Q(lmax, Ntheta, geometric_params, n0, ns, wavelength, 3, particle_type)
    [rQ, drQ] = compute_Q(lmax, Ntheta, geometric_params, n0, ns, wavelength, 1, particle_type)

    #T matrix is given by rQ * Q^(-1)
    Qinv = np.linalg.inv(Q)
    T = np.matmul(rQ, Qinv)
    dT = np.zeros((np.shape(drQ)), dtype=np.complex_)

    num_geometric_params = np.size(geometric_params)

    #dT matrix is given by (drQ - T*dQ)*Q^(-1)
    for geometric_idx in np.arange(0, num_geometric_params):
        dT[:, :, geometric_idx] = np.matmul(drQ[:, :, geometric_idx] - np.matmul(T, dQ[:, :, geometric_idx]), Qinv)

    return T, dT

#method for computing the Q matrices that are used for T-matrix computation
#inputs are all the same as above except nu
#nu = 1: compute rQ and drQ (surface and internal field boundary conditions)
#nu = 3: compute Q and dQ (scattered and surface field boundary conditions)

#expressions for P, R, S, U integrals taken from:
#   Scattering, Absorption, and Emission of Light by Small Particles by Lacis, Mishchenko, and Travis
def compute_Q(lmax, Ntheta, geometric_params, n0, ns, wavelength, nu, particle_type):

    if particle_type is 'cylinder':
        a = geometric_params[0]
        h = geometric_params[1]
        [J11, J12, J21, J22, dJ11, dJ12, dJ21, dJ22] = compute_J_cyl(lmax, Ntheta, a, h, n0, ns, wavelength, nu)
    elif particle_type is 'ellipsoid':
        print('ellipsoid not supported')
    else:
        print('particle type ' + particle_type + ' not supported.')
        return 0

    ki = 2*np.pi*n0/wavelength
    ks = 2*np.pi*ns/wavelength

    P = -1j * ki * (ks * J21 + ki * J12)
    R = -1j * ki * (ks * J11 + ki * J22)
    S = -1j * ki * (ks * J22 + ki * J11)
    U = -1j * ki * (ks * J12 + ki * J21)

    dP = -1j * ki * (ks * dJ21 + ki * dJ12)
    dR = -1j * ki * (ks * dJ11 + ki * dJ22)
    dS = -1j * ki * (ks * dJ22 + ki * dJ11)
    dU = -1j * ki * (ks * dJ12 + ki * dJ21)

    Q = np.block([
        [P, R],
        [S, U]
    ])
    nmax = np.size(Q[:, 1])
    num_geometric_params = np.size(geometric_params)
    dQ = np.zeros((nmax, nmax, num_geometric_params), dtype=np.complex_)

    for geometric_idx in np.arange(0, num_geometric_params):
        dQ[:, :, geometric_idx] = np.block([
            [dP[:, :, geometric_idx], dR[:, :, geometric_idx]],
            [dS[:, :, geometric_idx], dU[:, :, geometric_idx]]
        ])

    return Q, dQ

#function that computes the J surface integrals and their derivatives with respect to cylinder radius (a) and cylinder
#   height (h). Expands up to a specified lmax, and approximates the integrals using gaussian quadrature with Ntheta
#   points for the two integrals required.
#   n0 is refractive index of medium
#   ns is refractive index of scatterer
#   wavelength is illumination wavelength
#   nu = 1 or 3
#       1: b_li are the spherical Bessel functions of the first kind (j_n(x))
#           involved in rQ and drQ computation
#       3: b_li are the spherical Hankel functions of the first kind (h_n(x))
#           involved in Q and dQ computation

#care should be taken to expand lmax to sufficient order,
#where lmax should be greater than (ns-n_0)*max(2*a,h)/wavelength
def compute_J_cyl(lmax, Ntheta, a, h, n0, ns, wavelength, nu):
    #dimension of final T-matrix is 2*nmax x 2*nmax for each individual matrix
    nmax = int(lmax*(lmax+2))

    #preallocate space for both J and dJ matrices of size nmax x nmax for J matrices
    #and dJ matrices are nmax x nmax x 2
    #dJ[:,:,0] is dJ/da
    #dJ[:,:,1] is dJ/dh
    J11 = np.zeros((nmax, nmax), dtype=np.complex_)
    J12 = np.zeros((nmax, nmax), dtype=np.complex_)
    J21 = np.zeros((nmax, nmax), dtype=np.complex_)
    J22 = np.zeros((nmax, nmax), dtype=np.complex_)

    dJ11 = np.zeros((nmax, nmax, 2), dtype=np.complex_)
    dJ12 = np.zeros((nmax, nmax, 2), dtype=np.complex_)
    dJ21 = np.zeros((nmax, nmax, 2), dtype=np.complex_)
    dJ22 = np.zeros((nmax, nmax, 2), dtype=np.complex_)

    #find the angle theta at which the corner of the cylinder is at
    theta_edge = np.arctan(2*a/h)
    #prepare gauss-legendre quadrature for interval of [-1,1] to perform numerical integral
    [x_norm, wt_norm] = legendre.leggauss(Ntheta)

    #rescale integration points and weights to match actual bounds:
    #   circ covers the circular surface of the cylinder (end caps)
    #   body covers the rectangular surface of the cylinder (body area)
    #circ integral goes from 0 to theta_edge, b = theta_edge, a = 0
    theta_circ = theta_edge/2*x_norm+theta_edge/2
    wt_circ = theta_edge/2*wt_norm

    #body integral goes from theta_edge to pi/2, b = pi/2, a = theta_edge
    theta_body = (np.pi/2-theta_edge)/2*x_norm+(np.pi/2+theta_edge)/2
    wt_body = (np.pi/2-theta_edge)/2*wt_norm

    #merge the circ and body lists into a single map
    theta_map = np.concatenate((theta_circ, theta_body), axis=0)
    weight_map = np.concatenate((wt_circ, wt_body), axis=0)

    #identify indices corresponding to the circular end caps and rectangular body
    circ_idx = np.arange(0, Ntheta-1)
    body_idx = np.arange(Ntheta-1, 2*Ntheta)

    #k vectors of the light in medium (ki) and in scatterer (ks)
    ki = 2*np.pi*n0/wavelength
    ks = 2*np.pi*ns/wavelength

    #precompute trig functions
    ct = np.cos(theta_map)
    st = np.sin(theta_map)
    #normal vector for circular surface (circ) requires tangent
    tant = np.tan(theta_map[circ_idx])
    #normal vector for rectangular surface (body) requires cotangent
    cott = 1/np.tan(theta_map[body_idx])

    #precompute spherical angular polynomials
    [p_lm, pi_lm, tau_lm] = sf.legendre_normalized(ct, st, lmax)

    #radial coordinate of the surface, and the derivatives with respect to a and h
    #r_c: radial coordinate of circular end cap
    #r_b: radial coordinate of rectangular body
    r_c = h/2/ct[circ_idx]
    dr_c = r_c/h
    r_b = a/st[body_idx]
    dr_b = r_b/a

    #merge radial coordiantes into a single vector
    r = np.concatenate((r_c, r_b), axis=0)

    #derivatives of the integration limits for performing derivatives
    da_edge = 2*h/(h**2+4*a**2)
    dh_edge = -2*a/(h**2+4*a**2)

    Ntheta = Ntheta-1

    #loop through each individual element of the J11, J12, J21, J22 matrices
    for li in np.arange(1, lmax+1):

        #precompute bessel functiosn and derivatives
        b_li = bf.sph_bessel(nu, li, ki*r)
        db_li = bf.d1Z_Z_sph_bessel(nu, li, ki*r)
        db2_li = bf.d2Z_Z_sph_bessel(nu, li, ki*r)
        d1b_li = bf.d1Z_sph_bessel(nu, li, ki*r)
        for lp in np.arange(1, lmax+1):

            #precompute bessel functions and derivatives
            j_lp = bf.sph_bessel(1, lp, ks*r)
            dj_lp = bf.d1Z_Z_sph_bessel(1, lp, ks*r)
            dj2_lp = bf.d2Z_Z_sph_bessel(1, lp, ks*r)
            d1j_lp = bf.d1Z_sph_bessel(1, lp, ks*r)

            #compute normalization factor
            lfactor = 1/np.sqrt(li*(li+1)*lp*(lp+1))

            for mi in np.arange(-li, li+1):

                #compute row index where element is placed
                n_i = compute_n(lmax, 1, li, mi)-1

                #precompute spherical harmonic functions
                p_limi = p_lm[li][abs(mi)]
                pi_limi = pi_lm[li][abs(mi)]
                tau_limi = tau_lm[li][abs(mi)]

                for mp in np.arange(-lp, lp+1):

                    #compute col index where element is placed
                    n_p = compute_n(lmax, 1, lp, mp)-1

                    #precompute spherical harmonic functions
                    p_lpmp = p_lm[lp][abs(mp)]
                    pi_lpmp = pi_lm[lp][abs(mp)]
                    tau_lpmp = tau_lm[lp][abs(mp)]

                    #compute selection rules that includes symmetries
                    sr_1122 = selection_rules(li, mi, lp, mp, 1)
                    sr_1221 = selection_rules(li, mi, lp, mp, 2)

                    #perform integral about phi analytically. This is roughly a sinc function
                    if mi == mp:
                        phi_exp = np.pi
                    else:
                        phi_exp = -1j*(np.exp(1j*(mp-mi)*np.pi)-1)/(mp-mi)

                    #for J11 and J22 integrals
                    if sr_1122 != 0:
                        prefactor = sr_1122*lfactor*phi_exp
                        ang = mp*pi_lpmp*tau_limi+mi*pi_limi*tau_lpmp
                        #J11 computation
                        J11_r = -1j*weight_map*prefactor*r**2*st*j_lp*b_li*ang
                        J11[n_i, n_p] = np.sum(J11_r)
                        #dJ11 computation
                        dJ11dr = 2*r*j_lp*b_li+r**2*(ks*d1j_lp*b_li+ki*d1b_li*j_lp)
                        #dJ11/da
                        dJ11[n_i, n_p, 0] = np.sum(-1j*prefactor*weight_map[body_idx]*st[body_idx]*dJ11dr[body_idx]*ang[body_idx]*dr_b)
                        #dJ11/dh
                        dJ11[n_i, n_p, 1] = np.sum(-1j*prefactor*weight_map[circ_idx]*st[circ_idx]*dJ11dr[circ_idx]*ang[circ_idx]*dr_c)
                        #J22 computation split into radial and polar integrals
                        J22_r = -1j*prefactor*weight_map*st/ki/ks*dj_lp*db_li*ang
                        J22_db = lp*(lp+1)*mi*pi_limi*p_lpmp
                        J22_dj = li*(li+1)*mp*pi_lpmp*p_limi
                        J22_t = -1j*prefactor*weight_map*st/ki/ks*(J22_db*j_lp*db_li+J22_dj*b_li*dj_lp)
                        J22[n_i, n_p] = np.sum(J22_r)+np.sum(J22_t[circ_idx]*tant)+np.sum(J22_t[body_idx]*-cott)
                        #derivative of boundary term
                        dJ22edge = -1j/ki/ks*st[Ntheta]*(J22_db[Ntheta]*j_lp[Ntheta]*db_li[Ntheta]+J22_dj[Ntheta]*dj_lp[Ntheta]*b_li[Ntheta])*(st[Ntheta]/ct[Ntheta]+ct[Ntheta]/st[Ntheta])
                        #dJ22_r/da
                        dJ22da1 = -1j/ki/ks*(ks*dj2_lp[body_idx]*db_li[body_idx]+ki*db2_li[body_idx]*dj_lp[body_idx])*dr_b*st[body_idx]*ang[body_idx]
                        #dJ22_t/da
                        dJ22da2 = 1j/ki/ks*cott*st[body_idx]*dr_b*(J22_db[body_idx]*(ks*d1j_lp[body_idx]*db_li[body_idx]+ki*j_lp[body_idx]*db2_li[body_idx])+J22_dj[body_idx]*(ki*d1b_li[body_idx]*dj_lp[body_idx]+ks*dj2_lp[body_idx]*b_li[body_idx]))
                        #dJ22_r/dh
                        dJ22dh1 = -1j/ki/ks*(ks*dj2_lp[circ_idx]*db_li[circ_idx]+ki*db2_li[circ_idx]*dj_lp[circ_idx])*dr_c*st[circ_idx]*ang[circ_idx]
                        #dJ22_t/dh
                        dJ22dh2 = -1j/ki/ks*tant*st[circ_idx]*dr_c*(J22_db[circ_idx]*(ks*d1j_lp[circ_idx]*db_li[circ_idx]+ki*j_lp[circ_idx]*db2_li[circ_idx])+J22_dj[circ_idx]*(ki*d1b_li[circ_idx]*dj_lp[circ_idx]+ks*dj2_lp[circ_idx]*b_li[circ_idx]))
                        dJ22[n_i, n_p, 0] = np.sum(prefactor*weight_map[body_idx]*dJ22da1)+np.sum(prefactor*weight_map[body_idx]*dJ22da2)+prefactor*dJ22edge*da_edge
                        dJ22[n_i, n_p, 1] = np.sum(prefactor*weight_map[circ_idx]*dJ22dh1)+np.sum(prefactor*weight_map[circ_idx]*dJ22dh2)+prefactor*dJ22edge*dh_edge
                    #for J12 and J21 integrals
                    if sr_1221 != 0:
                        prefactor = sr_1221*lfactor*phi_exp
                        ang = mi*mp*pi_limi*pi_lpmp+tau_limi*tau_lpmp
                        #J12 computation split into radial and polar
                        J12_r = prefactor*weight_map/ki*r*st*j_lp*db_li*ang
                        J12_t = prefactor*weight_map/ki*r*st*li*(li+1)*j_lp*b_li*p_limi*tau_lpmp
                        J12[n_i, n_p] = np.sum(J12_r)+np.sum(J12_t[circ_idx]*tant)+np.sum(J12_t[body_idx]*-cott)
                        #derivative of boundary term
                        dJ12edge = li*(li+1)/ki*r[Ntheta]*st[Ntheta]*j_lp[Ntheta]*b_li[Ntheta]*tau_lpmp[Ntheta]*p_limi[Ntheta]*(st[Ntheta]/ct[Ntheta]+ct[Ntheta]/st[Ntheta])
                        #dJ12_r/da
                        dJ12da1 = dr_b/ki*(j_lp[body_idx]*db_li[body_idx]+r_b*(ks*d1j_lp[body_idx]*db_li[body_idx]+ki*j_lp[body_idx]*db2_li[body_idx]))*st[body_idx]*ang[body_idx]
                        #dJ12_t/da
                        dJ12da2 = -li*(li+1)/ki*dr_b*(j_lp[body_idx]*b_li[body_idx]+r_b*(ks*d1j_lp[body_idx]*b_li[body_idx]+ki*j_lp[body_idx]*d1b_li[body_idx]))*cott*st[body_idx]*tau_lpmp[body_idx]*p_limi[body_idx]
                        #dJ12_r/dh
                        dJ12dh1 = dr_c/ki*(j_lp[circ_idx]*db_li[circ_idx]+r_c*(ks*d1j_lp[circ_idx]*db_li[circ_idx]+ki*j_lp[circ_idx]*db2_li[circ_idx]))*st[circ_idx]*ang[circ_idx]
                        #dJ12_t/dh
                        dJ12dh2 = li*(li+1)/ki*dr_c*(j_lp[circ_idx]*b_li[circ_idx]+r_c*(ks*d1j_lp[circ_idx]*b_li[circ_idx]+ki*j_lp[circ_idx]*d1b_li[circ_idx]))*tant*st[circ_idx]*tau_lpmp[circ_idx]*p_limi[circ_idx]
                        dJ12[n_i, n_p, 0] = np.sum(prefactor*weight_map[body_idx]*dJ12da1)+np.sum(prefactor*weight_map[body_idx]*dJ12da2)+prefactor*dJ12edge*da_edge
                        dJ12[n_i, n_p, 1] = np.sum(prefactor*weight_map[circ_idx]*dJ12dh1)+np.sum(prefactor*weight_map[circ_idx]*dJ12dh2)+prefactor*dJ12edge*dh_edge
                        #J21 computation split into radial and polar
                        J21_r = -prefactor*weight_map/ks*r*st*dj_lp*b_li*ang
                        J21_t = -prefactor*weight_map/ks*r*st*lp*(lp+1)*j_lp*b_li*p_lpmp*tau_limi
                        J21[n_i, n_p] = np.sum(J21_r)+np.sum(J21_t[circ_idx]*tant)+np.sum(J21_t[body_idx]*-cott)
                        #derivative of boundary terms
                        dJ21edge = -lp*(lp+1)/ks*r[Ntheta]*st[Ntheta]*j_lp[Ntheta]*b_li[Ntheta]*tau_limi[Ntheta]*p_lpmp[Ntheta]*(st[Ntheta]/ct[Ntheta]+ct[Ntheta]/st[Ntheta])
                        #dJ21_r/da
                        dJ21da1 = -dr_b/ks*(b_li[body_idx]*dj_lp[body_idx]+r_b*(ki*d1b_li[body_idx]*dj_lp[body_idx]+ks*dj2_lp[body_idx]*b_li[body_idx]))*st[body_idx]*ang[body_idx]
                        #dJ21_t/da
                        dJ21da2 = lp*(lp+1)/ks*dr_b*(j_lp[body_idx]*b_li[body_idx]+r_b*(ks*d1j_lp[body_idx]*b_li[body_idx]+ki*d1b_li[body_idx]*j_lp[body_idx]))*cott*st[body_idx]*tau_limi[body_idx]*p_lpmp[body_idx]
                        #dJ21_r/dh
                        dJ21dh1 = -dr_c/ks*(b_li[circ_idx]*dj_lp[circ_idx]+r_c*(ki*d1b_li[circ_idx]*dj_lp[circ_idx]+ks*dj2_lp[circ_idx]*b_li[circ_idx]))*st[circ_idx]*ang[circ_idx]
                        #dJ21_t/dh
                        if n_i == 10:
                            abc = 2
                        dJ21dh2 = -lp*(lp+1)/ks*dr_c*(j_lp[circ_idx]*b_li[circ_idx]+r_c*(ks*d1j_lp[circ_idx]*b_li[circ_idx]+ki*d1b_li[circ_idx]*j_lp[circ_idx]))*tant*st[circ_idx]*tau_limi[circ_idx]*p_lpmp[circ_idx]
                        dJ21[n_i, n_p, 0] = np.sum(prefactor*weight_map[body_idx]*dJ21da1)+np.sum(prefactor*weight_map[body_idx]*dJ21da2)+prefactor*dJ21edge*da_edge
                        dJ21[n_i, n_p, 1] = np.sum(prefactor*weight_map[circ_idx]*dJ21dh1)+np.sum(prefactor*weight_map[circ_idx]*dJ21dh2)+prefactor*dJ21edge*dh_edge


    return J11, J12, J21, J22, dJ11, dJ12, dJ21, dJ22

#compute n index (single index) for matrix element given its p (polarization), l (orbital angular momementum index),
#   and m (azimuthal angular momentum index).
def compute_n(lmax, p, l, m):
    return (p-1)*lmax*(lmax+2)+(l-1)*(l+1)+m+l+1

#selection rules taking into account different symmetries for an axisymmetric particle
#   1 corresponds to J11 and J22
#   2 corresponds to J12 and J21
def selection_rules(li, mi, lp, mp, diag_switch):
    if diag_switch == 1:
        return np.float_power(-1, mi)*(1+np.float_power(-1, mp-mi))*(1+(-1)**(lp+li+1))
    elif diag_switch == 2:
        return np.float_power(-1, mi)*(1+np.float_power(-1, mp-mi))*(1+(-1)**(lp+li))
    else:
        return 0

#test the T-matrix terms and derivatives
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    lmax = 4
    ns = 2.0
    ni = 1.0
    wavelength = 1000.0
    cyl_params = np.array([200.0, 460.0])
    Ntheta = 100

    [T0, dT0] = compute_T(lmax, Ntheta, cyl_params, ni, ns, wavelength, 'cylinder')
    dh = 0.01
    da = 0.01

    [J11, J12, J21, J22, dJ11, dJ12, dJ21, dJ22] = compute_J_cyl(lmax, Ntheta, cyl_params[0], cyl_params[1], ni, ns, wavelength, 3)
    [J11j, J12j, J21j, J22j, dJ11j, dJ12j, dJ21j, dJ22j] = compute_J_cyl(lmax, Ntheta, cyl_params[0], cyl_params[1], ni, ns, wavelength, 1)

    [Q, dQ] = compute_Q(lmax, Ntheta, cyl_params, ni, ns, wavelength, 3, 'cylinder')
    [rQ, drQ] = compute_Q(lmax, Ntheta, cyl_params, ni, ns, wavelength, 1, 'cylinder')

    dJ11a = dJ11[:, :, 0]
    dJ11h = dJ11[:, :, 1]

    dJ12a = dJ12[:, :, 0]
    dJ12h = dJ12[:, :, 1]

    dJ21a = dJ21[:, :, 0]
    dJ21h = dJ21[:, :, 1]

    dJ22a = dJ22[:, :, 0]
    dJ22h = dJ22[:, :, 1]

    cyl_params_a = np.array([cyl_params[0]+da, cyl_params[1]])
    cyl_params_h = np.array([cyl_params[0], cyl_params[1]+dh])

    save_dict = {}
    save_dict['lmaxp'] = lmax
    save_dict['nsp'] = ns
    save_dict['nip'] = ni
    save_dict['wavelengthp'] = wavelength
    save_dict['Nthetap'] = Ntheta
    save_dict['radiusp'] = cyl_params[0]
    save_dict['heightp'] = cyl_params[1]
    save_dict['J11p'] = J11
    save_dict['J12p'] = J12
    save_dict['J21p'] = J21
    save_dict['J22p'] = J22
    save_dict['dJ11p'] = dJ11
    save_dict['dJ22p'] = dJ22
    save_dict['dJ12p'] = dJ12
    save_dict['dJ21p'] = dJ21
    save_dict['rJ11p'] = J11j
    save_dict['rJ12p'] = J12j
    save_dict['rJ21p'] = J21j
    save_dict['rJ22p'] = J22j
    save_dict['drJ11p'] = dJ11j
    save_dict['drJ22p'] = dJ22j
    save_dict['drJ12p'] = dJ12j
    save_dict['drJ21p'] = dJ21j
    save_dict['Qp'] = Q
    save_dict['dQp'] = dQ
    save_dict['rQp'] = rQ
    save_dict['drQp'] = drQ
    save_dict['T0p'] = T0
    save_dict['dT0p'] = dT0
    save_dict['invQp'] = np.linalg.inv(Q)

    sio.savemat('C:/users/azhan/Documents/Github/cylinder_t_matrix/cylinder_python_test.mat', mdict=save_dict)

    [Ta, dTa] = compute_T(lmax, Ntheta, cyl_params_a, ni, ns, wavelength, 'cylinder')
    [Th, dTh] = compute_T(lmax, Ntheta, cyl_params_h, ni, ns, wavelength, 'cylinder')

    dT0a = dT0[:, :, 0]
    dT0h = dT0[:, :, 1]

    dTna = (Ta-T0)/da
    dTnh = (Th-T0)/dh

    diffa = dT0a - dTna
    diffh = dT0h - dTnh

    plt.figure(1)
    plt.subplot(311)
    plt.imshow(np.abs(dTna))
    plt.colorbar()
    plt.subplot(312)
    plt.imshow(np.abs(dT0a))
    plt.colorbar()
    plt.subplot(313)
    plt.imshow(np.abs(diffa))
    plt.colorbar()

    plt.figure(2)
    plt.subplot(311)
    plt.imshow(np.abs(dTnh))
    plt.colorbar()
    plt.subplot(312)
    plt.imshow(np.abs(dT0h))
    plt.colorbar()
    plt.subplot(313)
    plt.imshow(np.abs(diffh))
    plt.colorbar()

    plt.figure(3)
    plt.subplot(211)
    plt.imshow(np.abs(dTna))
    plt.colorbar()
    plt.subplot(212)
    plt.imshow(np.abs(dTnh))
    plt.colorbar()

    plt.figure(4)
    plt.subplot(211)
    plt.imshow(np.real(dT0[:,:,0]))
    plt.colorbar()
    plt.subplot(212)
    plt.imshow(np.real(dT0[:,:,1]))
    plt.colorbar()

    plt.figure(5)
    plt.imshow(np.abs(T0))
    plt.colorbar()
    plt.show()

import numpy as np
import time
from copy import deepcopy
import multiprocessing
nthread = multiprocessing.cpu_count()
try:
    import numba as nb
    numba_avail = True
except ImportError:
    print("math_op.py: module Numba is not installed. Install it if you want speed up correlation calculations")
    numba_avail = False
def mut_coh_func_py(J, fld, norm=1):
    """
    Mutual Coherence function
    """
    n_x = len(fld[0,0,:])
    n_y = len(fld[0,:,0])
    n_z = len(fld[:,0,0])

    for i_x1 in range(n_x):
        for i_y1 in range(n_y):
                for i_x2 in range(n_x):
                    for i_y2 in range(n_y):
                        j = 0
                        for k in range(n_z):
                            j += (fld[k, i_y1, i_x1] * fld[k, i_y2, i_x2].conjugate())
                        if norm:
                            AbsE1 = 0
                            AbsE2 = 0
                            for k in range(n_z):
                                AbsE1 += abs(fld[k, i_y1, i_x1])
                                AbsE2 += abs(fld[k, i_y2, i_x2])
                            J[i_y1, i_x1, i_y2, i_x2] = j / (AbsE1 * AbsE2 / n_z**2) / n_z
                        else:
                            J[i_y1, i_x1, i_y2, i_x2] = j / n_z


mut_coh_func = nb.jit('void(complex128[:,:,:,:], complex128[:,:,:], int32)', nopython=True, nogil=True)(mut_coh_func_py) \
                if numba_avail else mut_coh_func_py

try:
    import pyfftw
    fftw_avail = True
except ImportError:
    print("wave.py: module PYFFTW is not installed. Install it if you want speed up dfl wavefront calculations")
    fftw_avail = False
pi = 3.141592653589793
speed_of_light = 299792458.0 # m/s
q_e = 1.6021766208e-19       # C - Elementary charge
m_e_kg = 9.10938215e-31      # kg
h_J_s = 6.626070040e-34      # Plancks constant [J*s]

m_e_eV = m_e_kg * speed_of_light**2 / q_e  # eV (510998.8671)
m_e_MeV = m_e_eV / 1e+6                    # MeV (0.510998928)
m_e_GeV = m_e_eV / 1e+9                    # GeV

mu_0 = 4 * pi * 1e-7                     # permeability of free space (1.2566370614359173e-06)
epsilon_0 = 1 / mu_0 / speed_of_light**2 # permittivity of free space (8.854187817620e-12 F/m)

h_eV_s = h_J_s / q_e                     # [eV*s]
hr_eV_s = h_eV_s/2./pi
ro_e = q_e**2/(4*pi*epsilon_0*m_e_kg*speed_of_light**2) # classical electron radius (2.8179403267e-15 m)
lambda_C = h_J_s / m_e_kg / speed_of_light # Compton wavelength [m]
lambda_C_r = lambda_C / 2 / np.pi # reduced Compton wavelength [m]
I_Alfven = 4 * np.pi * epsilon_0 * m_e_eV * speed_of_light # Alfven (Budker) current [A], ~17kA

Cgamma = 4.*pi/3.*ro_e/m_e_MeV**3
Cq = 55./(32.*np.sqrt(3)*2*pi)*h_eV_s*speed_of_light/m_e_eV

Z0 = 1./(speed_of_light*epsilon_0)  # Ohm - impedance of free space

alpha = q_e**2 * Z0 / (2*h_J_s)     # Fine-structure constant
def dfldomain_check(domains, both_req=False):
    err = ValueError(
        'domains should be a string with one or two letters from ("t" or "f") and ("s" or "k"), not {}'.format(
            str(domains)))

    # if type(domains) is not str:
    #     raise err
    if len(domains) < 1 or len(domains) > 2:
        raise err
    if len(domains) < 2 and both_req == True:
        raise ValueError('please provide both domains, e.g. "ts" "fs" "tk" "fk"')

    domains_avail = ['t', 'f', 's', 'k']
    for letter in domains:
        if letter not in domains_avail:
            raise err

    if len(domains) == 2:
        D = [['t', 'f'], ['s', 'k']]
        for d in D:
            if domains[0] in d and domains[1] in d:
                raise err

        """
        tranfers radiation to specified domains
        *domains is a string with one or two letters: 
            ("t" or "f") and ("s" or "k")
        where 
            't' (time); 'f' (frequency); 's' (space); 'k' (inverse space); 
        e.g.
            't'; 'f'; 's'; 'k'; 'ts'; 'fs'; 'tk'; 'fk'
        order does not matter
        
        **kwargs are passed down to self.fft_z and self.fft_xy
        """
def find_nearest_idx(array, value):
    if value == -np.inf:
        value = np.amin(array)
    if value == np.inf:
        value = np.amax(array)
    return (np.abs(array-value)).argmin()
from numpy import complex128
def calc_ph_sp_dens(spec, freq_ev, n_photons, spec_squared=1):
    """
    calculates number of photons per electronvolt
    """
    # _logger.debug('spec.shape = {}'.format(spec.shape))
    if spec.ndim == 1:
        axis = 0
    else:
        if spec.shape[0] == freq_ev.shape[0]:
            spec = spec.T
        axis = 1
        #     axis=0
        # elif spec.shape[1] == freq_ev.shape[0]:
        #     axis=1
        # else:
        #     raise ValueError('operands could not be broadcast together with shapes ', spec.shape, ' and ', freq_ev.shape)
    # _logger.debug('spec.shape = {}'.format(spec.shape))

    if spec_squared:
        spec_sum = np.trapz(spec, x=freq_ev, axis=axis)
    else:
        spec_sum = np.trapz(abs(spec) ** 2, x=freq_ev, axis=axis)

    if np.size(spec_sum) == 1:
        if spec_sum == 0:  # avoid division by zero
            spec_sum = np.inf
    else:
        spec_sum[spec_sum == 0] = np.inf  # avoid division by zero

    if spec_squared:
        norm_factor = n_photons / spec_sum
    else:
        norm_factor = np.sqrt(n_photons / spec_sum)

    if spec.ndim == 2:
        norm_factor = norm_factor[:, np.newaxis]
    # _logger.debug('spec.shape = {}'.format(spec.shape))
    # _logger.debug('norm_factor.shape = {}'.format(norm_factor.shape))
    spec = spec * norm_factor
    if axis == 1:
        spec = spec.T
    # _logger.debug('spec.shape = {}'.format(spec.shape))
    return spec


class RadiationField:
    """
    3d or 2d coherent radiation distribution, *.fld variable is the same as Genesis dfl structure
    """

    def __init__(self, shape=(0, 0, 0)):
        # self.fld=np.array([]) #(z,y,x)
        self.fld = np.zeros(shape, dtype=complex128)  # (z,y,x)
        self.dx = []
        self.dy = []
        self.dz = []
        self.xlamds = None  # carrier wavelength [m]
        self.domain_z = 't'  # longitudinal domain (t - time, f - frequency)
        self.domain_xy = 's'  # transverse domain (s - space, k - inverse space)
        self.filePath = ''

#    def fileName(self):
#        return filename_from_path(self.filePath)

    def copy_param(self, dfl1, version=1):
        if version == 1:
            self.dx = dfl1.dx
            self.dy = dfl1.dy
            self.dz = dfl1.dz
            self.xlamds = dfl1.xlamds
            self.domain_z = dfl1.domain_z
            self.domain_xy = dfl1.domain_xy
            self.filePath = dfl1.filePath
        elif version == 2:
            attr_list = dir(dfl1)
            for attr in attr_list:
                if attr.startswith('__') or callable(getattr(self, attr)):
                    continue
                if attr == 'fld':
                    continue
                setattr(self, attr, getattr(dfl1, attr))

    def __getitem__(self, i):
        return self.fld[i]

    def __setitem__(self, i, fld):
        self.fld[i] = fld

    def shape(self):
        '''
        returns the shape of fld attribute
        '''
        return self.fld.shape

    def domains(self):
        '''
        returns domains of the radiation field
        '''
        return self.domain_z, self.domain_xy

    def Lz(self):  
        '''
        full longitudinal mesh size
        '''
        return self.dz * self.Nz()

    def Ly(self):  
        '''
        full transverse vertical mesh size
        '''
        return self.dy * self.Ny()

    def Lx(self):  
        '''
        full transverse horizontal mesh size
        '''
        return self.dx * self.Nx()

    def Nz(self):
        '''
        number of points in z
        '''
        return self.fld.shape[0]

    def Ny(self):
        '''
        number of points in y
        '''
        return self.fld.shape[1]

    def Nx(self):
        '''
        number of points in x
        '''
        return self.fld.shape[2]

    def intensity(self):
        '''
        3d intensity, abs(fld)**2
        '''
        return self.fld.real ** 2 + self.fld.imag ** 2 # calculates faster

    def int_z(self):
        '''
        intensity projection on z
        power [W] or spectral density [arb.units]
        '''
        return np.sum(self.intensity(), axis=(1, 2))

    def ang_z_onaxis(self):
        '''
        on-axis phase
        '''
        xn = int((self.Nx() + 1) / 2)
        yn = int((self.Ny() + 1) / 2)
        fld = self[:, yn, xn]
        return np.angle(fld)

    def int_y(self):
        '''
        intensity projection on y
        '''
        return np.sum(self.intensity(), axis=(0, 2))

    def int_x(self):
        '''
        intensity projection on x
        '''
        return np.sum(self.intensity(), axis=(0, 1))

    def int_xy(self):
        # return np.swapaxes(np.sum(self.intensity(), axis=0), 1, 0)
        return np.sum(self.intensity(), axis=0)

    def int_zx(self):
        return np.sum(self.intensity(), axis=1)

    def int_zy(self):
        return np.sum(self.intensity(), axis=2)

    def E(self):
        '''
        energy in the pulse [J]
        '''
        if self.Nz() > 1:
            return np.sum(self.intensity()) * self.Lz() / self.Nz() / speed_of_light
        else:
            return np.sum(self.intensity())

    # propper scales in meters or 2 pi / meters
    def scale_kx(self):  # scale in meters or meters**-1
        if self.domain_xy == 's':  # space domain
            return np.linspace(-self.Lx() / 2, self.Lx() / 2, self.Nx())
        elif self.domain_xy == 'k':  # inverse space domain
            k = 2 * np.pi / self.dx
            return np.linspace(-k / 2, k / 2, self.Nx())
        else:
            raise AttributeError('Wrong domain_xy attribute')

    def scale_ky(self):  # scale in meters or meters**-1
        if self.domain_xy == 's':  # space domain
            return np.linspace(-self.Ly() / 2, self.Ly() / 2, self.Ny())
        elif self.domain_xy == 'k':  # inverse space domain
            k = 2 * np.pi / self.dy
            return np.linspace(-k / 2, k / 2, self.Ny())
        else:
            raise AttributeError('Wrong domain_xy attribute')

    def scale_kz(self):  # scale in meters or meters**-1
        if self.domain_z == 't':  # time domain
            return np.linspace(0, self.Lz(), self.Nz())
        elif self.domain_z == 'f':  # frequency domain
            dk = 2 * pi / self.Lz()
            k = 2 * pi / self.xlamds
            return np.linspace(k - dk / 2 * self.Nz(), k + dk / 2 * self.Nz(), self.Nz())
        else:
            raise AttributeError('Wrong domain_z attribute')

    def scale_x(self):  # scale in meters or radians
        if self.domain_xy == 's':  # space domain
            return self.scale_kx()
        elif self.domain_xy == 'k':  # inverse space domain
            return self.scale_kx() * self.xlamds / 2 / np.pi
        else:
            raise AttributeError('Wrong domain_xy attribute')

    def scale_y(self):  # scale in meters or radians
        if self.domain_xy == 's':  # space domain
            return self.scale_ky()
        elif self.domain_xy == 'k':  # inverse space domain
            return self.scale_ky() * self.xlamds / 2 / np.pi
        else:
            raise AttributeError('Wrong domain_xy attribute')

    def scale_z(self):  # scale in meters
        if self.domain_z == 't':  # time domain
            return self.scale_kz()
        elif self.domain_z == 'f':  # frequency domain
            return 2 * pi / self.scale_kz()
        else:
            raise AttributeError('Wrong domain_z attribute')

    def ph_sp_dens(self):
        if self.domain_z == 't':
            dfl = deepcopy(self)
            dfl.fft_z()
        else:
            dfl = self
        pulse_energy = dfl.E()
        spec0 = dfl.int_z()
        freq_ev = h_eV_s * speed_of_light / dfl.scale_z()
        freq_ev_mean = np.sum(freq_ev * spec0) / np.sum(spec0)
        n_photons = pulse_energy / q_e / freq_ev_mean
        spec = calc_ph_sp_dens(spec0, freq_ev, n_photons)
        return freq_ev, spec

    def curve_wavefront(self, r=np.inf, plane='xy', domain_z=None):
        """
        introduction of the additional
        wavefront curvature with radius r

        r can be scalar or vector with self.Nz() points
        r>0 -> converging wavefront

        plane is the plane in which wavefront is curved:
            'x' - horizontal focusing
            'y' - vertical focusing
            'xy' - focusing in both planes

        domain_z is the domain in which wavefront curvature is introduced
            'f' - frequency
            't' - time
            None - original domain (default)

        """

        domains = domain_o_z, domain_o_xy = self.domain_z, self.domain_xy

        if domain_z == None:
            domain_z = domain_o_z

#        _logger.debug('curving radiation wavefront by {}m in {} domain'.format(r, domain_z))

        if np.size(r) == 1:
            if r == 0:
#                _logger.error(ind_str + 'radius of curvature should not be zero')
                raise ValueError('radius of curvature should not be zero')
            elif r == np.inf:
#                _logger.debug(ind_str + 'radius of curvature is infinite, skipping')
                return
            else:
                pass

        if domain_z == 'f':
            self.to_domain('fs')
            x, y = np.meshgrid(self.scale_x(), self.scale_y())
            if plane == 'xy' or plane == 'yx':
                arg2 = x ** 2 + y ** 2
            elif plane == 'x':
                arg2 = x ** 2
            elif plane == 'y':
                arg2 = y ** 2
            else:
#                _logger.error('"plane" should be in ["x", "y", "xy"]')
                raise ValueError()
            k = 2 * np.pi / self.scale_z()
            if np.size(r) == 1:
                self.fld *= np.exp(-1j * k[:, np.newaxis, np.newaxis] / 2 * arg2[np.newaxis, :, :] / r)
            elif np.size(r) == self.Nz():
                self.fld *= np.exp(
                    -1j * k[:, np.newaxis, np.newaxis] / 2 * arg2[np.newaxis, :, :] / r[:, np.newaxis, np.newaxis])

        elif domain_z == 't':
            self.to_domain('ts')
            x, y = np.meshgrid(self.scale_x(), self.scale_y())
            if plane == 'xy' or plane == 'yx':
                arg2 = x ** 2 + y ** 2
            elif plane == 'x':
                arg2 = x ** 2
            elif plane == 'y':
                arg2 = y ** 2
            else:
 #               _logger.error('"plane" should be in ["x", "y", "xy"]')
                raise ValueError()
            k = 2 * np.pi / self.xlamds
            if np.size(r) == 1:
                self.fld *= np.exp(-1j * k / 2 * arg2 / r)[np.newaxis, :, :]
            elif np.size(r) == self.Nz():
                self.fld *= np.exp(-1j * k / 2 * arg2[np.newaxis, :, :] / r[:, np.newaxis, np.newaxis])
            else:
                raise ValueError('wrong dimensions of radius of curvature')
        else:
            ValueError('domain_z should be in ["f", "t", None]')

        self.to_domain(domains)

    def to_domain(self, domains='ts', **kwargs):
        """
        tranfers radiation to specified domains
        *domains is a string with one or two letters:
            ("t" or "f") and ("s" or "k")
        where
            't' (time); 'f' (frequency); 's' (space); 'k' (inverse space);
        e.g.
            't'; 'f'; 's'; 'k'; 'ts'; 'fs'; 'tk'; 'fk'
        order does not matter

        **kwargs are passed down to self.fft_z and self.fft_xy
        """
#        _logger.debug('transforming radiation field to {} domain'.format(str(domains)))
        dfldomain_check(domains)

        for domain in domains:
            domain_o_z, domain_o_xy = self.domain_z, self.domain_xy
            if domain in ['t', 'f'] and domain is not domain_o_z:
                self.fft_z(**kwargs)
            if domain in ['s', 'k'] and domain is not domain_o_xy:
                self.fft_xy(**kwargs)

    def fft_z(self, method='mp', nthread=multiprocessing.cpu_count(),
              **kwargs):  # move to another domain ( time<->frequency )
#        _logger.debug('calculating dfl fft_z from ' + self.domain_z + ' domain with ' + method)
 #       start = time.time()
        orig_domain = self.domain_z
        
        if nthread < 2:
            method = 'np'
        
        if orig_domain == 't':
            if method == 'mp' and fftw_avail:
                fft_exec = pyfftw.builders.fft(self.fld, axis=0, overwrite_input=True, planner_effort='FFTW_ESTIMATE',
                                               threads=nthread, auto_align_input=False, auto_contiguous=False,
                                               avoid_copy=True)
                self.fld = fft_exec()
            else:
                self.fld = np.fft.fft(self.fld, axis=0)
            # else:
            #     raise ValueError('fft method should be "np" or "mp"')
            self.fld = np.fft.ifftshift(self.fld, 0)
            self.fld /= np.sqrt(self.Nz())
            self.domain_z = 'f'
        elif orig_domain == 'f':
            self.fld = np.fft.fftshift(self.fld, 0)
            if method == 'mp' and fftw_avail:
                fft_exec = pyfftw.builders.ifft(self.fld, axis=0, overwrite_input=True, planner_effort='FFTW_ESTIMATE',
                                                threads=nthread, auto_align_input=False, auto_contiguous=False,
                                                avoid_copy=True)
                self.fld = fft_exec()
            else:
                self.fld = np.fft.ifft(self.fld, axis=0)
                
                # else:
                # raise ValueError("fft method should be 'np' or 'mp'")
            self.fld *= np.sqrt(self.Nz())
            self.domain_z = 't'
        else:
            raise ValueError("domain_z value should be 't' or 'f'")
        
#        t_func = time.time() - start
#        if t_func < 60:
#            _logger.debug(ind_str + 'done in %.2f sec' % (t_func))
 #       else:
 #           _logger.debug(ind_str + 'done in %.2f min' % (t_func / 60))

    def fft_xy(self, method='mp', nthread=multiprocessing.cpu_count(),
               **kwargs):  # move to another domain ( spce<->inverse_space )
 #       _logger.debug('calculating fft_xy from ' + self.domain_xy + ' domain with ' + method)
#        start = time.time()
        domain_orig = self.domain_xy

        if nthread < 2:
            method = 'np'
        
        if domain_orig == 's':
            self.fld = np.fft.ifftshift(self.fld, axes=(1, 2))
            if method == 'mp' and fftw_avail:
                fft_exec = pyfftw.builders.fft2(self.fld, axes=(1, 2), overwrite_input=False,
                                                planner_effort='FFTW_ESTIMATE', threads=nthread, auto_align_input=False,
                                                auto_contiguous=False, avoid_copy=True)
                self.fld = fft_exec()
            else:
                self.fld = np.fft.fft2(self.fld, axes=(1, 2))
                # else:
                # raise ValueError("fft method should be 'np' or 'mp'")
            self.fld = np.fft.fftshift(self.fld, axes=(1, 2))
            self.fld /= np.sqrt(self.Nx() * self.Ny())
            self.domain_xy = 'k'
        elif domain_orig == 'k':
            self.fld = np.fft.ifftshift(self.fld, axes=(1, 2))
            if method == 'mp' and fftw_avail:
                fft_exec = pyfftw.builders.ifft2(self.fld, axes=(1, 2), overwrite_input=False,
                                                 planner_effort='FFTW_ESTIMATE', threads=nthread,
                                                 auto_align_input=False, auto_contiguous=False, avoid_copy=True)
                self.fld = fft_exec()
            else:
                self.fld = np.fft.ifft2(self.fld, axes=(1, 2))
            self.fld = np.fft.fftshift(self.fld, axes=(1, 2))
            # else:
            #     raise ValueError("fft method should be 'np' or 'mp'")
            self.fld *= np.sqrt(self.Nx() * self.Ny())
            self.domain_xy = 's'
        
        else:
            raise ValueError("domain_xy value should be 's' or 'k'")
        
#        t_func = time.time() - start
#        if t_func < 60:
#            _logger.debug(ind_str + 'done in %.2f sec' % (t_func))
#        else:
#            _logger.debug(ind_str + 'done in %.2f min' % (t_func / 60))
#    
    def prop(self, z, fine=1, return_result=0, return_orig_domains=1, **kwargs):
        """
        Angular-spectrum propagation for fieldfile
        
        can handle wide spectrum
          (every slice in freq.domain is propagated
           according to its frequency)
        no kx**2+ky**2<<k0**2 limitation
        
        dfl is the RadiationField() object
        z is the propagation distance in [m]
        fine=1 is a flag for ~2x faster propagation.
            no Fourier transform to frequency domain is done
            assumes no angular dispersion (true for plain FEL radiation)
            assumes narrow spectrum at center of xlamds (true for plain FEL radiation)
        
        return_result does not modify self, but returns result
        
        z>0 -> forward direction
        """
#        _logger.info('propagating dfl file by %.2f meters' % (z))
        
        if z == 0:
#            _logger.debug(ind_str + 'z=0, returning original')
            if return_result:
                return self
            else:
                return
        
#        start = time.time()
        
        domains = self.domains()
        
        if return_result:
            copydfl = deepcopy(self)
            copydfl, self = self, copydfl
        
        if fine == 1:
            self.to_domain('kf')
        elif fine == -1:
            self.to_domain('kt')
        else:
            self.to_domain('k')
        
        if self.domain_z == 'f':
            k_x, k_y = np.meshgrid(self.scale_kx(), self.scale_ky())
            k = self.scale_kz()
            # H = np.exp(1j * z * (np.sqrt((k**2)[:,np.newaxis,np.newaxis] - (k_x**2)[np.newaxis,:,:] - (k_y**2)[np.newaxis,:,:]) - k[:,np.newaxis,np.newaxis]))
            # self.fld *= H
            for i in range(self.Nz()):  # more memory efficient
                H = np.exp(1j * z * (np.sqrt(k[i] ** 2 - k_x ** 2 - k_y ** 2) - k[i]))
                self.fld[i, :, :] *= H
        else:
            k_x, k_y = np.meshgrid(self.scale_kx(), self.scale_ky())
            k = 2 * np.pi / self.xlamds
            H = np.exp(1j * z * (np.sqrt(k ** 2 - k_x ** 2 - k_y ** 2) - k))
            # self.fld *= H[np.newaxis,:,:]
            for i in range(self.Nz()):  # more memory efficient
                self.fld[i, :, :] *= H
        
        if return_orig_domains:
            self.to_domain(domains)
        
#        t_func = time.time() - start
#        _logger.debug(ind_str + 'done in %.2f sec' % t_func)
        
        if return_result:
            copydfl, self = self, copydfl
            return copydfl
    
    def prop_m(self, z, m=1, fine=1, return_result=0, return_orig_domains=1, **kwargs):
        """
        Angular-spectrum propagation for fieldfile
        
        can handle wide spectrum
          (every slice in freq.domain is propagated
           according to its frequency)
        no kx**2+ky**2<<k0**2 limitation
        
        dfl is the RadiationField() object
        z is the propagation distance in [m]
        m is the output mesh size in terms of input mesh size (m = L_out/L_inp)
        which can be a number m or a pair of number m = [m_x, m_y]
        fine==0 is a flag for ~2x faster propagation.
            no Fourier transform to frequency domain is done
            assumes no angular dispersion (true for plain FEL radiation)
            assumes narrow spectrum at center of xlamds (true for plain FEL radiation)
        
        z>0 -> forward direction
        """
#        _logger.info('propagating dfl file by %.2f meters' % (z))
        
        start = time.time()
        domains = self.domains()
        
        if return_result:
            copydfl = deepcopy(self)
            copydfl, self = self, copydfl
        
        domain_z = self.domain_z
        if np.size(m)==1:
            m_x = m
            m_y = m
        elif np.size(m)==2:
            m_x = m[0]
            m_y = m[1]
        else:
#            _logger.error(ind_str + 'm mast have shape = 1 or 2')
            raise ValueError('m mast have shape = 1 or 2')
             
        if z==0:
 #           _logger.debug(ind_str + 'z=0, returning original')
#            if m_x != 1 and m_y != 1:
#                _logger.debug(ind_str + 'mesh is not resized in the case z = 0')
            if return_result:
                return self
            else:
                return
        
        if m_x != 1:
            self.curve_wavefront(-z / (1 - m_x), plane='x')
        if m_y != 1:
            self.curve_wavefront(-z / (1 - m_y), plane='y')
        
        if fine == 1:
            self.to_domain('kf')
        elif fine == -1:
            self.to_domain('kt')
        else:
            self.to_domain('k')
        
        if z != 0:
            H = 1
            if self.domain_z == 'f':
                k_x, k_y = np.meshgrid(self.scale_kx(), self.scale_ky())
                k = self.scale_kz()
                # H = np.exp(1j * z * (np.sqrt((k**2)[:,np.newaxis,np.newaxis] - (k_x**2)[np.newaxis,:,:] - (k_y**2)[np.newaxis,:,:]) - k[:,np.newaxis,np.newaxis]))
                # self.fld *= H
                #for i in range(self.Nz()):
                #    H = np.exp(1j * z / m * (np.sqrt(k[i] ** 2 - k_x ** 2 - k_y ** 2) - k[i]))
                #    self.fld[i, :, :] *= H
                if m_x != 0:
                    for i in range(self.Nz()):
                        H=np.exp(1j * z / m_x * (np.sqrt(k[i] ** 2 - k_x ** 2) - k[i]))
                        self.fld[i, :, :] *= H
                if m_y != 0:
                    for i in range(self.Nz()):
                        H=np.exp(1j * z / m_y * (np.sqrt(k[i] ** 2 - k_y ** 2) - k[i]))
                        self.fld[i, :, :] *= H           
            else:
                k_x, k_y = np.meshgrid(self.scale_kx(), self.scale_ky())
                k = 2 * np.pi / self.xlamds
                if m_x != 0:
                    H*=np.exp(1j * z / m_x * (np.sqrt(k ** 2 - k_x ** 2) - k))                
                if m_y != 0:
                    H*=np.exp(1j * z / m_y * (np.sqrt(k ** 2 - k_y ** 2) - k))
                for i in range(self.Nz()):
                    self.fld[i, :, :] *= H
        
        self.dx *= m_x
        self.dy *= m_y
        
        if return_orig_domains:
            self.to_domain(domains)
        if m_x != 1:
            self.curve_wavefront(-m_x * z / (m_x - 1), plane='x')
        if m_y != 1:
            self.curve_wavefront(-m_y * z / (m_y - 1), plane='y')
        
        t_func = time.time() - start
#        _logger.debug(ind_str + 'done in %.2f sec' % (t_func))
        
        if return_result:
            copydfl, self = self, copydfl
            return copydfl
    
    def mut_coh_func(self, norm=1, jit=1):
        '''
        calculates mutual coherence function
        see Goodman Statistical optics EQs 5.2-7, 5.2-11
        returns matrix [y,x,y',x']
        consider downsampling the field first
        '''
        if jit:
            J = np.zeros([self.Ny(), self.Nx(), self.Ny(), self.Nx()]).astype(np.complex128)
            mut_coh_func(J, self.fld, norm=norm)
        else:
            I = self.int_xy() / self.Nz()
            J = np.mean(
                self.fld[:, :, :, np.newaxis, np.newaxis].conjugate() * self.fld[:, np.newaxis, np.newaxis, :, :],
                axis=0)
            if norm:
                J /= (I[:, :, np.newaxis, np.newaxis] * I[np.newaxis, np.newaxis, :, :])
        return J
    
    # def mut_coh_func_c(self, center=(0,0), norm=1):
    #     '''
    #     Function to calculate mutual coherence function cenetered at xy position
    
    #     Parameters
    #     ----------
    #     center_xy : tuple, optional
    #         DESCRIPTION. 
    #         point with respect to which correlation is calculated
    #         The default is (0,0) (center)
    #         accepts values either in [m] if domain=='s'
    #                            or in [rad] if domain=='k'
    #     norm : TYPE, optional
    #         DESCRIPTION. 
    #         The default is 1.
    #         flag of normalization by intensity
    #     Returns
    #     -------
    #     J : TYPE
    #         mutual coherence function matrix [ny, nx]
    #     '''
    
    #     scalex = self.scale_x()
    #     scaley = self.scale_y()
        
    #     ix = find_nearest_idx(scalex, center[0])
    #     iy = find_nearest_idx(scaley, center[1])
        
    #     dfl1 = self.fld[:, iy, ix, np.newaxis, np.newaxis].conjugate()
    #     dfl2 = self.fld[:, :, :]
    #     J = np.mean(dfl1 * dfl2, axis=0)
    #     if norm:
    #         I = self.int_xy() / self.Nz()
    #         J = J / (I[Nyh, np.newaxis :] * I[:, Nxh, np.newaxis])
    #     return J
    
    def coh(self, jit=0):
        '''
        calculates degree of transverse coherence
        consider downsampling the field first
        '''
        I = self.int_xy() / self.Nz()
        J = self.mut_coh_func(norm=0, jit=jit)
        coh = np.sum(abs(J) ** 2) / np.sum(I) ** 2
        return coh
        
    def tilt(self, angle=0, plane='x', return_orig_domains=True):
        '''
        deflects the radaition in given direction by given angle
        by introducing transverse phase chirp
        '''
 #       _logger.info('tilting radiation by {:.4e} rad in {} plane'.format(angle, plane))
   #     _logger.warn(ind_str + 'in beta')
 #       angle_warn = ind_str + 'deflection angle exceeds inverse space mesh range'
        
        k = 2 * pi / self.xlamds
        domains = self.domains()
        
        self.to_domain('s')
        if plane == 'y':
#            if np.abs(angle) > self.xlamds / self.dy / 2:
#                _logger.warning(angle_warn)
            dphi =  angle * k * self.scale_y()
            self.fld = self.fld * np.exp(1j * dphi)[np.newaxis, :, np.newaxis]
        elif plane == 'x':
 #           if np.abs(angle) > self.xlamds / self.dx / 2:
#                _logger.warning(angle_warn)
            dphi =  angle * k * self.scale_x()
            self.fld = self.fld * np.exp(1j * dphi)[np.newaxis, np.newaxis, :]
        else:
            raise ValueError('plane should be "x" or "y"')
            
        if return_orig_domains:
            self.to_domain(domains)
        
        # _logger.info('tilting radiation by {:.4e} rad in {} plane'.format(angle, plane))
        # _logger.warn('in beta')
        # angle_warn = 'deflection angle exceeds inverse space mesh range'
        
        # domains = self.domains()
        
        # dk = 2 * pi / self.Lz()
        # k = 2 * pi / self.xlamds
        # K = np.linspace(k - dk / 2 * self.Nz(), k + dk / 2 * self.Nz(), self.Nz())
        
        # self.to_domain('s')
        # if plane == 'y':
            # if np.abs(angle) > self.xlamds / self.dy / 2:
                # _logger.warning(angle_warn)
            # dphi =  angle * K[:,np.newaxis] * self.scale_y()[np.newaxis, :]
            # self.fld = self.fld * np.exp(1j * dphi)[:, :, np.newaxis]
        # elif plane == 'x':
            # if np.abs(angle) > self.xlamds / self.dx / 2:
                # _logger.warning(angle_warn)
            # dphi =  angle * K[:,np.newaxis] * self.scale_x()[np.newaxis, :]
            # self.fld = self.fld * np.exp(1j * dphi)[:, np.newaxis, :]
        # else:
            # raise ValueError('plane should be "x" or "y"')
            
        # if return_orig_domains:
            # self.to_domain(domains)
            
    def disperse(self, disp=0, E_ph0=None, plane='x', return_orig_domains=True):
        '''
        introducing angular dispersion in given plane by deflecting the radaition by given angle depending on its frequency
        disp is the dispertion coefficient [rad/eV]
        E_ph0 is the photon energy in [eV] direction of which would not be changed (principal ray)
        '''
 #       _logger.info('introducing dispersion of {:.4e} [rad/eV] in {} plane'.format(disp, plane))
 #       _logger.warn(ind_str + 'in beta')
 #       angle_warn = ind_str + 'deflection angle exceeds inverse space mesh range'
        if E_ph0 == None:
            E_ph0 = 2 *np.pi / self.xlamds * speed_of_light * hr_eV_s
        
        dk = 2 * pi / self.Lz()
        k = 2 * pi / self.xlamds        
        phen = np.linspace(k - dk / 2 * self.Nz(), k + dk / 2 * self.Nz(), self.Nz()) * speed_of_light * hr_eV_s
        angle = disp * (phen - E_ph0)
        
 #       if np.amax([np.abs(np.min(angle)), np.abs(np.max(angle))]) > self.xlamds / self.dy / 2:
 #           _logger.warning(angle_warn)
        
        domains = self.domains()
        self.to_domain('sf')
        if plane =='y':
            dphi =  angle[:,np.newaxis] * k * self.scale_y()[np.newaxis, :]
            self.fld = self.fld * np.exp(1j *dphi)[:, :, np.newaxis]
        elif plane == 'x':
            dphi =  angle[:,np.newaxis] * k * self.scale_x()[np.newaxis, :]
            self.fld = self.fld * np.exp(1j *dphi)[:, np.newaxis, :]
        
        if return_orig_domains:
            self.to_domain(domains)

class HeightProfile:
    """
    1d surface of mirror
    """

    def __init__(self):
        self.points_number = None
        self.length = None
        self.h = None
        self.s = None

    def hrms(self):
        return np.sqrt(np.mean(np.square(self.h)))

    def set_hrms(self, rms):
        self.h *= rms / self.hrms()

    def psd(self):
        psd = 1 / (self.length * np.pi) * np.square(np.abs(np.fft.fft(self.h) * self.length / self.points_number))
        psd = psd[:len(psd) // 2]
        k = np.pi / self.length * np.linspace(0, self.points_number, self.points_number // 2)
        # k = k[len(k) // 2:]
        return (k, psd)

    def save(self, path):
        tmp = np.array([self.s, self.h]).T
        np.savetxt(path, tmp)

    def load(self, path, return_obj=False):
        tmp = np.loadtxt(path).T
        self.s, self.h = tmp[0], tmp[1]
        self.points_number = self.h.size
        self.length = self.s[-1] - self.s[0]
        if return_obj:
            return self

def generate_1d_profile(hrms, length=0.1, points_number=1000, wavevector_cutoff=0, psd=None, seed=None):
    """
    Function for generating HeightProfile of highly polished mirror surface

    :param hrms: [meters] height errors root mean square
    :param length: [meters] length of the surface
    :param points_number: number of points (pixels) at the surface
    :param wavevector_cutoff: [1/meters] point on k axis for cut off small wavevectors (large wave lengths) in the PSD
                                    (with default value 0 effects on nothing)
    :param psd: [meters^3] 1d array; power spectral density of surface (if not specified, will be generated)
            (if specified, must have shape = (points_number // 2 + 1, ), otherwise it will be cut to appropriate shape)
    :param seed: seed for np.random.seed() to allow reproducibility
    :return: HeightProfile object
    """



    # getting the heights map
    if seed is not None:
        np.random.seed(seed)

    if psd is None:
        k = np.pi / length * np.linspace(0, points_number, points_number // 2 + 1)
        # defining linear function PSD(k) in loglog plane
        a = -2  # free term of PSD(k) in loglog plane
        b = -2  # slope of PSD(k) in loglog plane
        psd = np.exp(a * np.log(10)) * np.exp(b * np.log(k[1:]))
        psd = np.append(psd[0], psd)  # ??? It doesn*t important, but we need to add that for correct amount of points
        if wavevector_cutoff != 0:
            idx = find_nearest_idx(k, wavevector_cutoff)
            psd = np.concatenate((np.full(idx, psd[idx]), psd[idx:]))
    elif psd.shape[0] > points_number // 2 + 1:
        psd = psd[:points_number // 2 + 1]

    phases = np.random.rand(points_number // 2 + 1)
    height_profile = HeightProfile()
    height_profile.points_number = points_number
    height_profile.length = length
    height_profile.s = np.linspace(-length / 2, length / 2, points_number)
    height_profile.h = (points_number / length) * np.fft.irfft(np.sqrt(length * psd) * np.exp(1j * phases * 2 * np.pi),
                                                               n=points_number) / np.sqrt(np.pi)
    # scaling height_map
    height_profile.set_hrms(hrms)
    
    np.random.seed()
    
    return height_profile
def generate_gaussian_dfl(xlamds=1e-9, shape=(51, 51, 100), dgrid=(1e-3, 1e-3, 50e-6), power_rms=(0.1e-3, 0.1e-3, 5e-6),
                          power_center=(0, 0, None), power_angle=(0, 0), power_waistpos=(0, 0), wavelength=None,
                          zsep=None, freq_chirp=0, en_pulse=None, power=1e6, **kwargs):
    """
    generates RadiationField object
    narrow-bandwidth, paraxial approximations

    xlamds [m] - central wavelength
    shape (x,y,z) - shape of field matrix (reversed) to dfl.fld
    dgrid (x,y,z) [m] - size of field matrix
    power_rms (x,y,z) [m] - rms size of the radiation distribution (gaussian)
    power_center (x,y,z) [m] - position of the radiation distribution
    power_angle (x,y) [rad] - angle of further radiation propagation
    power_waistpos (x,y) [m] downstrean location of the waist of the beam
    wavelength [m] - central frequency of the radiation, if different from xlamds
    zsep (integer) - distance between slices in z as zsep*xlamds
    freq_chirp dw/dt=[1/fs**2] - requency chirp of the beam around power_center[2]
    en_pulse, power = total energy or max power of the pulse, use only one
    """

    start = time.time()

    if dgrid[2] is not None and zsep is not None:
        if shape[2] == None:
            shape = (shape[0], shape[1], int(dgrid[2] / xlamds / zsep))
        else:
            print("_logger.error(ind_str + 'dgrid[2] or zsep should be None, since either determines longiduninal grid size')")

    if 'energy' in kwargs:
        en_pulse = kwargs.pop('energy', 1)

    dfl = RadiationField((shape[2], shape[1], shape[0]))
    

    
    k = 2 * np.pi / xlamds

    dfl.xlamds = xlamds
    dfl.domain_z = 't'
    dfl.domain_xy = 's'
    dfl.dx = dgrid[0] / dfl.Nx()
    dfl.dy = dgrid[1] / dfl.Ny()

    if dgrid[2] is not None:
        dz = dgrid[2] / dfl.Nz()
        zsep = int(dz / xlamds)
        if zsep == 0:

            zsep = 1
        dfl.dz = xlamds * zsep
    elif zsep is not None:
        dfl.dz = xlamds * zsep
    else:
        print("_logger.error('dgrid[2] or zsep should be not None, since they determine longiduninal grid size')")

    rms_x, rms_y, rms_z = power_rms  # intensity rms [m]

    xp, yp = power_angle
    x0, y0, z0 = power_center
    zx, zy = power_waistpos

    if z0 == None:
        z0 = dfl.Lz() / 2

    xl = np.linspace(-dfl.Lx() / 2, dfl.Lx() / 2, dfl.Nx())
    yl = np.linspace(-dfl.Ly() / 2, dfl.Ly() / 2, dfl.Ny())
    zl = np.linspace(0, dfl.Lz(), dfl.Nz())
    z, y, x = np.meshgrid(zl, yl, xl, indexing='ij')

    qx = 1j * np.pi * (2 * rms_x) ** 2 / xlamds + zx
    qy = 1j * np.pi * (2 * rms_y) ** 2 / xlamds + zy
    qz = 1j * np.pi * (2 * rms_z) ** 2 / xlamds

    if wavelength.__class__ in [list, tuple, np.ndarray] and len(wavelength) == 2:
        domega = 2 * np.pi * speed_of_light * (1 / wavelength[0] - 1 / wavelength[1])
        dt = (z[-1, 0, 0] - z[0, 0, 0]) / speed_of_light
        freq_chirp = domega / dt / 1e30 / zsep
        # freq_chirp = (wavelength[1] - wavelength[0]) / (z[-1,0,0] - z[0,0,0])

        wavelength = np.mean([wavelength[0], wavelength[1]])

    if wavelength == None and xp == 0 and yp == 0:
        phase_chirp_lin = 0
    elif wavelength == None:
        phase_chirp_lin = x * np.sin(xp) + y * np.sin(yp)
    else:
        phase_chirp_lin = (z - z0) / dfl.dz * (dfl.xlamds - wavelength) / wavelength * xlamds * zsep + x * np.sin(
            xp) + y * np.sin(yp)

    if freq_chirp == 0:
        phase_chirp_quad = 0
    else:
        # print(dfl.scale_z() / speed_of_light * 1e15)
        # phase_chirp_quad = freq_chirp *((z-z0)/dfl.dz*zsep)**2 * xlamds / 2# / pi**2
        phase_chirp_quad = freq_chirp / (speed_of_light * 1e-15) ** 2 * (zl - z0) ** 2 * dfl.xlamds  # / pi**2
        # print(phase_chirp_quad.shape)
    t_func = time.time() - start
    print('point at '+str(t_func) + ' sec\n')
    # if qz == 0 or qz == None:
    #     dfl.fld = np.exp(-1j * k * ( (x-x0)**2/2/qx + (y-y0)**2/2/qy - phase_chirp_lin + phase_chirp_quad ) )
    # else:
    arg = np.zeros_like(z).astype('complex128')
    t_func = time.time() - start
    print('point at '+str(t_func) + ' sec\n')
    
    if qx != 0:
        arg += (x - x0) ** 2 / 2 / qx
    if qy != 0:
        arg += (y - y0) ** 2 / 2 / qy
    if abs(qz) == 0:
        idx = abs(zl - z0).argmin()
        zz = -1j * np.ones_like(arg)
        zz[idx, :, :] = 0
        arg += zz
    else:
        arg += (z - z0) ** 2 / 2 / qz
        # print(zz[:,25,25])
        
    t_func = time.time() - start
    print('point at '+str(t_func) + ' sec\n')
    if np.size(phase_chirp_lin) > 1:
        arg -= phase_chirp_lin
    if np.size(phase_chirp_quad) > 1:
        arg += phase_chirp_quad[:, np.newaxis, np.newaxis]
    dfl.fld = np.exp(-1j * k * arg)  # - (grid[0]-z0)**2/qz
    # dfl.fld = np.exp(-1j * k * ( (x-x0)**2/2/qx + (y-y0)**2/2/qy + (z-z0)**2/2/qz - phase_chirp_lin + phase_chirp_quad) ) #  - (grid[0]-z0)**2/qz
    t_func = time.time() - start
    print('point at '+str(t_func) + ' sec\n')
    if en_pulse != None and power == None:
        dfl.fld *= np.sqrt(en_pulse / dfl.E())
    elif en_pulse == None and power != None:
        dfl.fld *= np.sqrt(power / np.amax(dfl.int_z()))
    else:
        print("_logger.error('Either en_pulse or power should be defined')")
        raise ValueError('Either en_pulse or power should be defined')

    dfl.filePath = ''

    t_func = time.time() - start
    print('done in '+str(t_func) + ' sec\n')


    return dfl
def dfl_reflect_surface(dfl, angle, hrms=None, height_profile=None, axis='x', seed=None, return_height_profile=False):
    """
    Function models the reflection of ocelot.optics.wave.RadiationField from the mirror surface considering effects
    of mirror surface height errors. The method based on phase modulation.
    The input RadiationField object is modified

    :param dfl: RadiationField object from ocelot.optics.wave
    :param angle: [radians] angle of incidence with respect to the surface
    :param hrms: [meters] height root mean square of reflecting surface
    :param height_profile: HeightProfile object of the reflecting surface (if not specified, will be generated using hrms)
    :param axis: direction along which reflection takes place
    :param seed: seed for np.random.seed() to allow reproducibility
    :param return_height_profile: boolean type variable; if it equals True the function will return height_profile
    """


    start = time.time()

    dict_axes = {'z': 0, 'y': 1, 'x': 2}
    dlength = {0: dfl.dz, 1: dfl.dy, 2: dfl.dx}
    if isinstance(axis, str):
        axis = dict_axes[axis]

    points_number = dfl.fld.shape[axis]
    footprint_len = points_number * dlength[axis] / np.sin(angle)

    # generating power spectral density
    if height_profile is None:
        if hrms is None:
           
            raise ValueError('hrms and height_profile not specified')
        height_profile = generate_1d_profile(hrms, length=footprint_len, points_number=points_number, seed=seed)


        # interpolation of height_profile to appropriate sizes
 
        s = np.linspace(-footprint_len / 2, footprint_len / 2, points_number)
        h = np.interp(s, height_profile.s, height_profile.h, right=height_profile.h[0], left=height_profile.h[-1])
        height_profile = HeightProfile()
        height_profile.points_number = dfl.fld.shape[axis]
        height_profile.length = footprint_len
        height_profile.s = s
        height_profile.h = h

    # getting 2d height_err_map
    # raws_number = dfl.fld.shape[1]
    # height_profiel_map = np.ones((raws_number, points_number)) * height_profile.h[np.newaxis, :]
    phase_delay = 2 * 2 * np.pi * np.sin(angle) * height_profile.h / dfl.xlamds

    # phase modulation

    if (axis == 0) or (axis == 'z'):
        dfl.fld *= np.exp(1j * phase_delay)[:, np.newaxis, np.newaxis]
    elif (axis == 1) or (axis == 'y'):
        dfl.fld *= np.exp(1j * phase_delay)[np.newaxis, :, np.newaxis]
    elif (axis == 2) or (axis == 'x'):
        dfl.fld *= np.exp(1j * phase_delay)[np.newaxis, np.newaxis, :]

    if return_height_profile:
        return height_profile
    t_func = time.time() - start
    print('done in '+str(t_func)+' sec')

start=time.time()
#HeightProfile with ```wavevector_cutoff = 10^3```

hprofile1 = generate_1d_profile(hrms=1e-9, length=0.03, points_number=1000, seed=666)
# generating gaussian RadiationField
dfl1 = generate_gaussian_dfl(1e-9, (1000, 1000, 1))
dfl_reflect_surface(dfl1, angle=np.pi * 2 / 180, height_profile=hprofile1, axis='x')
dfl1.prop(z=10)
t_func = time.time() - start
print('all done in '+str(t_func)+' sec')
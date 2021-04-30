from ocelot.optics.wave import * 
from ocelot.gui.dfl_plot import *

# setup logger level (it is not necessary)
from ocelot.optics.wave import _logger
import logging
_logger.setLevel(logging.ERROR)

from ocelot.gui.dfl_plot import _logger
_logger.setLevel(logging.ERROR)
import time
start=time.time()
#HeightProfile with ```wavevector_cutoff = 10^3```
hprofile1 = generate_1d_profile(hrms=1e-9, length=0.03, points_number=1000, seed=666)
plot_1d_hprofile(hprofile1, fig_name='mirror1 height profile and PSD')
hprofile2 = generate_1d_profile(hrms=1e-9, length=0.1, points_number=5000, wavevector_cutoff=1e3)
plot_1d_hprofile(hprofile2, fig_name='mirror2 height profile and PSD')
# generating gaussian RadiationField
dfl1 = generate_gaussian_dfl(1e-9, (1000, 1000, 1))
#radiation ritght before mirror1
plot_dfl(dfl1, phase=1, fig_name='radiation before the mirror1')
# reflecting RadiationField from mirror1 and plotting the result
dfl_reflect_surface(dfl1, angle=np.pi * 2 / 180, height_profile=hprofile1, axis='x')
plot_dfl(dfl1, phase=1, fig_name='radiation after reflection from mirror1')
# propagating RadiationField after reflection for 10 meters and plotting the result
dfl1.prop(z=10)
plot_dfl(dfl1, phase=1, fig_name='radiation after reflection from mirror1 and propagation')
# reflecting RadiationField from another imperfect mirror vertically (also ```height_profile``` is not specified, so it will be generated internally)
hprofile3 = dfl_reflect_surface(dfl1, angle=np.pi * 2 / 180, hrms=1e-9, axis='y', return_height_profile=1, seed=13)
plot_1d_hprofile(hprofile2, fig_name='internally generated mirror3 height profile and PSD') 
plot_dfl(dfl1, phase=1, fig_name='radiation after reflection from mirror3')
# propagating RadiationField for another 10 meters and plotting the result
dfl1.prop(z=10)
plot_dfl(dfl1, phase=1, fig_name='result of reflection from two rough mirrors')
t_func = time.time() - start
print('all done in '+str(t_func)+' sec')
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
# generating gaussian RadiationField
dfl1 = generate_gaussian_dfl(1e-9, (1000, 1000, 1))
dfl_reflect_surface(dfl1, angle=np.pi * 2 / 180, height_profile=hprofile1, axis='x')
# dfl1.prop(z=10)
t_func = time.time() - start
print('all done in '+str(t_func)+' sec')
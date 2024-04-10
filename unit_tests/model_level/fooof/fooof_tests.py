import os
import yaml
import numpy as np
import pylab as PLT
import random as rnd
from sys import argv
from fooof import FOOOF
from neurodsp import sim
from scipy.integrate import simpson
from neurodsp.utils import create_times
from fooof.sim.utils import set_random_seed
from neurodsp.spectral import compute_spectrum_welch

seednum = 42
np.random.seed(seednum)
rnd.seed(seednum)
os.environ['PYTHONHASHSEED'] = str(seednum)
set_random_seed(seednum)

class create_data:

    def __init__(self,config_path,dt=10,fs=1024):
        self.dt     = dt
        self.fs     = fs
        self.times  = create_times(self.dt, self.fs)
        self.config = yaml.safe_load(open(config_path))
        
    def get_timeseries(self):

        tdict = {}
        tkeys = list(self.config['timeseries'].keys())
        if 'dirac' in tkeys:
            tdict['dirac'] = self.dirac_delta()
        if 'white' in tkeys:
            tdict['white'] = self.white_noise()
        if 'pink' in tkeys:
            tdict['pink'] = self.pink_noise()
        if 'sinusoid' in tkeys:
            tdict['sinusoid'] = self.sinusoid()
        return self.times,tdict

    def dirac_delta(self):

        params               = self.config['timeseries']['dirac']
        dirac_sig            = np.zeros(self.times.size)
        dirac_sig[params[0]] = 10 
        return dirac_sig

    def white_noise(self):

        return np.random.normal(0,1,self.times.size)

    def pink_noise(self):

       return sim.sim_powerlaw(self.dt, self.fs, exponent=-1)
    
    def sinusoid(self):
        params = self.config['timeseries']['sinusoid']
        signal = params[0]*np.sin(2*np.pi*self.times/params[1])
        return signal

def bandpower_fooof(signal, fs=1024, win_size=2, win_stride=1):

    # Define the frequency bands
    start = 1
    dstep = 6
    nstep = 5
    bands = {}
    for idx in range(start,nstep*dstep,dstep):
        bands[idx] = (idx,idx+dstep)

    # Calculate the power spectrum using welch
    freqs, powers = compute_spectrum_welch(signal, fs)

    # Initialize a FOOOF object
    fg = FOOOF()

    # Set the frequency range to fit the model
    freq_range = [0, nstep*dstep+1]

    # Report: fit the model, print the resulting parameters, and plot the reconstruction
    fg.fit(freqs, powers, freq_range)
    
    # We want the aperiodic component from the fit. Returns the scalar,powerlaw pair
    fres = fg.get_results()
    
    # get the one over f curve
    b0,b1      = fres.aperiodic_params
    one_over_f = b0-np.log10(freqs**b1)
    
    # Get the residual periodic fit
    periodic_comp = powers-one_over_f
    
    # Loop over the bands to get the periodic power within
    power_dict = {'b0':b0,'b1':b1}
    for i_band, (lo, hi) in enumerate(bands.values()):
        inds = (freqs>=lo)&(freqs<hi)
        bp   = simpson(y=periodic_comp[inds],x=freqs[inds])
        power_dict[f"[{lo}-{hi}]"] = bp

    return freqs,powers,fg,power_dict,periodic_comp

if __name__ == '__main__':

    # Create timeseries
    CD          = create_data(argv[1])
    times,tdict = CD.get_timeseries() 

    # Merge timeseries
    signal = np.zeros(times.size)
    for v in tdict.values():signal+=v

    # Get the bandpower
    freqs,powers,fg,power_dict,periodic_comp = bandpower_fooof(signal)

    # Show the fooof report
    fg.report()
    PLT.show()

    # Print the band powers
    for k,v in power_dict.items():
        if k in ['b0','b1']:
            print(f"Aperiodic parameter {k} = {v}")
        else:
            print(f"Band {k} power is: {v}")
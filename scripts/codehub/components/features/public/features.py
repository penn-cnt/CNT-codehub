import os
import ast
import sys
import inspect
import numpy as np
import pandas as PD
from tqdm import tqdm
from fooof import FOOOF
from scipy.integrate import simpson
from scipy.signal import welch, find_peaks, detrend
from neurodsp.spectral import compute_spectrum_welch

# Import error logging (primarily for mne)
from components.core.internal.config_loader import *
from components.metadata.public.metadata_handler import *

# In some cases, we want variables to persist through steps. (i.e. A solver, fitting class, disk i/o, etc.) Persistance_dict can store results across steps.
global persistance_dict
persistance_dict = {}

class FOOOF_processing:

    def __init__(self, data, fs, freq_range, file, ichannel):
        self.data       = data
        self.fs         = fs
        self.freq_range = freq_range
        self.file       = file
        self.ichannel   = ichannel

    def create_initial_power_spectra(self):
        self.freqs, self.initial_power_spectrum = compute_spectrum_welch(self.data, self.fs)
        inds                                    = (self.freqs>0)&np.isfinite(self.initial_power_spectrum)&(self.initial_power_spectrum>0)
        self.freqs                              = self.freqs[inds]
        self.initial_power_spectrum             = self.initial_power_spectrum[inds]

        print(self.initial_power_spectrum)
        print(self.initial_power_spectrum.size)
        print(inds.sum()/inds.size)
        exit()

    def fit_fooof(self):

        # Initialize a FOOOF object
        fg = FOOOF(peak_width_limits=(2,12))

        # Report: fit the model, print the resulting parameters, and plot the reconstruction
        fg.fit(self.freqs, self.initial_power_spectrum, self.freq_range)

        # get the one over f curve
        b0,b1      = fg.get_results().aperiodic_params
        one_over_f = b0-np.log10(self.freqs**b1)

        # Get the residual periodic fit
        periodic_comp = self.initial_power_spectrum-one_over_f

        # Store the results for persistant fitting
        persistance_dict['fooof']                                    = {}
        persistance_dict['fooof'][self.file]                         = {}
        persistance_dict['fooof'][self.file][self.ichannel]          = {}
        persistance_dict['fooof'][self.file][self.ichannel]['model'] = fg
        persistance_dict['fooof'][self.file][self.ichannel]['data']  = (self.freqs,periodic_comp)

    def check_persistance(self):
        try:
            persistance_dict['fooof'][self.file][self.ichannel]
        except KeyError:
            self.create_initial_power_spectra()
            self.fit_fooof()

    def fooof_aperiodic_b0(self):
        """
        Return the constant offset for the aperiodic fit.

        Returns:
            float: Unitless float for the aperiodic offset parameter.
        """

        # Make the optional tag to identify the dataslice
        self.optional_tag = ''

        # Check for fooof model, then get aperiodic b0
        self.check_persistance()
        return persistance_dict['fooof'][self.file][self.ichannel]['model'].aperiodic_params_[0],self.optional_tag
    
    def fooof_aperiodic_b1(self):
        """
        Return the powerlaw exponent for the aperiodic fit.

        Returns:
            float: Unitless float for the aperiodic powerlaw parameter.
        """

        # Make the optional tag to identify the dataslice
        self.optional_tag = ''

        # Check for fooof model, then get aperiodic b1
        self.check_persistance()
        return persistance_dict['fooof'][self.file][self.ichannel]['model'].aperiodic_params_[1],self.optional_tag

    def fooof_bandpower(self,lo_freq,hi_freq):
        """
        Return the bandpower with the aperiodic component removed.

        Returns:
            float: Bandpower in the periodic component in the given frequency band.
        """

        # Add in the optional tagging to denote frequency range of this step
        low_freq_str      = f"{lo_freq:.2f}"
        hi_freq_str       = f"{hi_freq:.2f}"
        self.optional_tag = '['+low_freq_str+','+hi_freq_str+']'

        # Check for fooof model, then get aperiodic b1
        self.check_persistance()
        x,y = persistance_dict['fooof'][self.file][self.ichannel]['data']

        # Get the correct array slice to return the simpson integration
        inds = (x>=lo_freq)&(x<hi_freq)
        return simpson(y=y[inds],x=x[inds]),self.optional_tag

class signal_processing:
    """
    Class devoted to basic signal processing tasks. (Band-power/peak-finder/etc.)
    """
    
    def __init__(self, data, fs):
        self.data = data
        self.fs   = fs
    
    def spectral_energy_welch(self, low_freq=-np.inf, hi_freq=np.inf, win_size=2., win_stride=1.):
        """
        Returns the spectral energy using the Welch method.

        Args:
            low_freq (float, optional): Low frequency cutoff. Defaults to -np.inf.
            hi_freq (float, optional): High frequency cutoff. Defaults to np.inf.
            win_size (float, optional): Window size in units of sampling frequency. Defaults to 2.
            win_stride (float, optional): Window overlap in units of sampling frequency. Defaults to 1.

        Returns:
            spectral_energy (float): Spectral energy within the frequency band.
            optional_tag (string): Unique identifier that is added to the output dataframe to show the frequency window for which a welch spectral energy was calculated.
        """

        # Add in the optional tagging to denote frequency range of this step
        low_freq_str      = f"{low_freq:.2f}"
        hi_freq_str       = f"{hi_freq:.2f}"
        self.optional_tag = '['+low_freq_str+','+hi_freq_str+']'

        # Get the number of samples in each window for welch average and the overlap
        nperseg = int(float(win_size) * self.fs)
        noverlap = int(float(win_stride) * self.fs)

        # Calculate the welch periodogram
        frequencies, psd = welch(x=self.data.reshape((-1,1)), fs=self.fs, nperseg=nperseg, noverlap=noverlap, axis=0)
        psd              = psd.flatten()

        # Calculate the spectral energy
        mask            = (frequencies >= low_freq) & (frequencies <= hi_freq)
        spectral_energy = np.trapz(psd[mask], frequencies[mask])

        return spectral_energy,self.optional_tag
    
    def topographic_peaks(self,prominence_height,min_width,height_unit='zscore',width_unit='seconds',detrend_flag=False):
        """
        Find the topographic peaks in channel data. This is a naive/fast way of finding spikes or slowing.

        Args:
            prominence_height (_type_): Prominence threshold.
            min_width (_type_): Minimum width of peak.
            height_unit (str, optional): Unit for prominence height. zscore=Height by zscore. Else, absolute height. Defaults to 'zscore'.
            width_unit (str, optional): Unit for width of peaks. 'seconds'=width in seconds. Else, width in bins. Defaults to 'seconds'.
            detrend_flag (bool, optional): Detrend the data before searching for peaks. Defaults to False.

        Returns:
            out (string): Underscore concatenated string of peak, left edge of peak, and right edge of peak
            optional_tag (string): Underscore concatenated string of prominence height, width, and their unit types for this feature step to unique identify the feature.
        """

        # Make the optional tag output
        self.optional_tag = f"{prominence_height:.2f}_{height_unit}_{min_width:.2f}_{width_unit}_{detrend_flag}"

        # Detrend as needed
        if detrend_flag:
            data = detrend(self.data)
        else:
            data = np.copy(self.data)

        # Recast height into a pure number as needed
        if height_unit == 'zscore':
            prominence_height_input = np.median(data)+prominence_height*np.std(data)
        else:
            prominence_height_input = prominence_height

        # Recast width into a pure number as needed
        if width_unit == 'seconds':
            min_width_input = min_width*self.fs
        else:
            min_width_input = min_width

        # Calculate the peak info
        output = find_peaks(data,prominence=prominence_height_input,width=min_width_input)

        # Get peak info
        try:
            peak       = output[0][0]
            lwidth     = peak-output[1]['left_ips'][0]
            rwidth     = output[1]['right_ips'][0]-peak
        except IndexError:
            peak   = None
            lwidth = None
            rwidth = None

        # We can only return a single object that is readable by pandas, so pack results into a string to be broken down later by user
        out = [peak,lwidth,rwidth]

        # Return a tuple of (peak, left width, right width) to store all of the peak info
        return out,self.optional_tag

class basic_statistics:

    def __init__(self, data, fs):
        self.data = data
        self.fs   = fs

    def median(self):
        """
        Returns the median value in a channel.

        Returns:
            float: Median channel intensity.
        """

        return np.median(self.data),'median'
    
    def stdev(self):
        """
        Returns the standard deviation in a channel.

        Returns:
            float: Standard deviation in a channel.
        """

        return np.std(self.data),'stdev'
    
    def quantile(self,q,method='median_unbiased'):
        """
        Returns the q-th quantile of the data.

        Args:
            q (float): Probability to measure the quantile from. q∈[0:1] .
            method (str, optional): Interpolation method. Defaults to 'median_unbiased'.
        """

        optional_tag = f"quantile_{q:.2f}"
        return np.quantile(self.data,q=q,method=method),optional_tag

class features:
    """
    This class invokes the various features that can be calculated. This should not be altered without good reason.

    New feature extraction tasks should go into other classes in this script.

    Each feature should return either the scalar feature value, or a tuple with the scalar and some optional tagging for additional group distinctions.

    If you need vector returned, you can create an instance level variable and return the required elements as scalars so the resulting dataframe properly sorts values.
    """

    def __init__(self):
        """
        Use the feature extraction configuration file to step through the preprocessing pipeline on each data array
        in the output data container.
        """

        # Initialize some variables
        dummy_key       = list(self.metadata.keys())[0]
        channels        = self.metadata[dummy_key]['montage_channels']
        self.feature_df = PD.DataFrame(columns=['file','t_start','t_end','dt','method','tag']+channels)
        
        # Read in the feature configuration
        YL = config_loader(self.args.feature_file)
        config,self.feature_commands = YL.return_handler()

        # Get the current module (i.e., the script itself)
        current_module = sys.modules[__name__]

        # Use the inspect module to get a list of classes in the current module
        classes = [cls for name, cls in inspect.getmembers(current_module, inspect.isclass)]

        # Make a dummy list so we can append files to the dataframe in a staggered fashion (performance improvement)
        df_values = []

        # Iterate over steps, find the corresponding function, then invoke it.
        steps = np.sort(list(self.feature_commands.keys()))
        desc  = "Feature extraction with id %s:" %(self.unique_id)
        for istep in tqdm(steps, desc=desc, total=len(steps), bar_format=self.bar_frmt, position=self.worker_number, leave=False, disable=self.args.silent,dynamic_ncols=True):

            # Get information about the method
            method_name = self.feature_commands[istep]['method']
            method_args = self.feature_commands[istep]['args']
            error_flag  = False

            for cls in classes:
                if hasattr(cls,method_name):

                    # Loop over the datasets and the channels in each
                    for idx,dataset in enumerate(self.output_list):
                        
                        # Grab the current meta data object
                        imeta = self.metadata[idx]

                        # Get the input frequencies
                        fs = imeta['fs']

                        # Loop over the channels and get the updated values
                        output = [] 
                        for ichannel in range(dataset.shape[1]):

                            for key, value in method_args.items():
                                try:
                                    method_args[key] = ast.literal_eval(value)
                                except:
                                    pass

                            # Perform preprocessing step
                            try:
                                # Create namespaces for each class. Then choose which style of initilization is used by logic gate.
                                if cls.__name__ != 'FOOOF_processing':
                                    namespace = cls(dataset[:,ichannel],fs[ichannel])
                                else:
                                    namespace = cls(dataset[:,ichannel],fs[ichannel],[0.5,128], imeta['file'], ichannel)

                                # Get the method name and return results from the method
                                method_call         = getattr(namespace,method_name)
                                result_a, result_b  = method_call(**method_args)

                                # Check if we have a multivalue output
                                if type(result_a) == list:
                                    metadata_handler.add_metadata(self,idx,method_name,result_a)
                                    result_a = result_a[0]

                                output.append(result_a)
                            except Exception as e:
                                # We need a flexible solution to errors, so just populating a nan value
                                output.append(None)
                                try:
                                    result_b = getattr(namespace,'optional_tag')
                                except:
                                    result_b = "None"
                                
                                # Save the error for this step
                                if not error_flag:
                                    error_dir = f"{self.args.outdir}errors/"
                                    if not os.path.exists(error_dir):
                                        os.system(f"mkdir -p {error_dir}")

                                    fp = open(f"{error_dir}{self.worker_number}_features.error","a")
                                    fp.write(f"Error {e} in {method_name}\n")
                                    fp.close()
                                    error_flag = True

                        # Use metadata to allow proper feature grouping
                        meta_arr = [imeta['file'],imeta['t_start'],imeta['t_end'],imeta['dt'],method_name,result_b]
                        df_values.append(np.concatenate((meta_arr,output),axis=0))

                        # Stagger condition for pandas concat
                        if (idx%5000==0):

                            # Dataframe creations
                            iDF = PD.DataFrame(df_values,columns=self.feature_df.columns)
                            if not iDF[channels].isnull().values.all() and not iDF[channels].isna().values.all():
                                self.feature_df = PD.concat((self.feature_df,iDF))

                            # Clean up the dummy list
                            df_values = []

                    # Dataframe creations
                    iDF = PD.DataFrame(df_values,columns=self.feature_df.columns)
                    if not iDF[channels].isnull().values.all() and not iDF[channels].isna().values.all():
                        self.feature_df = PD.concat((self.feature_df,iDF))

                    # Downcast feature array to take up less space in physical and virtual memory. Use downcast first in case its a feature that cannot be made numeric
                    for ichannel in channels:
                        try:
                            self.feature_df[ichannel]=PD.to_numeric(self.feature_df[ichannel], downcast='integer')
                            self.feature_df[ichannel]=self.feature_df[ichannel].astype('float32')
                        except ValueError:
                            pass

        # The stagger condition seems to add duplicates. Need to fix eventually.
        self.feature_df = self.feature_df.drop_duplicates(ignore_index=True)

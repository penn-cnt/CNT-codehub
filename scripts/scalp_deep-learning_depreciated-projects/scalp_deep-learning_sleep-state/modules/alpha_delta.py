import numpy as np
import matplotlib.pyplot as PLT

class alpha_delta:

    def __init__(self,sleep_df,awake_df):
        self.sleep   = sleep_df.drop_duplicates()
        self.awake   = awake_df.drop_duplicates()
        self.basedir = "/Users/bjprager/Documents/GitHub/scalp_deep-learning/user_data/derivative/sleep_state/trial_03/features/five_second_two_second_overlap/" 

    def get_indices(self):
        self.sleep_indices = self.sleep.groupby(['file','t_start','t_end']).indices
        self.awake_indices = self.awake.groupby(['file','t_start','t_end']).indices
        self.sleep_keys    = list(self.sleep_indices.keys())
        self.awake_keys    = list(self.awake_indices.keys())

    def get_alpha_delta(self):
        
        self.sleep_alpha_delta = []
        self.sleep_alpha       = []
        self.sleep_delta       = []
        for ikey in self.sleep_keys:
            iDF   = self.sleep.iloc[self.sleep_indices[ikey]]
            alpha = iDF.loc[iDF.tag=='[8.0,12.0]']['C03-P03'].values[0]
            delta = iDF.loc[iDF.tag=='[1.0,4.0]']['C03-P03'].values[0]
            self.sleep_alpha_delta.append(alpha/delta)
            self.sleep_alpha.append(alpha)
            self.sleep_delta.append(delta)

        self.awake_alpha_delta = []
        self.awake_alpha       = []
        self.awake_delta       = []
        for ikey in self.awake_keys:
            iDF   = self.awake.iloc[self.awake_indices[ikey]]
            alpha = iDF.loc[iDF.tag=='[8.0,12.0]']['C03-P03'].values[0]
            delta = iDF.loc[iDF.tag=='[1.0,4.0]']['C03-P03'].values[0]
            self.awake_alpha_delta.append(alpha/delta)
            self.awake_alpha.append(alpha)
            self.awake_delta.append(delta)

    def plot_alpha(self):

        # Make the bins and steps
        self.bmin  = 0
        self.bmax  = 4e8 #max([r1,r2])
        self.nstep = 200
        self.bns   = np.linspace(self.bmin,self.bmax,self.nstep)
        self.x     = 0.5*(self.bns[:-1]+self.bns[1:]) 

        # Get the counts
        counts_sleep, _ = np.histogram(self.sleep_alpha,bins=self.bns)
        counts_awake, _ = np.histogram(self.awake_alpha,bins=self.bns)

        # Get the probabilities
        self.y_sleep = counts_sleep/np.trapz(counts_sleep,x=self.x)
        self.y_awake = counts_awake/np.trapz(counts_awake,x=self.x)

        # Set the labels
        self.x_label = r"x=$\alpha$"
        self.title   = r"$\alpha$"
        self.fname   = f"{self.basedir}plots/alpha.png"

        self.plot_pdf()

    def plot_delta(self):

        # Make the bins and steps
        self.bmin  = 0
        self.bmax  = 1e9
        self.nstep = 200
        self.bns   = np.linspace(self.bmin,self.bmax,self.nstep)
        self.x     = 0.5*(self.bns[:-1]+self.bns[1:]) 

        # Get the counts
        counts_sleep, _ = np.histogram(self.sleep_delta,bins=self.bns)
        counts_awake, _ = np.histogram(self.awake_delta,bins=self.bns)

        # Get the probabilities
        self.y_sleep = counts_sleep/np.trapz(counts_sleep,x=self.x)
        self.y_awake = counts_awake/np.trapz(counts_awake,x=self.x)

        # Set the labels
        self.x_label = r"x=$\delta$"
        self.title   = r"$\delta$"
        self.fname   = f"{self.basedir}plots/delta.png"
        
        self.plot_pdf()

    def plot_alpha_delta(self):

        # Make the bins and steps
        self.bmin  = 0
        self.bmax  = 2
        self.nstep = 200
        self.bns   = np.linspace(self.bmin,self.bmax,self.nstep)
        self.x     = 0.5*(self.bns[:-1]+self.bns[1:]) 

        # Get the counts
        counts_sleep, _ = np.histogram(self.sleep_alpha_delta,bins=self.bns)
        counts_awake, _ = np.histogram(self.awake_alpha_delta,bins=self.bns)

        # Get the probabilities
        self.y_sleep = counts_sleep/np.trapz(counts_sleep,x=self.x)
        self.y_awake = counts_awake/np.trapz(counts_awake,x=self.x)

        # Set the labels
        self.x_label = r"x=$\alpha$/$\delta$"
        self.title   = r"$\alpha$/$\delta$"
        self.fname   = f"{self.basedir}plots/ratio_alpha_delta.png"

        self.plot_pdf()

    def plot_pdf(self):

        # Initial plotting
        fig = PLT.figure(dpi=100,figsize=(8.,8.))
        ax  = fig.add_subplot(111)
        p1  = ax.step(self.x,self.y_awake,c='g',where='mid')[0]
        p2  = ax.step(self.x,self.y_sleep,c='r',where='mid')[0]

        # Adjust the x axis a bit
        dx = 0.1*(self.bmax-self.bmin)
        ax.set_xticks(np.arange(self.bmin,self.bmax+dx,dx))

        # Adjust y axis a bit
        yticks = ax.get_yticks()
        yticks = np.sort(np.concatenate((0.5*(yticks[:-1]+yticks[1:]),yticks)))
        ax.set_yticks(yticks)
        ax.set_ylim([0,max(yticks)])

        # Labeling
        ax.set_xlabel(self.x_label,fontsize=16)
        ax.set_ylabel(f"P(x<X)",fontsize=16)
        ax.legend([p1,p2],['Awake','Sleep'],prop={'size':14})
        ax.set_title(self.title,fontsize=16)

        # Clean up
        fig.tight_layout()
        PLT.grid(True)
        PLT.savefig(self.fname)
        PLT.close("all")
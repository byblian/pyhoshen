import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_version():
    return 2

def compute_correlations(cholesky_matrices): #samples['election21_2019_cholesky_matrix',-1000:]
    chol_array = T.tensor3('chol_array', dtype='float64')
    def compute_corr(chol):
      cov=T.dot(chol, chol.T)
      sd=T.sqrt(T.diag(cov))
      sd_1=T.diag(sd**-1)
      return T.nlinalg.matrix_dot(sd_1, cov, sd_1)
    chol_array_out,_= theano.scan(compute_corr, sequences=[chol_array])
    dot_chol_array = theano.function(inputs=[chol_array], outputs=chol_array_out)
    return dot_chol_array(cholesky_matrices)

def plot_correlation_matrix(correlation_matrix, labels, alignRight=False, cmap=None):
    # Generate a mask for the upper triangle
    mask = np.zeros_like(correlation_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    
    # Generate a custom diverging colormap
    if cmap is None:
        cmap = sns.diverging_palette(10, 150, s=80, as_cmap=True)
    
    mask = mask[1:,:-1]
    corr = correlation_matrix[1:,:-1]
    cbar_kws={"shrink": .5}
    if alignRight:
        cbar_kws['use_gridspec'] = False
        cbar_kws['location'] = 'left'
        
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws=cbar_kws,
                yticklabels=labels[1:], xticklabels=labels[:-1])

    if alignRight:
        ax.yaxis.tick_right()
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)        
        ax.invert_xaxis()

def plot_correlation_matrices(correlation_matrices, labels, alignRight=False, cmap=None):
    import seaborn as sns
    from seaborn.utils import iqr
    import matplotlib as mpl
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import numpy as np
    from distutils.version import LooseVersion
    
    def _freedman_diaconis_bins(a):
        """Calculate number of hist bins using Freedman-Diaconis rule."""
        # From https://stats.stackexchange.com/questions/798/
        a = np.asarray(a)
        if len(a) < 2:
            return 1
        h = 2 * iqr(a) / (len(a) ** (1 / 3))
        # fall back to sqrt(a) bins if iqr is 0
        if h == 0:
            return int(np.sqrt(a.size))
        else:
            return int(np.ceil((a.max() - a.min()) / h))
    
    corr_t = correlation_matrices.transpose(1,2,0)
    if cmap is None:
        cmap = sns.diverging_palette(10, 150, s=80, as_cmap=True)

    f, ax = plt.subplots(len(labels) - 1, len(labels) - 1, figsize=(11, 9))
    norm = mpl.colors.Normalize(vmin=-0.3,vmax=0.3)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    hist_kws = {}
    norm_hist = True
    if LooseVersion(mpl.__version__) < LooseVersion("2.2"):
        hist_kws.setdefault("normed", norm_hist)
    else:
        hist_kws.setdefault("density", norm_hist)

    xlabels = labels[:-1]
    if alignRight:
        xlabels = xlabels[::-1]
    for subplot, col in zip(ax[-1], xlabels):
        subplot.set_xlabel(col, rotation=90)
    
    ylabels = labels[1:]
    for subplot, row in zip(ax[:,-1 if alignRight else 0], ylabels):
        if alignRight:
          subplot.yaxis.tick_right()
          subplot.yaxis.set_label_position('right')
        subplot.set_ylabel(row, rotation=0, horizontalalignment='left' if alignRight else 'right')
    for yindex, yparty in enumerate(ylabels):
        plots = ax[yindex]
        for xindex, xparty in enumerate(labels[:-1]):
          subplot = plots[-xindex-1] if alignRight else plots[xindex]
          subplot.tick_params(
              bottom=False, labelbottom=False,
              top=False, labeltop=False,
              left=False, labelleft=False,
              right=False, labelright=False)
          subplot.grid(False)
          if xindex <= yindex:
            correlations = corr_t[yindex + 1][xindex]
            nbins = min(_freedman_diaconis_bins(correlations), 50)
            n, bins, patches = subplot.hist(correlations, nbins, **hist_kws)
            for i,p in enumerate(patches):
              p.set_facecolor(m.to_rgba(np.mean((bins[i], bins[i+1]))))
    
            subplot.set_facecolor(m.to_rgba(correlations.mean()))
            sns.kdeplot(correlations, ax=subplot, color='w', linewidth=1)
    
    m.set_array([])
    
    f.colorbar(m, ax=ax, location='left' if alignRight else 'right', shrink=0.5)
    return f


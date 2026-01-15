
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

def cm_to_inches(cm):
    return cm/2.54

def setup_plotting(fontsize_labels=8, fontsize_ticks=6, fontsize_title=12, fontsize_legend=8):
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["text.usetex"] = True

    label_font = {'fontsize': fontsize_labels, 'fontfamily': 'serif'}
    tick_font = {'fontsize': fontsize_ticks, 'fontfamily': 'serif'}
    title_font = {'fontsize': fontsize_title, 'fontfamily': 'serif'}
    legend_font = {'fontsize': fontsize_legend, 'fontfamily': 'serif'}

    return label_font, tick_font, title_font, legend_font
def set_spines_width(ax, width=0.5):
    for spine in ax.spines.values():
        spine.set_linewidth(width)

class ZeroIncludingMaxNLocator(ticker.MaxNLocator):
    def tick_values(self, vmin, vmax):
        ticks = super().tick_values(vmin, vmax)
        if 0 not in ticks:
            ticks = np.sort(np.append(ticks, 0))
        return ticks
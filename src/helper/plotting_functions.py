import matplotlib.pyplot as plt
import numpy as np

from typing import *

def loss_plot(train_loss: list, val_loss: list | None = None, title: str = 'Model Loss', 
               ylabel: str = 'Loss', legend: bool = False, savepath: str | None = None) -> None:
    """
    Creates a matplotlib loss plot for a Tensorflow model. Shows the plot.

    train_loss [list]: The train loss to plot. Required. <br>
    val_loss [list]: The validation loss to plot. Defaults to None. <br>
    title [str]: The title to use for the plot. Defaults to 'Model Loss'. <br>
    ylabel [str]: The y-axis label to use for the plot. Defaults to 'Loss'. <br>
    legend [bool]: If a legend should be on the plot. Defaults to False. <br>
    savepath [str]: The path to save the figure before showing. Defaults to None (meaning not saving).
    """

    plt.plot(train_loss)
    plt.xlabel('Epoch')

    if val_loss is not None:
        plt.plot(val_loss)
    
    plt.title(title)
    plt.ylabel(ylabel)
    
    if legend:
        plt.legend(['Train', 'Val'], loc='upper right')
    
    if savepath is not None:
        plt.savefig(savepath)
    
    plt.show()

def heatmap_2d(locations: list[list] | np.array | np.ndarray, bins: list = [np.arange(0, 1, 12/512), np.arange(0, 1, 12/512)],
               colorbar: bool = True, savepath: str | None = None) -> None:
    """
    Creates a 2D heatmap given an array of (x, y) coordinates. Shows the plot.

    locations [np.array or equiv.]: The list of (x, y) coordinate values/list-of-lists. Required. <br>
    bins [list]: The bins to use for the heatmap. Defaults to [np.arange(0, 1, 12/512), np.arange(0, 1, 12/512)]. <br>
    colorbar [bool]: Whether or not to show a colorbar. Defaults to True. <br>
    savepath [str]: The path to save the figure before showing. Defaults to None (meaning not saving).
    """

    x, y = locations[:, 0], locations[:, 1]

    plt.hist2d(x, y, bins=bins)

    if colorbar:
        plt.colorbar()
    
    if savepath is not None:
        plt.savefig(savepath)

    plt.show()

def hist(x: list, bins: int = 30, title: str = 'Histogram', xlabel: str = 'Value', savepath: str | None = None) -> None:
    """
    Creates a matplotlib histogram of a given list. Shows the plot.

    x [list]: The data to histogram. Required. <br>
    bins [int]: The number of bins to use. Defaults to 30. <br>
    title [str]: The title of the plot to use. Defaults to 'Histogram'. <br>
    xlabel [str]: The x-axis label to use. Defaults to 'Value'.
    savepath [str]: The path to save the figure before showing. Defaults to None (meaning not saving).
    """

    plt.hist(x, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.title(title)

    if savepath is not None:
        plt.savefig(savepath)
    
    plt.show()
__author__ = 'Guilherme Varela'
__date__ = '2019-10-24'

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_times(times, series, series_labels, xy_labels, title):
    """ Makes an hourly plot of series

    PARAMETERS:
    ----------
    * times: list of strings
        those are the times that will represent the x-axis

    * series: list or list of lists
        those are the values that will be plotted over y-axis, if
        it's containts are also lists than is a multi-series

    * series_labels: string or list of strings
        use string for single series or list of strings for multiple
        series.

    * xy_labels: None or list of strings
       If present expects to have two values xlabel and ylabel respectively

    * title: None or string
       A glorious title

   USAGE:
   -----
   > times = ["00:00:00", "00:15:00", "00:30:00", ..., "23:45:00"]
   > series =[[20, 300, 327.5, ... 20], [10, 45, 27, ..., 5]]
   > series_labels = ["means", "std"]
   > xy_labels = ["Times", "# vehicles"]
   > title = "Induction loop reading"
   > plot_times(times, series, series_labels, xy_labels, title)

   REFS:
   ----
   * Time plots
     See https://stackoverflow.com/questions/

     13515471/matplotlib-how-to-prevent-x-axis-labels-from-overlapping-each-other#13521621
     1574088/plotting-time-in-python-with-matplotlib#1574146
     14946371/editing-the-date-formatting-of-x-axis-tick-labels-in-matplotlib

    """
    # time_tick = table['Time'].values
    # means = table['mean'].values
    # stds = table['std'].values

    num_series = len(series_labels)
    if num_series > 2:
        raise ValueError("Only 1 or 2 series are supported")

    times = [datetime.strptime(tt, "%H:%M:%S") for tt in times]
    time_only_format = mdates.DateFormatter("%H:%M")
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(time_only_format)

    ax.plot(times, series[0], label=series_labels[0])
    if num_series == 2:
        ax.plot(times, series[1], marker='o', linestyle='--', color='r', label=series_labels[1])

    ax.set_xlabel('Time')
    ax.set_ylabel('# vehicles')

    if title:
        ax.set_title(title)

    ax.xaxis_date()
    fig.autofmt_xdate()
    plt.legend()
    plt.show()

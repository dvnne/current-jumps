import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import copy
from bisect import bisect_left
from warnings import warn

#####################
# UTILITY FUNCTIONS #
#####################

def parse_excel_data(d, voltage_col=3, field_col=6, temp_col=5):
    """
    Return DataFrame of summary data from current vs time experiments

    Parameters
    ----------
    d : string, bytes
        Single Excel file or directory of excel (*.xlsx) files with current jumps data
    voltage_col, field_col, temp_col : int, optional
        Columns (0-indexed) where voltage, field, and temperature data can be
        found
    """
    if os.path.isdir(d):
        wkbs = [f for f in os.listdir(d) if f.endswith(".xlsx")]
    elif d.endswith(".xlsx"):
        wkbs = [os.path.basename(d)]
        d = os.path.dirname(d)
    output = []
    for wkb in wkbs:
        dfs = pd.read_excel(os.path.join(d, wkb), header=None, sheet_name=None)
        for sheetname, df in dfs.items():
            avgs = df.mean()
            voltage, field, temp = 1000 * avgs[voltage_col], avgs[field_col], avgs[temp_col]
            output.append([wkb, sheetname, field, voltage, temp])
    dfout = pd.DataFrame(output, columns=["FileName", "SheetName", "Field (T)", "Voltage (mV)", "Temp (K)"])
    return dfout

def append_sort_fields(df):
    """
    Add columns `DeviceName`, `sort_field`, `sort_v`, and `sort_temp` to
    summary dataframe

    Parameters
    ----------
    df : pandas.dataFrame
        Output from `parse_excel_data`
    """
    fields = df.SheetName.apply(lambda s : s.split('_'))
    name = fields.apply(lambda a : a[0])
    field = fields.apply(lambda a : int(a[1][:-1]))
    voltage = fields.apply(lambda a : int(a[2][:-2]))
    temp = fields.apply(lambda a : int(a[3][:-1]))
    fields = pd.DataFrame({"DeviceName": name, "sort_field": field, "sort_v": voltage, "sort_temp": temp})
    # insert sort_temp columm
    df.insert(0, fields.sort_temp.name, fields.sort_temp)
    df.insert(0, fields.sort_v.name, fields.sort_v)
    df.insert(0, fields.sort_field.name, fields.sort_field)
    df.insert(0, fields.DeviceName.name, fields.DeviceName)
    return df

def plot_excel(xlsx_path, **kwargs):
    """
    Wrapper function for summary_plot for plotting all sheets in an excel
    file
    
    Parameters
    ----------
    xlsx_path: str
    **kwargs: dict
    """
    summary = append_sort_fields(parse_excel_data(xlsx_path))
    xlsxdir = os.path.dirname(xlsx_path)
    return summary_plot(summary, xlsxdir, **kwargs)



def summary_plot(summary, xlsxdir, filter=None, plot_type="hist", sort_by="temp", **kwargs):
    """
    Create stacked plot of data from a summary DataFrame
    
    Parameters
    ----------
    summary : str
        Path to a (tab-deliminated) summary file
    xlsxdir : str
        Directory where files can be found
    filter : pandas.Series, optional
        Series of bools to filter summary data for plotting
    plot_type : {"hist", "scatter"}, default "hist"
        Type of plot
    sort_by : {"temp", "voltage"}, default "temp"
        Whether to sort by temperature or bias voltage
    **kwargs : dict
        Options for `Data.from_excel`

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : list
        matplotlib.axes.Axes objects
    """
    # filter
    if filter:
        filtered = summary.loc[filter].copy()
    else:
        filtered = summary.copy()
    # sort
    if sort_by == "temp":
        sort_by = "sort_temp"
    elif sort_by == "voltage":
        sort_by = "sort_v"
    else:
        raise ValueError("Invalid entry for `sort_by`: {}".format(sort_by))
    filtered.sort_values(sort_by, ascending=False, inplace=True)
    # plot
    fig, axs = plt.subplots(len(filtered), 1, sharex=True)
    if not hasattr(axs, '__iter__'):
        axs = [axs]
    sheets, files = filtered.SheetName, filtered.FileName
    for ax, sheet, filename in zip(axs, sheets, files):
        data = Data.from_excel(os.path.join(xlsxdir, filename), sheet_name=sheet, **kwargs)
        if plot_type == "scatter":
            ax.plot(data.time, data.current, '.', label=sheet)
        elif plot_type == "hist":
            ax.hist(data.current, bins="auto", label=sheet)
    return fig, axs


# Kernel Function #
def biased_mean(window, lims, threshold=0.1, mode="full"):
    """
    Take the mean of array of values in `window` with a bias to the center
    value. Throw out values from left or right side of window until relative
    difference between center value and the mean of the left or right sides
    of the window are within `threshold` 
    """
    center_index = len(window) // 2
    center_val = window[center_index]
    win_left, win_right = window[:center_index], window[center_index + 1:]
    if mode not in {"window", "full"}:
        raise ValueError("invalid mode: %s Use 'window' or 'full'" % mode)
    if mode == "window":
        win_range = max(window) - min(window)
    if mode == "full":
        win_range = max(lims) - min(lims)
    while len(win_left) > 0 and len(win_right) > 0:
        window = np.concatenate((win_left, [center_val], win_right))
        mean_left, mean_right = np.mean(win_left), np.mean(win_right)
        diff_left, diff_right = abs(mean_left - center_val), abs(mean_right - center_val)
        if win_range == 0:
            return np.mean(window)
        else:
            reldiff = abs(diff_left - diff_right) / win_range
        if reldiff <= threshold:
            return np.mean(window)
        elif diff_left > diff_right:
            win_left = win_left[1:]
        elif diff_right > diff_left:
            win_right = win_right[:-1]
    return np.mean(np.concatenate((win_left, [center_val], win_right)))

#TODO: Fix this method (use binary operations on the pd.Series elements)
def select_data(indexfile, 
                devicename = None, 
                field = None,
                voltage = None, 
                temp = None):
    """
    Returns subset of rows from file describing individual experiments
    
    Parameters
    ----------
    indexfile : str, bytes
    devicename : str, optional
    field : {0, 5}, optional
    voltage : int, optional
    temp : int, optional

    Returns
    -------
    df : DataFrame
        Rows matching description
    """
    df = pd.read_csv(indexfile, sep = '\t')
    if devicename is not None:
        df = df.loc[lambda d : d["DeviceName"] == devicename]
    if field is not None:
        df = df.loc[lambda d : d["sort_field"] == field]
    if voltage is not None:
        df = df.loc[lambda d : d["sort_v"] == voltage]
    if temp is not None:
        df = df.loc[lambda d: d["sort_temp"] == temp]
    return df


class State:
    """
    Class for defining a current state

    Attributes
    ----------
    value : float
        average value of current state
    value_std : float
        standard deviation of cuurent state (controls variance within state)
    lifetime : float
        average lifetime of state
    lifetime_std : float
        standard deviation of state lifetime
    """
    def __init__(self, value = 0, value_std = 0.2, 
                lifetime = 100, lifetime_std = 30):
        (self.value,
        self.value_std, 
        self.lifetime,
        self.lifetime_std) = (value,
                            value_std, 
                            lifetime,
                            lifetime_std)

class Data:
    """
    Provides methods for processing current jumps data

    Attributes
    ----------
    current : array_like
        Current or conductance values
    time : array_like, optional
        Vector of time values associated with `current`. If `None` index 
        of `current` will be used
    kernel_window_length : int, default: 11
        Positive, odd integer for window length of filtering kernel
    kernel_function : function, default: np.mean
        Function to apply for filtering. Must be a function of one 
        variable that takes an array and returns a single value
    min_jump_size : int, default: 1
        Minimum length of peaks (in index units) allowed by filtering
    thresholds
    """

    def __init__(self, current, time = None,
                kernel_function = np.mean, 
                kernel_window_length = 11,
                min_jump_size = 1,
                voltage=None,
                temp=None):
        """
        Parameters
        ----------
        current : array_like
            Current or conductance values
        time : array_like, optional
            Vector of time values associated with `current`. If `None` index 
            of `current` will be used
        kernel_window_length : int, default: 11
            Positive, odd integer for window length of filtering kernel
        kernel_function : function, default: np.mean
            Function to apply for filtering. Must be a function of one 
            variable that takes an array and returns a single value
        min_jump_size : int, default: 1
            Minimum length of peaks (in index units) allowed by filtering
        voltage : np.ndarray
            Voltage data
        temp : np.ndarray
            Temperature data
        """
        self.current = np.array(current)
        if time is None:
            self.time = time
        else:
            self.time = np.array(time) - time[0]
        # Kernel Parameters # -- These should probably be properties
        self.kernel_window_length = kernel_window_length
        self.set_kernel_function(kernel_function)
        self.min_jump_size = min_jump_size
        self._thresholds = np.array([[]])
        self.voltage = voltage
        self.temp = temp

    def __len__(self):
        return len(self.current)

    @property
    def thresholds(self):
        return self._thresholds

    @thresholds.deleter
    def thresholds(self):
        self._thresholds = np.array([[]])

    @thresholds.setter
    def thresholds(self, thresholds):
        """
        State boundaries for state assignment

        One, single-valued boundary:
            `thresholds = 2.6`
        One boundary with given points in (index, value) pairs (Points will
        be connected with straight line with ends flat)
            `thresholds = [(100, 2.6), (200, 4.2), ...]`
        Multiple, single-valued boundaries:
            `thresholds = [2.6, 4.2, ...]`
        Can mix methods to define multiple boundaries:
            `thresholds = [[(100, 2.6), (200, 3)], 3, ...]]`
        Can also leave out indices to evenly space values:
            `thresholds = [[2.6, 4.3, 1.2], ...]]`
        """
        # Case: single horizontal line
        if isinstance(thresholds, (int, float)):
            self._thresholds = np.array([[thresholds] * len(self.current)])
        # Case: single curve, given indices
        elif isinstance(thresholds[0], tuple):
            self._thresholds = np.array([self.interpolate(thresholds)])
        # Case: multiple curves
        else:
            curves = []
            for elem in thresholds:
                if isinstance(elem, (int, float)):
                    curves.append([elem] * len(self.current))
                elif isinstance(elem[0], tuple):
                    curves.append(self.interpolate(elem))
                else:
                    indices = [(len(self.current) * (i+1)) // (len(elem) + 1) for i in range(len(elem))]
                    curves.append(self.interpolate(list(zip(indices, elem))))
            self._thresholds = np.array(curves)

    @property
    def kernel_function(self):
        return self._kernel_function

    def set_kernel_function(self, fn, **kwargs):
        """
        Set given function as kernel for filtering

        Parameters
        ----------
        fn : function
            Function of one or two variables: f(window [, lims] [, **kwargs]).
            Accepts array_like window and returns one value. Tuple of
            `min(self.current), max(self.current)` will be passed to lims
        **kwargs : dict
            Options to pass to `fn`
        """
        lims = min(self.current), max(self.current)
        test_window = np.ones(5)
        try:
            fn(test_window, **kwargs)
            self._kernel_function = lambda w : fn(w, **kwargs)
        except TypeError:
            fn(test_window, lims, **kwargs)
            self._kernel_function = lambda w : fn(w, lims, **kwargs)


    @property
    def kernel_window_length(self):
        return self._kernel_window_length

    @kernel_window_length.setter
    def kernel_window_length(self, n):
        if n % 2 != 1 or n < 1:
            raise ValueError("kernel_window_length must be a positive, odd integer")
        self._kernel_window_length = n

    def time_to_index(self, t):
        '''Convert time to nearest index in time data'''
        i = bisect_left(self.time, t)
        if i == len(self.time):
            msg = ("Caclulated index is equal to length of dataset." 
                    "You may be extrapolating")
            warn(msg)
        return i

    def interpolate(self, a):
        """
        Interpolate between points
        
        Parameters
        ----------
        a : array_like
            Sequence of tuples (index, value)
            
        Returns
        -------
        output : np.ndarray
            Interpolated values. Length is equal to length of data sequence
        """
        output = np.zeros(len(self.current))
        for i in range(1, len(a)):
            end, start = a[i][0],  a[i-1][0]
            seg_len = end - start
            vals = np.linspace(a[i-1][1], a[i][1], num=seg_len)
            output[start:end] = vals
        output[:a[0][0]] = [a[0][1]] * a[0][0] # first values
        output[a[-1][0]:] = [a[-1][1]] * (len(self.current) - a[-1][0]) #last values
        return output

    # TODO: Test units_are_time = True
    def fit_points(self, polyorder, pts, units_are_time=False):
        """
        Estimated baseline to correct for conductance drift
        
        Parameters
        ----------
        polyorder : int
            Order of polynomial fit
        pts : array_like
            Sequence of tuples (x, y) for polynomial fit
        units_are_time : bool, default False
            Whether x values are time or indices
        
        Returns
        -------
        fit : np.ndarray
            1D array of values fitted to `pts`
        """
        if polyorder == 0:
            return self.interpolate(pts)
        x, y = tuple(zip(*pts))
        x, y = np.array(x), np.array(y)
        polyfit = np.poly1d(np.polyfit(x, y, polyorder))
        if units_are_time:
            fit = polyfit(self.time)
        else:
            fit = polyfit(np.arange(len(self.current)))
        return fit

    def select_points(self, npoints):
        """
        Pick `npoints` evenly spaced points in data for baseline fitting

        Returns
        -------
        indices : np.ndarray
        pts : np.ndarray
        """
        segments = np.arange(npoints)
        pts = []
        indices = []
        for s in segments:
            start = s*len(self.current) // npoints
            end = (s + 1)*len(self.current) // npoints
            seg = self.current[start:end]
            pts.append(np.median(seg))
            indices.append((end + start) // 2)
        indices, pts = np.array(indices), np.array(pts)
        points = list(zip(indices, pts))
        return points

    def auto_baseline(self, polyorder, npoints):
        """
        Automaticaly create polynomial baseline with evenly spaced sample points
        
        Parameters
        ----------
        polyorder : int
            Order of polynomial fit
        npoints : int
            Number of points for sampling
            
        Returns
        -------
        baseline : np.ndarray
            Estimated baseline fit to data
        """
        pts = self.select_points(npoints)
        baseline = self.fit_points(polyorder, pts, 
                                    units_are_time=False)
        return baseline

    def correct_baseline(self, baseline):
        """
        Correct for overall drift according to array `baseline`
        
        Returns
        -------
        new_current : np.ndarray
        """
        if len(baseline) != len(self.current):
            raise ValueError("baseline must be same length as data")
        new_current = self.current - baseline + np.median(baseline)
        return new_current

    def export_corrected(self, baseline):
        """
        Create new Data object with baseline corrected
        
        Parameters
        ----------
        baseline : np.ndarray
        
        Returns
        -------
        data : current_jumps.Data
        """
        new_current = self.correct_baseline(baseline)
        data = copy(self)
        del data.thresholds
        data.current = new_current
        return data

    def filtered(self):
        """
        Filter current data according to kernel_function and
        kernel_window_length. Values at the begining and end are padded by
        using the first and last values as a mirror axis
        """
        halfwindow = self.kernel_window_length // 2
        output = np.zeros(len(self.current))
        imax = len(self.current) - 1
        # fill middle states
        for i in range(len(self.current)):
            window_end = i + self.kernel_window_length - 1
            if window_end <= imax:
                window = self.current[i:i+self.kernel_window_length]
                output[i+halfwindow] = self.kernel_function(window)
        # fill beginning/end states
        first = output[halfwindow]
        last = output[-(halfwindow + 1)]
        for i in range(halfwindow):
            j = (self.kernel_window_length - 1) - i # 'symmetry' index
            output[i] = first - (output[j] - first)
            k = imax - i
            l = (imax - self.kernel_window_length) + i
            output[k] = last - (output[l] - last)
        return output

    def labeled_data(self) -> dict:
        """
        Current and time values organized by state label index
            
        Returns
        -------
        labeled_data : dict
            labeled_data[state_label] -> 
            np.array([[time1, time2, ...], [current1, current2, ...]])
        """
        labeled_data = dict()
        state_labels = self.state_labels()
        for state_label in set(state_labels):
            indices, current = [], []
            for i, l in enumerate(state_labels):
                if l == state_label:
                    if self.time is None:
                        indices.append(i)
                    else:
                        indices.append(self.time[i])
                    current.append(self.current[i])
            labeled_data[state_label] = np.array([indices, current])
        return labeled_data

    def state_labels(self):
        """
        Assign a state label (integer, zero-indexed) for each current
        value in `self.current`
        
        Returns 
        -------
        assignments : list
            state label for each current value
        """
        if len(self.thresholds[0]) < 1:
            msg = "No thresholds specified. Assign using `self.thresholds`"
            raise ValueError(msg)
        assignments = []
        thresholds = sorted(self.thresholds, key = lambda x : x[0])
        for i, c in enumerate(self.filtered()):
            assigned_state = 0
            bounds = [t[i] for t in thresholds]
            for b in bounds:
                if c > b:
                    assigned_state += 1
            assignments.append(assigned_state)
        assignments = self.remove_short_jumps(assignments)
        return assignments

    # TODO: Make private method
    def remove_short_jumps(self, assignments:list):
        """
        Reassigns assigned state labels to only allow lifetimes greater than
        or equal to `self.min_jump_size`
        """
        has_short_jumps = True
        while has_short_jumps:
            has_short_jumps = False
            steps = [0]
            for i in range(1, len(assignments)):
                if assignments[i] != assignments[i-1]:
                    steps.append(i)
            steps.append(len(assignments))
            for j in range(1, len(steps) - 1):
                dist_left, dist_right = steps[j] - steps[j-1], steps[j+1] - steps[j]
                step0, step1, step2 = steps[j-1], steps[j], steps[j+1]
                # Find left edge of section to modify
                if dist_left < self.min_jump_size:
                    has_short_jumps = True
                    selection, next_step = step0, step1
                    if dist_right < self.min_jump_size and dist_right < dist_left:
                        selection, next_step = step1, step2
                    # Edge Cases
                    if selection == 0:
                        assignments[:next_step] = [assignments[next_step]] * len(assignments[:next_step])
                    elif next_step == len(assignments):
                        assignments[selection:] = [assignments[selection-1]] * len(assignments[selection:])
                    # Remove short jumps
                    else:
                        level_left, level_right = assignments[selection-1], assignments[next_step]
                        if abs(level_left - assignments[selection]) <= abs(level_right - assignments[selection]):
                            assignments[selection:next_step] = [level_left] * (next_step - selection)
                        else:
                            assignments[selection:next_step] = [level_right] * (next_step - selection)
                    # Recompute steps
                    break
        if steps[-1] - steps[-2] < self.min_jump_size: # edge case
            level_left = assignments[steps[-2] - 1]
            for i in range(steps[-2], len(assignments)):
                assignments[i] = level_left
        return assignments

    def lifetime_data(self):
        """Returns dict of StateData objects indexed by state labels"""
        d = dict()
        state_labels = self.state_labels()
        for state_label in set(state_labels):
            lifetimes = self.count_runs(state_labels, state_label)
            values = [self.current[i] for i in range(len(state_labels)) if state_labels[i] == state_label]
            d[state_label] = StateData(values, lifetimes)
        return d

    # TODO: Make private method
    def count_runs(self, assignments, target_state) -> list:
        steps, labels = [0], [assignments[0]]
        for i in range(1, len(assignments)):
            if assignments[i] != assignments[i-1]:
                steps.append(i)
                labels.append(assignments[i])
        steps.append(len(assignments))
        lifetimes = []
        for j in range(1, len(steps)):
            if labels[j-1] == target_state:
                interval = steps[j] - steps[j - 1]
                if self.time is None:
                    lifetimes.append(interval)
                else:
                    dt = (self.time[-1] - self.time[0])/(len(self.time) - 1)
                    lifetimes.append(interval * dt)
        return lifetimes

    def slice(self, start, stop, units_are_time=False):
        '''
        Returns new object where data fields are truncated between
        [start, stop)

        Parameters
        ----------
        start, stop : int
            Bounds for new dataset
        units_are_time : bool, default: False
            Whether start, stop given in index or time units
        '''
        if units_are_time:
            start, stop = self.time_to_index(start), self.time_to_index(stop)
        cpy = copy(self)
        cpy.current = self.current[start:stop]
        if self.time is not None:
            cpy.time = self.time[start:stop]
        if self.thresholds.size > 0:
            cpy._thresholds = self.thresholds[:, start:stop]
        if self.voltage is not None:
            cpy.voltage = self.voltage[start:stop]
        if self.temp is not None:
            cpy.temp = self.temp[start:stop]
        return cpy

    @classmethod
    def from_excel(cls, filename, sheet_name=0, **kwargs):
        df = pd.read_excel(filename, sheet_name=sheet_name, header=0)
        return cls.from_dataframe(df, **kwargs)

    @classmethod
    def from_dataframe(cls, df, use_conductance=False, 
                        time_col=0, current_col=4, vsd_col=3,
                        voltage_col=3, temp_col=5, **kwargs):
        (time, current, vsd, voltage, temp) = (df.iloc[:, time_col].to_numpy(), 
                                                df.iloc[:, current_col].to_numpy(), 
                                                df.iloc[:, vsd_col].to_numpy(),
                                                df.iloc[:, voltage_col].to_numpy(),
                                                df.iloc[:, temp_col].to_numpy())
        if use_conductance:
            conductance = current / vsd
            return cls(conductance, time=time, voltage=voltage, temp=temp, **kwargs)
        else:
            return cls(current, time=time, voltage=voltage, temp=temp, **kwargs)

    @classmethod
    def from_index(cls, index_df, row_i, xlsxdir, **kwargs):
        dataset = index_df.loc[row_i]
        f = os.path.join(xlsxdir, dataset.FileName)
        sheet = dataset.SheetName
        return cls.from_excel(f, sheet_name=sheet, **kwargs)

    # INCOMPLETE #
    def autocorrelation(self):
        variance = np.var(self.current)
        c = self.current - np.mean(self.current)
        return np.correlate(c, c, mode = 'same')

    #####################################
    ############# PLOTTING ##############
    #####################################

    def plot_filtered(self, **kwargs):
        ax = plt.gca()
        if self.time is None:
            ax.plot(self.filtered(), **kwargs)
        else:
            ax.plot(self.time, self.filtered(), **kwargs)

    def plot_data(self, use_index_for_time=False, **kwargs):
        ax = plt.gca()
        if use_index_for_time:
            time = np.arange(len(self.current))
        elif self.time is not None:
            time = self.time
        else:
            time = np.arange(len(self.current))
        ax.plot(time, self.current, '.', **kwargs)
        
    def histogram(self, **kwargs):
        ax = plt.gca()
        return ax.hist(self.current, **kwargs)

    def filtered_histogram(self, **kwargs):
        ax = plt.gca()
        return ax.hist(self.filtered(), **kwargs)

    def plot_labeled_data(self, plot_filtered=True):
        ax = plt.gca()
        for label, data in self.labeled_data().items():
            ax.plot(data[0], data[1], '.', label = label)
        if self.time is None:
            if plot_filtered:
                ax.plot(self.filtered(), c = 'grey', lw = 0.5, marker = '.', ms = 1)
            for t in self.thresholds:
                ax.plot(t, c = 'k', ls = '--')
        else:
            if plot_filtered:
                ax.plot(self.time, self.filtered(), c = 'grey', lw = 0.5, marker = '.', ms = 1)
            for t in self.thresholds:
                ax.plot(self.time, t, c = 'k', ls = '--')

    def plot_lifetimes(self):
        ax = plt.gca()
        ld_dict = self.lifetime_data()
        for state, data in ld_dict.items():
            ax.boxplot(data.lifetimes, positions = [state])
        
    def view_data(self, show_now=True):
        fig, axs = plt.subplots(1, 2)
        axs[0].plot(self.current, '.')
        axs[1].hist(self.current, bins='auto')
        if show_now:
            plt.show()

    def plot_baseline(self, baseline):
        fig, axs = plt.subplots(2, 1, sharex=True)
        if self.time is None:
            time = np.arange(len(self.current))
        else:
            time = self.time
        # Bottom: Raw Data with Baseline
        axs[1].plot(time, self.current, '.')
        axs[1].plot(time, baseline, '-', lw=3)
        # Top: Preview of Correction
        corrected = self.correct_baseline(baseline)
        axs[0].plot(time, corrected, '.', c='tab:green')

    def plot_points(self, pts, use_index_for_time=False,
                    line_style= '', marker_size=10, marker_style='.'):
        ax = plt.gca()
        indices, values = tuple(zip(*pts))
        if self.time is None:
            time = indices
        elif use_index_for_time:
            time = indices
        else:
            time = [self.time[i] for i in indices]
        self.plot_data(use_index_for_time)
        ax.plot(time, values, 
                marker=marker_style, ms=marker_size, ls=line_style)

class SimulatedData(Data):
    def __init__(self, *states, N = 1000):
        self.generate_data(*states, N = N)
        super().__init__(self.current)
        self.states = states

    def generate_data(self, *states, N = 1000):
        data = []
        groundtruth = []
        groundtruth_states = []
        state_index = np.random.randint(0, len(states))
        n = 0 # array index
        while n < N:
            # current state
            c_state = states[state_index]
            # pick random state lifetime
            lifetime = int(np.random.normal(loc = c_state.lifetime, 
                                            scale = c_state.lifetime_std))
            lifetime = max(1, lifetime)
            groundtruth.extend([c_state.value] * lifetime)
            groundtruth_states.extend([state_index] * lifetime)
            # add random offsets to data
            jitter = lambda x : np.random.normal(loc = x, 
                                                scale = c_state.value_std)
            newdata = map(jitter, [c_state.value] * lifetime)
            data.extend(newdata)
            # pick a new state that isn't the current state
            state_options = list(range(0, len(states)))
            state_options.pop(state_index)
            state_index = np.random.choice(state_options)
            # update index
            n += lifetime
        self.current = np.array(data) # this is redundant
        self.groundtruth = np.array(groundtruth)
        self.groundtruth_states = groundtruth_states

    def plot_data(self, **kwargs):
        ax = plt.gca()
        ax.plot(self.groundtruth, 'k-')
        ax.plot(self.current, '.', **kwargs)
        ax.set_xlabel("Index (Time)")
        ax.set_ylabel('"Current"')

    def view_data(self):
        fig, axs = plt.subplots(1, 2)
        plt.sca(axs[0])
        axs[0].set_title("Simulation")
        self.plot_data()
        plt.sca(axs[1])
        axs[1].set_title("Frequency")
        self.histogram(bins = 50)


class StateData:
    def __init__(self, values, lifetimes):
        self.values = np.array(values)
        self.lifetimes = np.array(lifetimes)

    def __str__(self):
        params = ["Average Current ({:d})".format(len(self.values)), 
                    "Average Lifetime ({:d})".format(len(self.lifetimes)), 
                    "Std. Dev. Current", 
                    "Std. Dev. Lifetime"]
        values  = [self.avg_value(), self.avg_lifetime(), self.std_value(), self.std_lifetime()]
        max_param_length = len(max(params, key = len))
        lines = []
        for p, val in zip(params, values):
            pad = ' ' * (max_param_length - len(p))
            lines.append(pad + p + ':' + '{0:11.3g}'.format(val))
        return '\n'.join(lines)

    def avg_value(self):
        return np.mean(self.values)

    def std_value(self):
        return np.std(self.values)

    def avg_lifetime(self):
        return np.mean(self.lifetimes)

    def std_lifetime(self):
        return np.std(self.lifetimes)

    def rates(self):
        return 1/self.lifetimes
    
    def avg_rate(self):
        return np.mean(self.rates())

    def std_rate(self):
        return np.std(self.rates())

# -*- coding: utf-8 -*-
from __future__ import division
import copy
import fnmatch
#import math
import os
import io
import numpy as np
import warnings
from myobspy.compatibility import round_away
from myobspy.obspycore.trace import Trace
from myobspy.obspycore.utcdatetime import UTCDateTime

from collections import OrderedDict


class NamedTemporaryFile(io.BufferedIOBase):
    """ Weak replacement for the Python's tempfile.TemporaryFile """
    def __init__(self, dir=None, suffix='.tmp', prefix='myobspy-'):
        fd, self.name = tempfile.mkstemp(dir=dir, prefix=prefix, suffix=suffix)
        self._fileobj = os.fdopen(fd, 'w+b', 0)  # 0 -> do not buffer

    def read(self, *args, **kwargs):
        return self._fileobj.read(*args, **kwargs)

    def write(self, *args, **kwargs):
        return self._fileobj.write(*args, **kwargs)

    def seek(self, *args, **kwargs):
        self._fileobj.seek(*args, **kwargs)
        return self._fileobj.tell()

    def tell(self, *args, **kwargs):
        return self._fileobj.tell(*args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # @UnusedVariable
        self.close()  # flush internal buffer
        self._fileobj.close()
        os.remove(self.name)



class Stream(object):
    def __init__(self, traces=None):
        self.traces = []
        if isinstance(traces, Trace):
            traces = [traces]
        if traces:
            self.traces.extend(traces)

    def __add__(self, other):
        """ Add two streams or a stream with a single trace """
        if isinstance(other, Trace):
            other = Stream([other])
        if not isinstance(other, Stream):
            raise TypeError
        traces = self.traces + other.traces
        return self.__class__(traces=traces)

    def __iadd__(self, other):
        """ Add two streams with self += other """
        if isinstance(other, Trace):
            other = Stream([other])
        if not isinstance(other, Stream):
            raise TypeError
        self.extend(other.traces)
        return self

    def __mul__(self, num):
        """ Create a new Stream containing num copies of this stream"""
        if not isinstance(num, int):
            raise TypeError("Integer expected")
        st = Stream()
        for _i in range(num):
            st += self.copy()
        return st

    def __iter__(self):
        """ Return a robust iterator for stream.traces """
        return list(self.traces).__iter__()

    def __nonzero__(self):
        """ A Stream is considered zero if has no Traces """
        return bool(len(self.traces))

    def __len__(self):
        """ Return the number of Traces in the Stream object """
        return len(self.traces)

    count = __len__

    def __setitem__(self, index, trace):
        self.traces.__setitem__(index, trace)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(traces=self.traces.__getitem__(index))
        else:
            return self.traces.__getitem__(index)

    def __delitem__(self, index):
        return self.traces.__delitem__(index)

    def __getslice__(self, i, j, k=1):
        return self.__class__(traces=self.traces[max(0, i):max(0, j):k])

    def append(self, trace):
        """ Append a single Trace object to the current Stream object """
        if isinstance(trace, Trace):
            self.traces.append(trace)
        else:
            msg = 'Append only supports a single Trace object as an argument.'
            raise TypeError(msg)
        return self

    def copy(self):
        """ Return a deepcopy of the Stream object """
        return copy.deepcopy(self)

    def extend(self, trace_list):
        """ Extend the current Stream object with a list of Trace objects """
        if isinstance(trace_list, list):
            for _i in trace_list:
                # Make sure each item in the list is a trace.
                if not isinstance(_i, Trace):
                    msg = 'Extend only accepts a list of Trace objects.'
                    raise TypeError(msg)
            self.traces.extend(trace_list)
        elif isinstance(trace_list, Stream):
            self.traces.extend(trace_list.traces)
        else:
            msg = 'Extend only supports a list of Trace objects as argument.'
            raise TypeError(msg)
        return self

    def insert(self, position, object):
        """ Insert either a single Trace or a list of Traces before index """
        if isinstance(object, Trace):
            self.traces.insert(position, object)
        elif isinstance(object, list):
            # Make sure each item in the list is a trace.
            for _i in object:
                if not isinstance(_i, Trace):
                    msg = 'Trace object or a list of Trace objects expected!'
                    raise TypeError(msg)
            # Insert each item of the list.
            for _i in range(len(object)):
                self.traces.insert(position + _i, object[_i])
        elif isinstance(object, Stream):
            self.insert(position, object.traces)
        else:
            msg = 'Only accepts a Trace object or a list of Trace objects.'
            raise TypeError(msg)
        return self

    def pop(self, index=(-1)):
        """ Remove and return the Trace object specified by index from the Stream """
        return self.traces.pop(index)

    def trim(self, starttime=None, endtime=None, pad=False,
             nearest_sample=True, fill_value=None):
        """ Cut all traces of this Stream object to given start and end time """
        if not self:
            return
        # select start/end time fitting to a sample point of the first trace
        if nearest_sample:
            tr = self.traces[0]
            if starttime:
                delta = round_away(
                    (starttime - tr.stats.starttime) * tr.stats.sampling_rate)
                starttime = tr.stats.starttime + delta * tr.stats.delta
            if endtime:
                delta = round_away(
                    (endtime - tr.stats.endtime) * tr.stats.sampling_rate)
                # delta is negative!
                endtime = tr.stats.endtime + delta * tr.stats.delta
        for trace in self.traces:
            trace.trim(starttime, endtime, pad=pad,
                       nearest_sample=nearest_sample, fill_value=fill_value)
        # remove empty traces after trimming
        self.traces = [_i for _i in self.traces if _i.stats.npts]
        return self

    def _ltrim(self, starttime, pad=False, nearest_sample=True):
        """ Cut all traces of this Stream object to given start time """
        for trace in self.traces:
            trace.trim(starttime=starttime, pad=pad,
                       nearest_sample=nearest_sample)
        # remove empty traces after trimming
        self.traces = [tr for tr in self.traces if tr.stats.npts]
        return self

    def _rtrim(self, endtime, pad=False, nearest_sample=True):
        """ Cut all traces of this Stream object to given end time """
        for trace in self.traces:
            trace.trim(endtime=endtime, pad=pad, nearest_sample=nearest_sample)
        # remove empty traces after trimming
        self.traces = [tr for tr in self.traces if tr.stats.npts]
        return self

    def select(self, network=None, station=None, location=None, channel=None,
               sampling_rate=None, npts=None, component=None, id=None):
        """ 
        Return new Stream object only with these traces that match the given
        stats criteria (e.g. all traces with ``channel="EHZ"``).
        """
        # make given component letter uppercase (if e.g. "z" is given)
        if component and channel:
            component = component.upper()
            channel = channel.upper()
            if channel[-1] != "*" and component != channel[-1]:
                msg = "Selection criteria for channel and component are " + \
                      "mutually exclusive!"
                raise ValueError(msg)
        traces = []
        for trace in self:
            # skip trace if any given criterion is not matched
            if id and not fnmatch.fnmatch(trace.id.upper(), id.upper()):
                continue
            if network is not None:
                if not fnmatch.fnmatch(trace.stats.network.upper(),
                                       network.upper()):
                    continue
            if station is not None:
                if not fnmatch.fnmatch(trace.stats.station.upper(),
                                       station.upper()):
                    continue
            if location is not None:
                if not fnmatch.fnmatch(trace.stats.location.upper(),
                                       location.upper()):
                    continue
            if channel is not None:
                if not fnmatch.fnmatch(trace.stats.channel.upper(),
                                       channel.upper()):
                    continue
            if sampling_rate is not None:
                if float(sampling_rate) != trace.stats.sampling_rate:
                    continue
            if npts is not None and int(npts) != trace.stats.npts:
                continue
            if component is not None:
                if len(trace.stats.channel) < 3:
                    continue
                if not fnmatch.fnmatch(trace.stats.channel[-1].upper(),
                                       component.upper()):
                    continue
            traces.append(trace)
        return self.__class__(traces=traces)

    def verify(self):
        """ Verify all traces of current Stream against available meta data """
        for trace in self:
            trace.verify()
        return self

    def _mergeChecks(self):
        """
        Sanity checks for merging.
        """
        sr = {}
        dtype = {}
        calib = {}
        for trace in self.traces:
            # skip empty traces
            if len(trace) == 0:
                continue
            # Check sampling rate.
            sr.setdefault(trace.id, trace.stats.sampling_rate)
            if trace.stats.sampling_rate != sr[trace.id]:
                msg = "Can't merge traces with same ids but differing " + \
                      "sampling rates!"
                raise Exception(msg)
            # Check dtype.
            dtype.setdefault(trace.id, trace.data.dtype)
            if trace.data.dtype != dtype[trace.id]:
                msg = "Can't merge traces with same ids but differing " + \
                      "data types!"
                raise Exception(msg)
            # Check calibration factor.
            calib.setdefault(trace.id, trace.stats.calib)
            if trace.stats.calib != calib[trace.id]:
                msg = "Can't merge traces with same ids but differing " + \
                      "calibration factors.!"
                raise Exception(msg)

    def max(self):
        """
        Get the values of the absolute maximum amplitudes of all traces in the
        stream. See :meth:`~obspy.core.trace.Trace.max`.
        """
        return [tr.max() for tr in self]

    def _cleanup(self, misalignment_threshold=1e-2):
        """ Merge consistent trace objects but leave everything else alone """
        # first of all throw away all empty traces
        self.traces = [_i for _i in self.traces if _i.stats.npts]
        # check sampling rates and dtypes
        try:
            self._mergeChecks()
        except Exception as e:
            if "Can't merge traces with same ids but" in str(e):
                msg = "Incompatible traces (sampling_rate, dtype, ...) " + \
                      "with same id detected. Doing nothing."
                warnings.warn(msg)
                return
        # order matters!
        self.sort(keys=['network', 'station', 'location', 'channel',
                        'starttime', 'endtime'])
        # build up dictionary with lists of traces with same ids
        traces_dict = {}
        # using pop() and try-except saves memory
        try:
            while True:
                trace = self.traces.pop(0)
                # add trace to respective list or create that list
                traces_dict.setdefault(trace.id, []).append(trace)
        except IndexError:
            pass
        # clear traces of current stream
        self.traces = []
        # loop through ids
        for id_ in traces_dict.keys():
            trace_list = traces_dict[id_]
            cur_trace = trace_list.pop(0)
            delta = cur_trace.stats.delta
            allowed_micro_shift = misalignment_threshold * delta
            # work through all traces of same id
            while trace_list:
                trace = trace_list.pop(0)
                gap = trace.stats.starttime - (cur_trace.stats.endtime + delta)
                if misalignment_threshold > 0 and gap <= allowed_micro_shift:
                    misalignment = gap % delta
                    if misalignment != 0:
                        misalign_percentage = misalignment / delta
                        if (misalign_percentage <= misalignment_threshold or
                                misalign_percentage >=
                                1 - misalignment_threshold):
                            # now we align the sampling points of both traces
                            trace.stats.starttime = (
                                cur_trace.stats.starttime +
                                round((trace.stats.starttime -
                                       cur_trace.stats.starttime) / delta) *
                                delta)
                #
                subsample_shift_percentage = (
                    trace.stats.starttime.timestamp -
                    cur_trace.stats.starttime.timestamp) % delta / delta
                subsample_shift_percentage = min(
                    subsample_shift_percentage, 1 - subsample_shift_percentage)
                if (trace.stats.starttime <= cur_trace.stats.endtime and
                        subsample_shift_percentage < misalignment_threshold):
                    # check if common time slice [t1 --> t2] is equal:
                    t1 = trace.stats.starttime
                    t2 = min(cur_trace.stats.endtime, trace.stats.endtime)
                    # if consistent: add them together
                    if np.array_equal(cur_trace.slice(t1, t2).data,
                                      trace.slice(t1, t2).data):
                        cur_trace += trace
                    # if not consistent: leave them alone
                    else:
                        self.traces.append(cur_trace)
                        cur_trace = trace
                # traces are perfectly adjacent: add them together
                elif trace.stats.starttime == cur_trace.stats.endtime + \
                        cur_trace.stats.delta:
                    cur_trace += trace
                # no common parts (gap):
                # leave traces alone and add current to list
                else:
                    self.traces.append(cur_trace)
                    cur_trace = trace
            self.traces.append(cur_trace)
        self.traces = [tr for tr in self.traces if tr.stats.npts]
        return self

    def merge(self, method=0, fill_value=None, interpolation_samples=0,
              **kwargs):
        """ Merge ObsPy Trace objects with same IDs
        :param method: Methodology to handle overlaps/gaps of traces. Defaults
            to ``0``.
            See :meth:`obspy.core.trace.Trace.__add__` for details on
            methods ``0`` and ``1``,
            see :meth:`obspy.core.stream.Stream._cleanup` for details on
            method ``-1``. Any merge operation performs a cleanup merge as
            a first step (method ``-1``).
        :type fill_value: int, float, str or ``None``, optional
        :param fill_value: Fill value for gaps. Defaults to ``None``. Traces
            will be converted to NumPy masked arrays if no value is given and
            gaps are present. The value ``'latest'`` will use the latest value
            before the gap. If value ``'interpolate'`` is provided, missing
            values are linearly interpolated (not changing the data
            type e.g. of integer valued traces). Not used for ``method=-1``.
        :type interpolation_samples: int, optional
        :param interpolation_samples: Used only for ``method=1``. It specifies
            the number of samples which are used to interpolate between
            overlapping traces. Default to ``0``. If set to ``-1`` all
            overlapping samples are interpolated.

        Importing waveform data containing gaps or overlaps results into
        a :class:`~obspy.core.stream.Stream` object with multiple traces having
        the same identifier. This method tries to merge such traces inplace,
        thus returning nothing. Merged trace data will be converted into a
        NumPy :class:`~numpy.ma.MaskedArray` type if any gaps are present. This
        behavior may be prevented by setting the ``fill_value`` parameter.
        The ``method`` argument controls the handling of overlapping data
        values.
        """
        def listsort(order, current):
            """ Helper method for keeping trace's ordering """
            try:
                return order.index(current)
            except ValueError:
                return -1
        #
        self._cleanup(**kwargs)
        if method == -1: return
        # check sampling rates and dtypes
        self._mergeChecks()
        # remember order of traces
        order = [id(i) for i in self.traces]
        # order matters!
        self.sort(keys=['network', 'station', 'location', 'channel',
                        'starttime', 'endtime'])
        # build up dictionary with with lists of traces with same ids
        traces_dict = {}
        # using pop() and try-except saves memory
        try:
            while True:
                trace = self.traces.pop(0)
                # skip empty traces
                if len(trace) == 0:
                    continue
                _id = trace.getId()
                if _id not in traces_dict:
                    traces_dict[_id] = [trace]
                else:
                    traces_dict[_id].append(trace)
        except IndexError:
            pass
        # clear traces of current stream
        self.traces = []
        # loop through ids
        for _id in traces_dict.keys():
            cur_trace = traces_dict[_id].pop(0)
            # loop through traces of same id
            for _i in range(len(traces_dict[_id])):
                trace = traces_dict[_id].pop(0)
                # disable sanity checks because there are already done
                cur_trace = cur_trace.__add__(
                    trace, method, fill_value=fill_value, sanity_checks=False,
                    interpolation_samples=interpolation_samples)
            self.traces.append(cur_trace)

        # trying to restore order, newly created traces are placed at
        # start
        self.traces.sort(key=lambda x: listsort(order, id(x)))
        return self

    def sort(self, keys=['network', 'station', 'location', 'channel',
                         'starttime', 'endtime'], reverse=False):
        """ Sort the traces in the Stream object """
        # check if list
        msg = "keys must be a list of strings. Always available items to " + \
            "sort after: \n'network', 'station', 'channel', 'location', " + \
            "'starttime', 'endtime', 'sampling_rate', 'npts', 'dataquality'"
        if not isinstance(keys, list):
            raise TypeError(msg)
        # Loop over all keys in reversed order.
        for _i in keys[::-1]:
            self.traces.sort(key=lambda x: x.stats[_i], reverse=reverse)
        return self

    def remove(self, trace):
        """ Remove the first occurrence of the specified Trace in Stream """
        self.traces.remove(trace)
        return self

    def __str__(self, extended=False):
        """ Return short summary string of the current stream """
        # get longest id
        if self.traces:
            id_length = self and max(len(tr.id) for tr in self) or 0
        else:
            id_length = 0
        out = str(len(self.traces)) + ' Trace(s) in Stream:\n'
        if len(self.traces) <= 20 or extended is True:
            out = out + "\n".join([_i.__str__(id_length) for _i in self])
        else:
            out = out + "\n" + self.traces[0].__str__() + "\n" + \
                '...\n(%i other traces)\n...\n' % (len(self.traces) - 2) + \
                self.traces[-1].__str__() + '\n\n[Use "print(' + \
                'Stream.__str__(extended=True))" to print all Traces]'
        return out


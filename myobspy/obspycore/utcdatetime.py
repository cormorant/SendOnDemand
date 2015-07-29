# -*- coding: utf-8 -*-
"""
Module containing a UTC-based datetime class.
"""
from __future__ import division

import datetime
import math
import time


TIMESTAMP0 = datetime.datetime(1970, 1, 1, 0, 0)


class UTCDateTime(object):
    """ A UTC-based datetime object """
    timestamp = 0.0
    DEFAULT_PRECISION = 3

    def __init__(self, *args, **kwargs):
        """ Creates a new UTCDateTime object """
        # set default precision
        self.precision = kwargs.pop('precision', self.DEFAULT_PRECISION)
        # iso8601 flag
        iso8601 = kwargs.pop('iso8601', False) is True
        # check parameter
        if len(args) == 0 and len(kwargs) == 0:
            # use current time if no time is given
            self.timestamp = time.time()
            return
        elif len(args) == 1 and len(kwargs) == 0:
            value = args[0]
            # check types
            try:
                # got a timestamp
                self.timestamp = value.__float__()
                return
            except:
                pass
            if isinstance(value, datetime.datetime):
                # got a Python datetime.datetime object
                self._fromDateTime(value)
                return
            elif isinstance(value, datetime.date):
                # got a Python datetime.date object
                dt = datetime.datetime(value.year, value.month, value.day)
                self._fromDateTime(dt)
                return
            elif isinstance(value, (bytes, str)):
                if not isinstance(value, (str, str)):
                    value = value.decode()
                # got a string instance
                value = value.strip()
                # check for ISO8601 date string
                if value.count("T") == 1 or iso8601:
                    try:
                        self.timestamp = self._parseISO8601(value).timestamp
                        return
                    except:
                        if iso8601:
                            raise
                # try to apply some standard patterns
                value = value.replace('T', ' ')
                value = value.replace('_', ' ')
                value = value.replace('-', ' ')
                value = value.replace(':', ' ')
                value = value.replace(',', ' ')
                value = value.replace('Z', ' ')
                value = value.replace('W', ' ')
                # check for ordinal date (julian date)
                parts = value.split(' ')
                # check for patterns
                if len(parts) == 1 and len(value) == 7 and value.isdigit():
                    # looks like an compact ordinal date string
                    pattern = "%Y%j"
                elif len(parts) > 1 and len(parts[1]) == 3 and \
                        parts[1].isdigit():
                    # looks like an ordinal date string
                    value = ''.join(parts)
                    if len(parts) > 2:
                        pattern = "%Y%j%H%M%S"
                    else:
                        pattern = "%Y%j"
                else:
                    # some parts should have 2 digits
                    for i in range(1, min(len(parts), 6)):
                        if len(parts[i]) == 1:
                            parts[i] = '0' + parts[i]
                    # standard date string
                    value = ''.join(parts)
                    if len(value) > 8:
                        pattern = "%Y%m%d%H%M%S"
                    else:
                        pattern = "%Y%m%d"
                ms = 0
                if '.' in value:
                    parts = value.split('.')
                    value = parts[0].strip()
                    try:
                        ms = float('.' + parts[1].strip())
                    except:
                        pass
                # all parts should be digits now - here we filter unknown
                # patterns and pass it directly to Python's  datetime.datetime
                if not ''.join(parts).isdigit():
                    dt = datetime.datetime(*args, **kwargs)
                    self._fromDateTime(dt)
                    return
                dt = datetime.datetime.strptime(value, pattern)
                self._fromDateTime(dt, ms)
                return
        # check for ordinal/julian date kwargs
        if 'julday' in kwargs:
            if 'year' in kwargs:
                # year given as kwargs
                year = kwargs['year']
            elif len(args) == 1:
                # year is first (and only) argument
                year = args[0]
            try:
                temp = "%4d%03d" % (int(year),
                                    int(kwargs['julday']))
                dt = datetime.datetime.strptime(temp, '%Y%j')
            except:
                pass
            else:
                kwargs['month'] = dt.month
                kwargs['day'] = dt.day
                kwargs.pop('julday')

        # check if seconds are given as float value
        if len(args) == 6 and isinstance(args[5], float):
            _frac, _sec = math.modf(round(args[5], 6))
            kwargs['microsecond'] = int(_frac * 1e6)
            kwargs['second'] = int(_sec)
            args = args[0:5]
        dt = datetime.datetime(*args, **kwargs)
        self._fromDateTime(dt)

    def _set(self, **kwargs):
        """
        Sets current timestamp using kwargs.
        """
        year = kwargs.get('year', self.year)
        month = kwargs.get('month', self.month)
        day = kwargs.get('day', self.day)
        hour = kwargs.get('hour', self.hour)
        minute = kwargs.get('minute', self.minute)
        second = kwargs.get('second', self.second)
        microsecond = kwargs.get('microsecond', self.microsecond)
        julday = kwargs.get('julday', None)
        if julday:
            self.timestamp = UTCDateTime(year=year, julday=julday, hour=hour,
                                         minute=minute, second=second,
                                         microsecond=microsecond).timestamp
        else:
            self.timestamp = UTCDateTime(year, month, day, hour, minute,
                                         second, microsecond).timestamp

    def _fromDateTime(self, dt, ms=0):
        """
        Use Python datetime object to set current time.

        :type dt: :class:`datetime.datetime`
        :param dt: Python datetime object.
        :type ms: float
        :param ms: extra seconds to add to current UTCDateTime object.
        """
        # see datetime.timedelta.total_seconds
        try:
            td = (dt - TIMESTAMP0)
        except TypeError:
            td = (dt.replace(tzinfo=None) - dt.utcoffset()) - TIMESTAMP0
        self.timestamp = (td.microseconds + (td.seconds + td.days * 86400) *
                          1000000) / 1000000.0 + ms

    @staticmethod
    def _parseISO8601(value):
        """
        Parses an ISO8601:2004 date time string.
        """
        # remove trailing 'Z'
        value = value.replace('Z', '')
        # split between date and time
        try:
            (date, time) = value.split("T")
        except:
            date = value
            time = ""
        # remove all hyphens in date
        date = date.replace('-', '')
        # remove colons in time
        time = time.replace(':', '')
        # guess date pattern
        length_date = len(date)
        if date.count('W') == 1 and length_date == 8:
            # we got a week date: YYYYWwwD
            # remove week indicator 'W'
            date = date.replace('W', '')
            date_pattern = "%Y%W%w"
            year = int(date[0:4])
            # [Www] is the week number prefixed by the letter 'W', from W01
            # through W53.
            # strpftime %W == Week number of the year (Monday as the first day
            # of the week) as a decimal number [00,53]. All days in a new year
            # preceding the first Monday are considered to be in week 0.
            week = int(date[4:6]) - 1
            # [D] is the weekday number, from 1 through 7, beginning with
            # Monday and ending with Sunday.
            # strpftime %w == Weekday as a decimal number [0(Sunday),6]
            day = int(date[6])
            if day == 7:
                day = 0
            date = "%04d%02d%1d" % (year, week, day)
        elif length_date == 7 and date.isdigit() and value.count('-') != 2:
            # we got a ordinal date: YYYYDDD
            date_pattern = "%Y%j"
        elif length_date == 8 and date.isdigit():
            # we got a calendar date: YYYYMMDD
            date_pattern = "%Y%m%d"
        else:
            raise ValueError("Wrong or incomplete ISO8601:2004 date format")
        # check for time zone information
        # note that the zone designator is the actual offset from UTC and
        # does not include any information on daylight saving time
        if time.count('+') == 1 and '+' in time[-6:]:
            (time, tz) = time.rsplit('+')
            delta = -1
        elif time.count('-') == 1 and '-' in time[-6:]:
            (time, tz) = time.rsplit('-')
            delta = 1
        else:
            delta = 0
        if delta:
            while len(tz) < 3:
                tz += '0'
            delta = delta * (int(tz[0:2]) * 60 * 60 + int(tz[2:]) * 60)
        # split microseconds
        ms = 0
        if '.' in time:
            (time, ms) = time.split(".")
            ms = float('0.' + ms.strip())
        # guess time pattern
        length_time = len(time)
        if length_time == 6 and time.isdigit():
            time_pattern = "%H%M%S"
        elif length_time == 4 and time.isdigit():
            time_pattern = "%H%M"
        elif length_time == 2 and time.isdigit():
            time_pattern = "%H"
        elif length_time == 0:
            time_pattern = ""
        else:
            raise ValueError("Wrong or incomplete ISO8601:2004 time format")
        # parse patterns
        dt = datetime.datetime.strptime(date + 'T' + time,
                                        date_pattern + 'T' + time_pattern)
        # add microseconds and eventually correct time zone
        return UTCDateTime(dt) + (float(delta) + ms)

    def _getTimeStamp(self):
        """ Returns UTC timestamp in seconds """
        return self.timestamp

    def __float__(self):
        """ Returns UTC timestamp in seconds """
        return self.timestamp

    def _getDateTime(self):
        """ Returns a Python datetime object """
        # datetime.utcfromtimestamp will cut off but not round
        # avoid through adding timedelta - also avoids the year 2038 problem
        return TIMESTAMP0 + datetime.timedelta(seconds=self.timestamp)

    datetime = property(_getDateTime)

    def _getDate(self):
        """ Returns a Python date object """
        return self._getDateTime().date()

    date = property(_getDate)

    def _getYear(self):
        """ Returns year of the current UTCDateTime object """
        return self._getDateTime().year

    def _setYear(self, value):
        """ Sets year of current UTCDateTime object """
        self._set(year=value)

    year = property(_getYear, _setYear)

    def _getMonth(self):
        """ Returns month as an integer (January is 1, December is 12) """
        return self._getDateTime().month

    def _setMonth(self, value):
        """ Sets month of current UTCDateTime object """
        self._set(month=value)

    month = property(_getMonth, _setMonth)

    def _getDay(self):
        """ Returns day as an integer """
        return self._getDateTime().day

    def _setDay(self, value):
        """ Sets day of current UTCDateTime object """
        self._set(day=value)

    day = property(_getDay, _setDay)

    def _getWeekday(self):
        """ Return the day of the week as an integer (Monday is 0, Sunday is 6) """
        return self._getDateTime().weekday()

    weekday = property(_getWeekday)

    def _getTime(self):
        """ Returns a Python time object """
        return self._getDateTime().time()

    time = property(_getTime)

    def _getHour(self):
        """ Returns hour as an integer """
        return self._getDateTime().hour

    def _setHour(self, value):
        """ Sets hours of current UTCDateTime object """
        self._set(hour=value)

    hour = property(_getHour, _setHour)

    def _getMinute(self):
        """ Returns minute as an integer """
        return self._getDateTime().minute

    def _setMinute(self, value):
        """
        Sets minutes of current UTCDateTime object.

        :param value: Minutes
        :type value: int

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11, 10, 11, 12)
        >>> dt.minute = 20
        >>> dt
        UTCDateTime(2012, 2, 11, 10, 20, 12)
        """
        self._set(minute=value)

    minute = property(_getMinute, _setMinute)

    def _getSecond(self):
        """
        Returns seconds as an integer.

        :rtype: int
        :return: Returns seconds as an integer.

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11, 10, 11, 12)
        >>> dt.second
        12
        """
        return self._getDateTime().second

    def _setSecond(self, value):
        """
        Sets seconds of current UTCDateTime object.

        :param value: Seconds
        :type value: int

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11, 10, 11, 12)
        >>> dt.second = 20
        >>> dt
        UTCDateTime(2012, 2, 11, 10, 11, 20)
        """
        self.timestamp += value - self.second

    second = property(_getSecond, _setSecond)

    def _getMicrosecond(self):
        """
        Returns microseconds as an integer.

        :rtype: int
        :return: Returns microseconds as an integer.

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11, 10, 11, 12, 345234)
        >>> dt.microsecond
        345234
        """
        return self._getDateTime().microsecond

    def _setMicrosecond(self, value):
        """
        Sets microseconds of current UTCDateTime object.

        :param value: Microseconds
        :type value: int

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 2, 11, 10, 11, 12, 345234)
        >>> dt.microsecond = 999123
        >>> dt
        UTCDateTime(2012, 2, 11, 10, 11, 12, 999123)
        """
        self._set(microsecond=value)

    microsecond = property(_getMicrosecond, _setMicrosecond)

    def _getJulday(self):
        """
        Returns Julian day as an integer.

        :rtype: int
        :return: Julian day as an integer.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.julday
        275
        """
        return self.utctimetuple().tm_yday

    def _setJulday(self, value):
        """
        Sets Julian day of current UTCDateTime object.

        :param value: Julian day
        :type value: int

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 12, 5, 12, 30, 35, 45020)
        >>> dt.julday = 275
        >>> dt
        UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        """
        self._set(julday=value)

    julday = property(_getJulday, _setJulday)

    def timetuple(self):
        """
        Return a time.struct_time such as returned by time.localtime().

        :rtype: time.struct_time
        """
        return self._getDateTime().timetuple()

    def utctimetuple(self):
        """
        Return a time.struct_time of current UTCDateTime object.

        :rtype: time.struct_time
        """
        return self._getDateTime().utctimetuple()

    def __add__(self, value):
        """
        Adds seconds and microseconds to current UTCDateTime object.

        :type value: int, float
        :param value: Seconds to add
        :rtype: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :return: New UTCDateTime object.

        .. rubric:: Example

        >>> dt = UTCDateTime(1970, 1, 1, 0, 0)
        >>> dt + 2
        UTCDateTime(1970, 1, 1, 0, 0, 2)

        >>> UTCDateTime(1970, 1, 1, 0, 0) + 1.123456
        UTCDateTime(1970, 1, 1, 0, 0, 1, 123456)
        """
        if isinstance(value, datetime.timedelta):
            # see datetime.timedelta.total_seconds
            value = (value.microseconds + (value.seconds + value.days *
                     86400) * 1000000) / 1000000.0
        return UTCDateTime(self.timestamp + value)

    def __sub__(self, value):
        """
        Subtracts seconds and microseconds from current UTCDateTime object.

        :type value: int, float or :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param value: Seconds or UTCDateTime object to subtract. Subtracting an
            UTCDateTime objects results into a relative time span in seconds.
        :rtype: :class:`~obspy.core.utcdatetime.UTCDateTime` or float
        :return: New UTCDateTime object or relative time span in seconds.

        .. rubric:: Example

        >>> dt = UTCDateTime(1970, 1, 2, 0, 0)
        >>> dt - 2
        UTCDateTime(1970, 1, 1, 23, 59, 58)

        >>> UTCDateTime(1970, 1, 2, 0, 0) - 1.123456
        UTCDateTime(1970, 1, 1, 23, 59, 58, 876544)

        >>> UTCDateTime(1970, 1, 2, 0, 0) - UTCDateTime(1970, 1, 1, 0, 0)
        86400.0
        """
        if isinstance(value, UTCDateTime):
            return round(self.timestamp - value.timestamp, self.__precision)
        elif isinstance(value, datetime.timedelta):
            # see datetime.timedelta.total_seconds
            value = (value.microseconds + (value.seconds + value.days *
                     86400) * 1000000) / 1000000.0
        return UTCDateTime(self.timestamp - value)

    def __str__(self):
        """
        Returns ISO8601 string representation from current UTCDateTime object.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> str(dt)
        '2008-10-01T12:30:35.045020Z'
        """
        return "%s%sZ" % (self.strftime('%Y-%m-%dT%H:%M:%S'),
                          (self.__ms_pattern % (abs(self.timestamp % 1)))[1:])

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __unicode__(self):
        """
        Returns ISO8601 unicode representation from current UTCDateTime object.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.__unicode__()
        '2008-10-01T12:30:35.045020Z'
        """
        return str(self.__str__())

    def __eq__(self, other):
        """
        Rich comparison operator '=='.

        .. rubric: Example

        Comparing two UTCDateTime object will always compare timestamps rounded
        to a precision of 6 digits by default.

        >>> t1 = UTCDateTime(123.000000012)
        >>> t2 = UTCDateTime(123.000000099)
        >>> t1 == t2
        True

        But the actual timestamp differ

        >>> t1.timestamp == t2.timestamp
        False

        Resetting the precision changes the behavior of the operator

        >>> t1.precision = 11
        >>> t1 == t2
        False
        """
        try:
            return round(self.timestamp - float(other), self.__precision) == 0
        except (TypeError, ValueError):
            return False

    def __ne__(self, other):
        """
        Rich comparison operator '!='.

        .. rubric: Example

        Comparing two UTCDateTime object will always compare timestamps rounded
        to a precision of 6 digits by default.

        >>> t1 = UTCDateTime(123.000000012)
        >>> t2 = UTCDateTime(123.000000099)
        >>> t1 != t2
        False

        But the actual timestamp differ

        >>> t1.timestamp != t2.timestamp
        True

        Resetting the precision changes the behavior of the operator

        >>> t1.precision = 11
        >>> t1 != t2
        True
        """
        return not self.__eq__(other)

    def __lt__(self, other):
        """
        Rich comparison operator '<'.

        .. rubric: Example

        Comparing two UTCDateTime object will always compare timestamps rounded
        to a precision of 6 digits by default.

        >>> t1 = UTCDateTime(123.000000012)
        >>> t2 = UTCDateTime(123.000000099)
        >>> t1 < t2
        False

        But the actual timestamp differ

        >>> t1.timestamp < t2.timestamp
        True

        Resetting the precision changes the behavior of the operator

        >>> t1.precision = 11
        >>> t1 < t2
        True
        """
        try:
            return round(self.timestamp - float(other), self.__precision) < 0
        except (TypeError, ValueError):
            return False

    def __le__(self, other):
        """
        Rich comparison operator '<='.

        .. rubric: Example

        Comparing two UTCDateTime object will always compare timestamps rounded
        to a precision of 6 digits by default.

        >>> t1 = UTCDateTime(123.000000099)
        >>> t2 = UTCDateTime(123.000000012)
        >>> t1 <= t2
        True

        But the actual timestamp differ

        >>> t1.timestamp <= t2.timestamp
        False

        Resetting the precision changes the behavior of the operator

        >>> t1.precision = 11
        >>> t1 <= t2
        False
        """
        try:
            return round(self.timestamp - float(other), self.__precision) <= 0
        except (TypeError, ValueError):
            return False

    def __gt__(self, other):
        """
        Rich comparison operator '>'.

        .. rubric: Example

        Comparing two UTCDateTime object will always compare timestamps rounded
        to a precision of 6 digits by default.

        >>> t1 = UTCDateTime(123.000000099)
        >>> t2 = UTCDateTime(123.000000012)
        >>> t1 > t2
        False

        But the actual timestamp differ

        >>> t1.timestamp > t2.timestamp
        True

        Resetting the precision changes the behavior of the operator

        >>> t1.precision = 11
        >>> t1 > t2
        True
        """
        try:
            return round(self.timestamp - float(other), self.__precision) > 0
        except (TypeError, ValueError):
            return False

    def __ge__(self, other):
        """
        Rich comparison operator '>='.

        .. rubric: Example

        Comparing two UTCDateTime object will always compare timestamps rounded
        to a precision of 6 digits by default.

        >>> t1 = UTCDateTime(123.000000012)
        >>> t2 = UTCDateTime(123.000000099)
        >>> t1 >= t2
        True

        But the actual timestamp differ

        >>> t1.timestamp >= t2.timestamp
        False

        Resetting the precision changes the behavior of the operator

        >>> t1.precision = 11
        >>> t1 >= t2
        False
        """
        try:
            return round(self.timestamp - float(other), self.__precision) >= 0
        except (TypeError, ValueError):
            return False

    def __repr__(self):
        """
        Returns a representation of UTCDatetime object.
        """
        return 'UTCDateTime' + self._getDateTime().__repr__()[17:]

    def __abs__(self):
        """
        Returns absolute timestamp value of the current UTCDateTime object.
        """
        # needed for unittest.assertAlmostEqual tests on Linux
        return abs(self.timestamp)

    def __hash__(self):
        """
        An object is hashable if it has a hash value which never changes
        during its lifetime. As an UTCDateTime object may change over time,
        it's not hashable. Use the :meth:`~UTCDateTime.datetime()` method to
        generate a :class:`datetime.datetime` object for hashing. But be aware:
        once the UTCDateTime object changes, the hash is not valid anymore.
        """
        # explicitly flag it as unhashable
        return None

    def strftime(self, format):
        """
        Return a string representing the date and time, controlled by an
        explicit format string.

        :type format: str
        :param format: Format string.
        :return: Formatted string representing the date and time.

        Format codes referring to hours, minutes or seconds will see 0 values.
        See methods :meth:`~datetime.datetime.strftime()` and
        :meth:`~datetime.datetime.strptime()` for more information.
        """
        return self._getDateTime().strftime(format)

    def strptime(self, date_string, format):
        """
        Return a UTCDateTime corresponding to date_string, parsed according to
        given format.

        :type date_string: str
        :param date_string: Date and time string.
        :type format: str
        :param format: Format string.
        :return: :class:`~obspy.core.utcdatetime.UTCDateTime`

        See methods :meth:`~datetime.datetime.strftime()` and
        :meth:`~datetime.datetime.strptime()` for more information.
        """
        return UTCDateTime(datetime.datetime.strptime(date_string, format))

    def timetz(self):
        """
        Return time object with same hour, minute, second, microsecond, and
        tzinfo attributes. See also method :meth:`datetime.datetime.time()`.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.timetz()
        datetime.time(12, 30, 35, 45020)
        """
        return self._getDateTime().timetz()

    def utcoffset(self):
        """
        Returns None (to stay compatible with :class:`datetime.datetime`)

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.utcoffset()
        """
        return self._getDateTime().utcoffset()

    def dst(self):
        """
        Returns None (to stay compatible with :class:`datetime.datetime`)

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.dst()
        """
        return self._getDateTime().dst()

    def tzname(self):
        """
        Returns None (to stay compatible with :class:`datetime.datetime`)

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.tzname()
        """
        return self._getDateTime().tzname()

    def ctime(self):
        """
        Return a string representing the date and time.

        .. rubric:: Example

        >>> UTCDateTime(2002, 12, 4, 20, 30, 40).ctime()
        'Wed Dec  4 20:30:40 2002'
        """
        return self._getDateTime().ctime()

    def isoweekday(self):
        """
        Return the day of the week as an integer (Monday is 1, Sunday is 7).

        :rtype: int
        :return: Returns day of the week as an integer, where Monday is 1 and
            Sunday is 7.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.isoweekday()
        3
        """
        return self._getDateTime().isoweekday()

    def isocalendar(self):
        """
        Returns a tuple containing (ISO year, ISO week number, ISO weekday).

        :rtype: tuple of ints
        :return: Returns a tuple containing ISO year, ISO week number and ISO
            weekday.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.isocalendar()
        (2008, 40, 3)
        """
        return self._getDateTime().isocalendar()

    def isoformat(self, sep="T"):
        """
        Return a string representing the date and time in ISO 8601 format.

        :rtype: str
        :return: String representing the date and time in ISO 8601 format like
            YYYY-MM-DDTHH:MM:SS.mmmmmm or, if microsecond is 0,
            YYYY-MM-DDTHH:MM:SS.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> dt.isoformat()
        '2008-10-01T12:30:35.045020'

        >>> dt = UTCDateTime(2008, 10, 1)
        >>> dt.isoformat()
        '2008-10-01T00:00:00'
        """
        return self._getDateTime().isoformat(sep=str(sep))

    def formatFissures(self):
        """
        Returns string representation for the IRIS Fissures protocol.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> print(dt.formatFissures())
        2008275T123035.0450Z
        """
        return "%04d%03dT%02d%02d%02d.%04dZ" % \
            (self.year, self.julday, self.hour, self.minute, self.second,
             self.microsecond // 100)

    def formatArcLink(self):
        """
        Returns string representation for the ArcLink protocol.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> print(dt.formatArcLink())
        2008,10,1,12,30,35,45020
        """
        return "%d,%d,%d,%d,%d,%d,%d" % (self.year, self.month, self.day,
                                         self.hour, self.minute, self.second,
                                         self.microsecond)

    def formatSeedLink(self):
        """
        Returns string representation for the SeedLink protocol.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35.45020)
        >>> print(dt.formatSeedLink())
        2008,10,1,12,30,35
        """
        # round seconds down to integer
        seconds = int(float(self.second) + float(self.microsecond) / 1.0e6)
        return "%d,%d,%d,%d,%d,%g" % (self.year, self.month, self.day,
                                      self.hour, self.minute, seconds)

    def formatSEED(self, compact=False):
        """
        Returns string representation for a SEED volume.

        :type compact: bool, optional
        :param compact: Delivers a compact SEED date string if enabled. Default
            value is set to False.
        :rtype: string
        :return: Datetime string in the SEED format.

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 10, 1, 12, 30, 35, 45020)
        >>> print(dt.formatSEED())
        2008,275,12:30:35.0450

        >>> dt = UTCDateTime(2008, 10, 1, 0, 30, 0, 0)
        >>> print(dt.formatSEED(compact=True))
        2008,275,00:30
        """
        if not compact:
            if not self.time:
                return "%04d,%03d" % (self.year, self.julday)
            return "%04d,%03d,%02d:%02d:%02d.%04d" % (self.year, self.julday,
                                                      self.hour, self.minute,
                                                      self.second,
                                                      self.microsecond // 100)
        temp = "%04d,%03d" % (self.year, self.julday)
        if not self.time:
            return temp
        temp += ",%02d" % self.hour
        if self.microsecond:
            return temp + ":%02d:%02d.%04d" % (self.minute, self.second,
                                               self.microsecond // 100)
        elif self.second:
            return temp + ":%02d:%02d" % (self.minute, self.second)
        elif self.minute:
            return temp + ":%02d" % (self.minute)
        return temp

    def formatIRISWebService(self):
        """
        Returns string representation usable for the IRIS Web services.

        :return: string

        .. rubric:: Example

        >>> dt = UTCDateTime(2008, 5, 27, 12, 30, 35, 45020)
        >>> print(dt.formatIRISWebService())
        2008-05-27T12:30:35.045
        """
        return "%04d-%02d-%02dT%02d:%02d:%02d.%03d" % \
            (self.year, self.month, self.day, self.hour, self.minute,
             self.second, self.microsecond // 1000)

    def _getPrecision(self):
        """
        Returns precision of current UTCDateTime object.

        :return: int

        .. rubric:: Example

        >>> dt = UTCDateTime()
        >>> dt.precision
        6
        """
        return self.__precision

    def _setPrecision(self, value=6):
        """
        Set precision of current UTCDateTime object.

        :type value: int, optional
        :param value: Precision value used by the rich comparison operators.
            Defaults to ``6``.

        .. rubric:: Example

        (1) Default precision

            >>> dt = UTCDateTime()
            >>> dt.precision
            6

        (2) Set precision during initialization of UTCDateTime object.

            >>> dt = UTCDateTime(precision=5)
            >>> dt.precision
            5

        (3) Set precision for an existing UTCDateTime object.

            >>> dt = UTCDateTime()
            >>> dt.precision = 12
            >>> dt.precision
            12
        """
        self.__precision = int(value)
        self.__ms_pattern = "%%0.%df" % (self.__precision)

    precision = property(_getPrecision, _setPrecision)

    def toordinal(self):
        """
        Return proleptic Gregorian ordinal. January 1 of year 1 is day 1.

        See :meth:`datetime.datetime.toordinal()`.

        :return: int

        .. rubric:: Example

        >>> dt = UTCDateTime(2012, 1, 1)
        >>> dt.toordinal()
        734503
        """
        return self._getDateTime().toordinal()

    @staticmethod
    def now():
        """
        Returns current UTC datetime.
        """
        return UTCDateTime()

    @staticmethod
    def utcnow():
        """
        Returns current UTC datetime.
        """
        return UTCDateTime()

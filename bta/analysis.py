import numpy as np
from typing import List, Union, Tuple
from enum import IntEnum
from bta._types import TDTEpoc, TDTSnip, TDTStream, TDTScalar
import copy

class EventCodes(IntEnum):
    Mag_CSe = 47103
    Mag_CSm = 48127
    Mag_CSp_Pump = 48607
    Mag_CSp = 48639
    Mag = 49151
    CSe = 63487
    CSm = 64511
    Pump_CSp = 64991
    CSp = 65023
    Pump = 65503
    Offset = 65535


def get_time_ranges(epoc: TDTEpoc, values: List[int]) -> np.ndarray:
    """
    Given a list of values, find the time ranges (in seconds) of the valid values in the epocs data.

    Parameters
    ----------
    epocs : TDTEpoc
        The epocs struct from the TDT data. example: data.epocs.PtAB
    values : list
        A list of values to find the time ranges of. example: [65023, 65024]

    Returns
    -------
    np.2darray
        A 2d array with 2 rows and n columns where n is the number of valid time ranges. The first row is the onset and the second row is the offset of the time ranges.

    Raises
    ------
    Exception
        If there is an error in the values filter.
    """
    
    if values:
        # find valid time ranges
        try:
            # use this for numbers
            valid = np.isclose(epoc.data, values)
        except:
            try:
                # use this for Note strings
                valid = np.isin(epoc.notes, values)
            except:
                raise Exception('error in valid filter')
        
        time_ranges = np.vstack((epoc.onset[valid], epoc.offset[valid]))
    return time_ranges

def modify_time_ranges(epoc: TDTEpoc, range_filtter: Union[Tuple[Union[int, float]], Tuple[Union[int, float], Union[int, float]]], time_ranges: Union[np.ndarray, None]=None) -> np.ndarray:
    """
    Filter the time ranges with a range filter. For example, a range filter of [0.0, 0.5] modify the time ranges to be the onset + 0.0 to onset + 0.5. If the offset is not provided, the epoc offset is used. 

    Parameters
    ----------
    epocs : TDTEpoc
        The epocs struct from the TDT data. example: data.epocs.PtAB
    range_filtter : Union[Tuple[Union[int, float]], Tuple[Union[int, float], Union[int, float]]]
        A tuple of 1 or 2 elements representing the seconds before and after the onset of the time range to filter. If a second element is not provided, the epoc offset is used as the offset of the time range.
    time_ranges : np.ndarray, optional
        A 2d array with 2 rows and n columns where n is the number of valid time ranges. The first row is the onset and the second row is the offset for a given time range (column). If not provided, the epoc onset and offset are used.

    Returns
    -------
    TDTEpoc
        A new epoc struct with the filtered time ranges.

    Examples
    -------

    filter the time ranges to be 1.5 seconds before and 0.5 seconds after the onset of epocs the time range
    >>> time_ranges = get_time_ranges(data.epocs.PtAB, [65023])
    >>> print(time_ranges[0][0], time_ranges[1][0])
    60 70
    >>> filtered_trs = modify_time_ranges(data.epocs.PtAB, time_ranges, [-0.1, 0.5])
    >>> print(filtered_trs[0][0], filtered_trs[1][0])
    59.5 60.5
    # filter the time ranges to be 5.5 seconds before the onset of epocs the time range
    >>> filtered_trs = modify_time_ranges(data.epocs.PtAB, time_ranges, [-0.1])
    >>> print(filtered_trs[0][0], filtered_trs[1][0])
    55.5 70
    """

    t1 = range_filtter[0]
    t2 = np.nan if len(range_filtter) < 2 else range_filtter[1] # type: ignore
    _time_ranges = copy.deepcopy(time_ranges)
    if not isinstance(_time_ranges, np.ndarray):
        _time_ranges = np.zeros((2, len(epoc.onset)))
        for j in range(len(epoc.onset)):
            _time_ranges[:, j] = [epoc.onset[j], epoc.offset[j]]
    # iterate over all the time range columns
    # NOTE: time_ranges[:, j] means all the rows of the jth column, which is the jth time range where j is the column index, row 0 is the onset and row 1 is the offset
    for j in range(_time_ranges.shape[1]):
        if np.isnan(t2): # if t2 is not provided then we are only modifying t1
            _time_ranges[:, j] = [_time_ranges[0,j]+t1, _time_ranges[1,j]]
        else:
            # convert from [onset, offset] to [onset+t1, onset+t1+t2]
            _time_ranges[:, j] = [_time_ranges[0,j]+t1, _time_ranges[0,j]+t1+t2]
    return _time_ranges

def get_filtered_stream_data(stream: TDTStream, time_ranges: np.ndarray) -> List[np.ndarray]:
    """
    Given a stream struct and time ranges, filter the stream data to only include the data in the time ranges.

    Parameters
    ----------
    stream : tdt.StructType
        The stream struct from the TDT data. example: data.streams._405S
    time_ranges : np.ndarray
        A 2d array with 2 rows and n columns where n is the number of valid time ranges. The first row is the onset and the second row is the offset of the time ranges.

    Returns
    -------
    List[np.ndarray]
        The stream data filtered to only include the data in the time ranges.
    """
    # sampling frequency
    fs = stream.fs
    # samples per frame
    sf = 1/(2.56e-6*fs)
    # the sample where the data starts, as start_time is in seconds
    td_sample = np.uint64(stream.header.start_time[0]/2.56e-6)
    filtered = []
    stream_data = np.array(stream.data)
    # the number of samples in the data
    max_ind = max(stream_data.shape)
    _time_ranges = copy.deepcopy(time_ranges)
    
    for j in range(_time_ranges.shape[1]):
        # _time_ranges[0,j] is the onset time in seconds, so we convert it to samples by dividing by the time step 2.56e-6 seconds
        tlo_sample = np.uint64(_time_ranges[0,j]/2.56e-6) # the onset time sample
        thi_sample = np.uint64(_time_ranges[1,j]/2.56e-6) # the offset time sample
        onset = np.uint64(np.maximum(np.round((tlo_sample - td_sample)/sf),0)) # the onset sample

        # if the offset is infinity (which can happen for some reason) then we set it to the maximum index
        if np.isinf(_time_ranges[1,j]):
            if onset <= max_ind and onset > -1:
                # if the data is 1d (i.e. a single channel)
                if stream_data.ndim == 1:
                    filtered.append(stream_data[onset:])
                else: 
                    # if the data is 2d (i.e. multiple channels)
                    filtered.append(stream_data[:,onset:])
                    break
        # Normal case
        else:
            # Get the stream data from the onset to the offset and append it to the filtered list
            offset = np.uint64(np.maximum(np.round((thi_sample-td_sample)/sf),0))
            
            if offset <= max_ind and offset > -1 and onset <= max_ind and onset > -1:
                if stream_data.ndim == 1:
                    filtered.append(stream_data[onset:offset])
                else:
                    filtered.append(stream_data[:,onset:offset])
    return filtered


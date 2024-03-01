import copy
import os
import re
import warnings
from datetime import datetime
from typing import List, Literal, Union

from bta._types import TDTData, AllowedEvtypes, AllowedFormats, TankEventType, TDTDataHeader, SampleInfo, StoreType, TDTData, StoreType, DataFormatEnum, EventMarker, TDTEpoc, TDTNote, TDTStream, TDTSnip, TDTScalar
import numpy as np
import tdt
import logging

logger = logging.getLogger(__name__)

MAX_UINT64 = np.iinfo(np.uint64).max

def time2sample(ts: float, fs: float = 195312.5, t1: bool = False, t2: bool = False, to_time: bool = False) -> np.uint64:
    """
    Convert time in seconds to sample index.

    Parameters
    ----------
    ts : float
        Time in seconds.
    fs : float, optional 
        Sampling frequency in Hz. Defaults to 195312.5.
    t1 : bool, optional 
        Round sample index up to the nearest integer. Defaults to False.
    t2 : bool, optional 
        Round sample index down to the nearest integer. Defaults to False.
    to_time : bool, optional 
        Convert sample index to time in seconds. Defaults to False.

    Returns
    -------
    np.uint64 
        Sample index or time in seconds.
    """
    sample = ts * fs
    if t2:
        # drop precision beyond 1e-9
        exact = np.round(sample * 1e9) / 1e9
        sample = np.floor(sample)
        if exact == sample:
            sample -= 1
    else:
        # drop precision beyond 1e-9
        sample = np.round(sample * 1e9) / 1e9
        if t1:
            sample = np.ceil(sample)
        else:
            sample = np.round(sample)
    sample = np.uint64(sample)
    if to_time:
        return np.float64(sample) / fs
    return sample


def parse_tbk(tbk_path) -> dict[str, StoreType]:
    block_notes: dict[str, StoreType] = {}
    try:
        with open(tbk_path, "rb") as tbk:
            s = tbk.read().decode("cp437") # random encoding?

        # find the block notes. First, find the first and second delimiters
        delimInd = [m.start() for m in re.finditer("\[USERNOTEDELIMITER\]", s)]
        # remove the delimiter
        s = s[delimInd[1] : delimInd[2]].replace("[USERNOTEDELIMITER]", "")

        lines = s.split("\n")
        lines = lines[:-1]
        # loop through rows
        temp_store = StoreType()
        first_pass = True
        for line in lines:
            # check if this is a new store by looking for the StoreName in the line
            # if it is, insert the previous store into the block notes and start a new store
            if "StoreName" in line:
                if not first_pass:
                    block_notes[temp_store.StoreName] = copy.deepcopy(temp_store)
                else:
                    first_pass = False
                temp_store = StoreType()
            # find delimiters
            parts = line.split(";")[:-1]
            # grab field and value between the '=' and ';'
            field_str = parts[0].split("=")[-1]
            value = parts[-1].split("=")[-1]
            # convert value to appropriate type
            if field_str == "CircType":
                temp_store.CircType = int(value)
            elif field_str == "DataFormat":
                temp_store.DataFormat = DataFormatEnum(int(value))
            elif field_str == "Enabled":
                temp_store.Enabled = bool(int(value))
            elif field_str == "HeadName":
                temp_store.HeadName = value
            elif field_str == "NumChan":
                temp_store.NumChan = int(value)
            elif field_str == "NumPoints":
                temp_store.NumPoints = int(value)
            elif field_str == "SampleFreq":
                temp_store.SampleFreq = float(value)
            elif field_str == "SecTag":
                temp_store.SecTag = value
            elif field_str == "StoreName":
                temp_store.StoreName = value
            elif field_str == "StrobeBuddy":
                temp_store.StrobeBuddy = value
            elif field_str == "StrobeMode":
                temp_store.StrobeMode = int(value)
            elif field_str == "TankEvType":
                temp_store.TankEvType = int(value)
    except Exception as e:
        warnings.warn(f"Unable to parse tbk file: {tbk_path}\n\n\tPossibly a bad tbk file, try running the TankRestore tool to correct. See https://www.tdt.com/docs/technotes/tn/TN0935/", Warning, stacklevel=2)
        raise e
    return block_notes


def code_to_type(code: int) -> Literal["epocs", "snips", "streams", "scalars", "unknown"]:
    """
    Convert event code to event type.

    Parameters
    ----------
    code : int
        Event code.

    Returns
    -------
    Literal["epocs", "snips", "streams", "scalars", "unknown"]
        Event type.
    """
    
    strobe_types = [TankEventType.STRON.value, TankEventType.STROFF.value, TankEventType.MARK.value]
    scalar_types = [TankEventType.SCALAR.value]
    snip_types = [TankEventType.SNIP.value]

    if code in strobe_types:
        s = "epocs"
    elif code in snip_types:
        s = "snips"
    elif code & TankEventType.MASK.value == TankEventType.STREAM.value:
        s = "streams"
    elif code in scalar_types:
        s = "scalars"
    else:
        s = "unknown"
    return s


def check_ucf(code: int) -> bool:
    """
    Check if event code has unique channel files.

    Parameters
    ----------
    code : int
        Event code.

    Returns
    -------
    bool
        True if event code has unique channel files, False otherwise.
    """
    return code & TankEventType.UCF.value == TankEventType.UCF.value


def code_to_name(code: int):
    return int(code).to_bytes(4, byteorder="little").decode("cp437")


def epoc_to_type(code: int) -> Literal["onset", "offset", "unknown"]:
    # given epoc event code, return if it is 'onset' or 'offset' event

    strobe_on_types = [TankEventType.STRON.value, TankEventType.MARK.value]
    strobe_off_types = [TankEventType.STROFF.value]
    if code in strobe_on_types:
        return "onset"
    elif code in strobe_off_types:
        return "offset"
    else: 
        return "unknown"


def get_files(dir: str, ext: str, ignore_mac: bool = False) -> List[str]:
    """
    Get all files in a directory with a specified extension.

    Parameters
    ----------
    dir : str
        Directory path.
    ext : str
        File extension.
    ignore_mac : bool, optional
        Ignore Mac OS hidden files. Defaults to False.

    Returns
    -------
    List[str]
        List of file paths.
    """
    result = []
    for file in os.listdir(dir):
        if file.endswith(ext):
            if ignore_mac and file.startswith('._'):
                continue
            result.append(os.path.join(dir, file))
    return result

def header_to_text(header, scale):
    hhh = []
    hhh.append("Path:\t" + header.tev_path)
    hhh.append("Start:\t" + str(header.start_time[0]))
    hhh.append("Stop:\t" + str(header.stop_time[0]))
    hhh.append("ScaleFactor:\t{0}".format(scale))
    hhh.append("Stores:")
    for k in header.stores.keys():
        type_str = header.stores[k].type_str
        hhh.append("\tName:\t" + header.stores[k].name)
        hhh.append("\tType:\t" + header.stores[k].type_str)
        if type_str in ["streams", "snips"]:
            hhh.append("\tFreq:\t" + str(header.stores[k].fs))
            hhh.append("\tNChan:\t" + str(int(max(header.stores[k].chan))))
        if type_str in ["snips"]:
            hhh.append("\tSort:\t" + header.stores[k].sortname)
        if type_str in ["scalars"]:
            hhh.append("\t\tNChan:\t" + str(int(max(header.stores[k].chan))))
        hhh.append("")
    return "\n".join(hhh)

# def get_store_notes(user_notes, store_name, heads, header_start_time, valid_ind) -> Union[TDTNote, None]:
#     """This function returns a TDTNote object if the store has notes, otherwise it returns None.
#     NEEDS TO BE IMPROVED
#     NOTE: Only epocs and scalar stores can have notes."""
#     note = None
#     note_index = user_notes != 0
#     if np.any(note_index):
#         note = TDTNote()
#         note.name = store_name
#         ts_ind = valid_ind[note_index]
#         note_ts = (np.reshape(heads[[[4], [5]], ts_ind].T, (-1, 1)).T.view(np.float64) - header_start_time)
#         note_ts = time2sample(note_ts, to_time=True)
#         note_index = user_notes[note_index]
#         try:
#             note.ts.extend(note_ts)
#             note.index.extend(note_index)
#         except:
#             pass
#     return note

def _time_filter(event: Union[TDTEpoc, TDTSnip, TDTStream, TDTScalar, TDTNote], valid_time_range: np.ndarray, num_ranges: int):
     # do a time filter

    firstStart = valid_time_range[0, 0]
    last_stop = valid_time_range[1, -1]
    filtered_data = []
    filtered_onset = []
    filtered_offset = []
    filtered_ts = []
    filtered_chan = []
    filtered_sort_code = []
    start_time = []
    # all events have a ts attribute
    if event.header.type_str == "streams":
        event.header.start_time = [
            0 for jj in range(num_ranges)
        ]
    else:
        filtered_ts = [np.array([], dtype=event.ts.dtype) for jj in range(num_ranges)]
    if event.header.type_str in ["snips", "streams", "scalars"]:
        filtered_chan = [[] for jj in range(num_ranges)]
    if event.header.type_str == "snips":
        filtered_sort_code = [np.array([], dtype=event.sortcode.dtype) for jj in range(num_ranges)]
    if True:
        this_dtype = np.array(event.data).dtype
        filtered_data = [np.array([], dtype=this_dtype) for jj in range(num_ranges)]
        filter_ind = [[] for i in range(num_ranges)]
        for jj in range(num_ranges):
            start = valid_time_range[0, jj]
            stop = valid_time_range[1, jj]
            ind1 = event.ts >= start
            ind2 = event.ts < stop
            filter_ind[jj] = np.where(ind1 & ind2)[0]
            bSkip = 0
            if len(filter_ind[jj]) == 0:
                # if it's a stream and a short window, we might have missed it
                if event.header.type_str == "streams":
                    ind2 = np.where(ind2)[0]
                    if len(ind2) > 0:
                        ind2 = ind2[-1]
                        # keep one prior for streams (for all channels)
                        nchan = max(event.chan)
                        if ind2 - nchan >= -1:
                            filter_ind[jj] = ind2 - np.arange(nchan - 1, -1, -1)
                            temp = event.ts[filter_ind[jj]]
                            start_time[jj] = temp[0]
                            bSkip = 1
            if len(filter_ind[jj]) > 0:
                # parse out the information we need
                if event.header.type_str == "streams":
                    # keep one prior for streams (for all channels)
                    if not bSkip:
                        nchan = max(event.chan)
                        temp = filter_ind[jj]
                        if temp[0] - nchan > -1:
                            filter_ind[jj] = np.concatenate([-np.arange(nchan, 0, -1) + temp[0], filter_ind[jj]])
                            temp = event.ts[filter_ind[jj]]
                            start_time[jj] = temp[0]
                else:
                    filtered_ts[jj] = event.ts[filter_ind[jj]]
                if event.header.type_str in ["snips", "streams", "scalars"]:
                    if len(event.chan) > 1:
                        filtered_chan[jj] = event.chan[filter_ind[jj]]
                    else:
                        filtered_chan[jj] = event.chan
                if event.header.type_str == "snips":
                    filtered_sort_code[jj] = event.sortcode[filter_ind[jj]]
                if True:
                    filtered_data[jj] = event.data[filter_ind[jj]]

        if event.header.type_str == "streams":
            if filtered_chan == []:
                filtered_chan = [[] for i in range(num_ranges)]
            if filtered_chan == []:
                filtered_data = [[] for i in range(num_ranges)]
            filtered_data = filtered_data
        else:
            if filtered_ts != []:
                # event.ts = np.concatenate(filtered_ts)
                pass
            if event.header.type_str in ["snips", "streams", "scalars"]:
                if filtered_chan != []:
                    # event.chan = np.concatenate(filtered_chan)
                    pass
                if event.header.type_str == "snips":
                    # if len(set(event.chan)) == 1:
                    #     event.chan = [event.chan[0]]
                    # event.sortcode = np.concatenate(filtered_sort_code)
                    pass
            if True:
                # event.data = np.concatenate(filtered_data)
                pass
    if event.header.type_str == "epocs":
        # handle epoc events
        filter_ind = []
        for jj in range(num_ranges):
            start = valid_time_range[0, jj]
            stop = valid_time_range[1, jj]
            ind1 = event.onset >= start
            ind2 = event.onset < stop
            filter_ind.append(np.where(ind1 & ind2)[0])

        filter_ind = np.concatenate(filter_ind)
        if len(filter_ind) > 0:
            filtered_onset = event.onset[filter_ind]
            filtered_data = event.data[filter_ind]
            filtered_offset = event.offset[filter_ind]
            if (
                filtered_offset[0]
                < filtered_onset[0]
            ):
                if filtered_onset[0] > firstStart:
                    filtered_onset = np.concatenate([[firstStart], filtered_onset]
                    )
            if filtered_offset[-1] > last_stop:
                filtered_offset[-1] = last_stop

    return start_time, np.array(filtered_data), np.array(filtered_ts), np.array(filtered_chan), np.array(filtered_sort_code), np.array(filtered_onset), np.array(filtered_offset)

def read_block(
    block_path,
    *,
    bitwise="",
    channel=0,
    combine=None,
    headers_only: bool = False,
    custom_header: Union[TDTDataHeader, None] = None,
    nodata=False,
    ranges=None,
    speecify_store_names: List[str]=[],
    t1: int =0,
    t2: int =0,
    evtype: List[AllowedEvtypes] = [AllowedEvtypes.ALL],
    verbose=0,
    sortname="TankSort",
    export=None,
    scale=1,
):
    """TDT tank data extraction.

    data = read_block(block_path), where block_path is a string, retrieves
    all data from specified block directory in struct format. This reads
    the binary tank data and requires no Windows-based software.

    data.epocs      contains all epoc store data (onsets, offsets, values)
    data.snips      contains all snippet store data (timestamps, channels,
                    and raw data)
    data.streams    contains all continuous data (sampling rate and raw
                    data)
    data.scalars    contains all scalar data (samples and timestamps)
    data.info       contains additional information about the block

    optional keyword arguments:
        t1          scalar, retrieve data starting at t1 (default = 0 for
                        beginning of recording)
        t2          scalar, retrieve data ending at t2 (default = 0 for end
                        of recording)
        sortname    string, specify sort ID to use when extracting snippets
                        (default = 'TankSort')
        evtype      array of strings, specifies what type of data stores to
                        retrieve from the tank. Can contain 'all' (default),
                        'epocs', 'snips', 'streams', or 'scalars'.
                      example:
                          data = read_block(block_path, evtype=['epocs','snips'])
                              > returns only epocs and snips
        ranges      array of valid time range column vectors.
                      example:
                          tr = np.array([[1,3],[2,4]])
                          data = read_block(block_path, ranges=tr)
                              > returns only data on t=[1,2) and [3,4)
        nodata      boolean, only return timestamps, channels, and sort
                        codes for snippets, no waveform data (default = false).
                        Useful speed-up if not looking for waveforms
        speecify_store_names : List[str]
            specify store names to extract.
        channel  :
            integer or list, choose a single channel or array of channels to extract from stream or snippet events. Default is 0, to extract all channels.
        bitwise     string, specify an epoc store or scalar store that
                        contains individual bits packed into a 32-bit
                        integer. Onsets/offsets from individual bits will
                        be extracted.
        headers_only: bool, optional
            If True, only the headers will be returned. Defaults to False.
                        time, for faster consecutive reads.
        header: Header, optional
            If specified, the header will be used instead of reading from the block_path.

        combine     list, specify one or more data stores that were saved
                        by the Strobed Data Storage gizmo in Synapse (or an
                        Async_Stream_store macro in OpenEx). By default,
                        the data is stored in small chunks while the strobe
                        is high. This setting allows you to combine these
                        small chunks back into the full waveforms that were
                        recorded while the strobe was enabled.
                      example:
                        data = read_block(block_path, combine=['StS1'])
        export      string, choose a data exporting format.
                        csv:        data export to comma-separated value files
                                    streams: one file per store, one channel per column
                                    epocs: one column onsets, one column offsets
                        binary:     streaming data is exported as raw binary files
                                    one file per channel per store
                        interlaced: streaming data exported as raw binary files
                                    one file per store, data is interlaced
        scale       float, scale factor for exported streaming data. Default = 1.
        signal      a pyqt signal object, if specified, progress updates will be sent
    """

    if "all" in evtype:
        evtype = ["epocs", "snips", "streams", "scalars"]

    evtype = list(set(evtype))
    data = TDTData()
    use_outside_headers = False
    if custom_header is not None:
        use_outside_headers = True
        data.header = custom_header

    block_path = os.path.join(block_path, "")

    if not use_outside_headers:
        tsq_list = get_files(block_path, ".tsq", ignore_mac=True)

        if len(tsq_list) < 1:
            if not os.path.isdir(block_path):
                raise Exception("block path {0} not found".format(block_path))

            if "streams" in evtype:
                warnings.warn(
                    "no tsq file found",
                    Warning,
                    stacklevel=2,
                )
                return None
            else:
                raise Exception("no TSQ file found in {0}".format(block_path))

        elif len(tsq_list) > 1:
            raise Exception("multiple TSQ files found\n{0}".format(",".join(tsq_list)))

        try:
            tsq = open(tsq_list[0], "rb")
        except:
            raise Exception("tsq file {0} could not be opened".format(tsq_list[0]))

        data.header.tev_path = tsq_list[0].replace(".tsq", ".tev")

    if not headers_only:
        try:
            tev = open(data.header.tev_path, "rb")
        except:
            raise Exception("tev file {0} could not be opened".format(data.header.tev_path))

    # look for epoch tagged notes
    tnt_path = data.header.tev_path.replace(".tev", ".tnt")
    note_str = np.array([])
    try:
        lines = np.array([line.rstrip("\n") for line in open(tnt_path)])
        # file version is in first line
        # note_file_version = lines[0]
        note_str = lines[1:]
    except:
        warnings.warn("tnt file could not be processed", Warning, stacklevel=2)

    # look for custom sort_ids
    custom_sort_event = []
    custom_sort_channel_map = []
    custom_sort_codes = []

    if "snips" in evtype and sortname != "TankSort":
        # we want a custom one, parse all custom sorts
        sort_ids = {"fileNames": [], "event": [], "sort_id": []}
        sort_path = os.path.join(block_path, "sort")

        try:
            for sort_id in os.listdir(sort_path):
                if os.path.isdir(os.path.join(sort_path, sort_id)):
                    # parse sort result file name
                    sort_files = get_files(
                        os.path.join(sort_path, sort_id), ".SortResult", ignore_mac=True
                    )
                    for sort_file in sort_files:
                        head, tail = os.path.split(sort_file)
                        sort_ids["event"].append(tail.split(".")[0])
                        sort_ids["fileNames"].append(sort_file)
                        sort_ids["sort_id"].append(os.path.split(head)[-1])

            # now look for the exact sortname specified by user
            if sortname in sort_ids["sort_id"]:
                for i in range(len(sort_ids["sort_id"])):
                    if sort_ids["sort_id"][i] == sortname:
                        custom_sort_event.append(sort_ids["event"][i])
                        ddd = np.fromfile(sort_ids["fileNames"][i], dtype=np.uint8)
                        custom_sort_channel_map.append(ddd[:1024])
                        custom_sort_codes.append(ddd[1024:])
            else:
                warnings.warn(
                    "sort_id:{0} not found\n".format(sortname), Warning, stacklevel=2
                )
        except:
            pass

    """
    tbk file has block events information and on second time offsets
    to efficiently locate events if the data is queried by time.

    tsq file is a list of event headers, each 40 bytes long, ordered strictly
    by time.

    tev file contains event binary data

    tev and tsq files work together to get an event's data and attributes

    tdx file contains just information about epoc stores,
    is optionally generated after recording for fast retrieval
    of epoc information
    """

    # read TBK notes to get event info
    tbk_path = data.header.tev_path.replace(".tev", ".Tbk")
    block_notes = parse_tbk(tbk_path)

    if not use_outside_headers:
        # read start time
        tsq.seek(0, os.SEEK_SET)
        xxx = tsq.read(8)
        tsq.seek(48, os.SEEK_SET)
        code1 = np.fromfile(tsq, dtype=np.int32, count=1)
        assert code1 == EventMarker.STARTBLOCK, "Block start marker not found"
        tsq.seek(56, os.SEEK_SET)
        start_time = np.fromfile(tsq, dtype=np.float64, count=1)
        if start_time:
            data.header.start_time = float(start_time[0])
        # read stop time
        tsq.seek(-32, os.SEEK_END)
        code2 = np.fromfile(tsq, dtype=np.int32, count=1)
        if code2 != EventMarker.STOPBLOCK:
            warnings.warn(
                "Block end marker not found, block did not end cleanly. Try setting T2 smaller if errors occur",
                Warning,
                stacklevel=2,
            )
            data.header.stop_time = None
        else:
            tsq.seek(-24, os.SEEK_END)
            stop_time = np.fromfile(tsq, dtype=np.float64, count=1)
            if stop_time:
                data.header.stop_time = float(stop_time[0])

    # set info fields
    [data.info.tankpath, data.info.blockname] = os.path.split(
        os.path.normpath(block_path)
    )
    data.info.start_date = datetime.fromtimestamp(data.header.start_time)
    if data.header.start_time:
        data.info.utc_start_time = data.info.start_date.strftime("%H:%M:%S")
    else:
        data.info.utc_start_time = None

    if data.header.stop_time:
        data.info.stop_date = datetime.fromtimestamp(data.header.stop_time)
        data.info.utc_stop_time = data.info.stop_date.strftime("%H:%M:%S")
    else:
        data.info.stop_date = None
        data.info.utc_stop_time = None

    if data.header.stop_time > 0:
        data.info.duration = (
            data.info.stop_date - data.info.start_date
        )  # datestr(s2-s1,'HH:MM:SS')
    data.info.stream_channel = channel
    data.info.snip_channel = channel

    # look for Synapse recording notes
    notes_txt_path = os.path.join(block_path, "Notes.txt")
    notes_txt_lines = []
    try:
        with open(notes_txt_path, "rt") as txt:
            notes_txt_lines = txt.readlines()
    except:
        # warnings.warn('Synapse Notes file could not be processed', Warning, stacklevel=2)
        pass

    note_text = []
    note_ts = []
    do_once = 1
    this_note_text = ""
    if len(notes_txt_lines) > 1:
        targets = ["Experiment", "Subject", "User", "Start", "Stop"]
        in_note = False
        for note_line in notes_txt_lines:
            note_line = note_line.strip()
            if len(note_line) == 0:
                continue

            target_found = False
            for target in targets:
                test_str = target + ":"
                eee = len(test_str)
                if len(note_line) >= eee + 2:
                    if note_line.startswith(test_str):
                        setattr(data.info, target.lower(), note_line[eee + 1 :])
                        target_found = True
                        break
            if target_found:
                continue

            if do_once:
                if "-" in data.info.start:
                    yearfmt = "%Y-%m-%d"
                else:
                    yearfmt = "%m/%d/%Y"
                if "m" in data.info.start.lower():
                    timefmt = "%I:%M:%S%p"
                else:
                    timefmt = "%H:%M:%S"
                rec_start = datetime.strptime(
                    data.info.start.lower(), timefmt + " " + yearfmt
                )
                curr_day = rec_start.day
                curr_month = rec_start.month
                curr_year = rec_start.year
                do_once = 0

            # look for actual notes
            test_str = "Note-"
            eee = len(test_str)
            note_start = False
            no_buttons = False
            if len(note_line) >= eee + 2:
                if note_line.startswith(test_str):
                    in_note = False
                    note_start = True
            if in_note:
                if '"' in note_line:
                    this_note_text += note_line.split('"')[0]
                    note_text.append(this_note_text)
                    in_note = False
                else:
                    this_note_text += "\n" + note_line
            if note_start:
                # start of new note
                this_note_text = ""
                try:
                    note_id = re.findall("(?<=\[)(.*?)(?=\s*\])", note_line)
                    if len(note_id):
                        note_id = note_id[0]
                    else:
                        no_buttons = True
                except:
                    note_id = re.findall(test_str + "(.*?)(?=\s*:)", note_line)[0]

                note_parts = note_line.split(" ")
                note_time = note_parts[1]
                note_dt = datetime.strptime(note_time.lower(), timefmt)
                note_dt = note_dt.replace(year=curr_year)
                note_dt = note_dt.replace(month=curr_month)
                note_dt = note_dt.replace(day=curr_day)

                note_time_relative = note_dt - rec_start
                date_changed = False
                if note_id == "none" or no_buttons:
                    quotes = note_line.split('"')
                    if len(quotes) <= 2:
                        in_note = True
                    this_note_text = quotes[1]

                    if "date changed to" in this_note_text:
                        date_changed = True
                        temp = datetime.strptime(this_note_text[16:].lower(), yearfmt)
                        curr_day = temp.day
                        curr_month = temp.month
                        curr_year = temp.year
                    if not in_note:
                        note_text.append(this_note_text)
                else:
                    note_text.append(note_id)
                if not date_changed:
                    note_ts.append(note_time_relative.seconds)

    note_text = np.array(note_text)

    """
    # TTank event header structure
    tsqEventHeader = struct(...
        'size', 0, ...
        'type', 0, ...  % (long) event type: snip, pdec, epoc etc
        'code', 0, ...  % (long) event name: must be 4 chars, cast as a long
        'channel', 0, ... % (unsigned short) data acquisition channel
        'sortcode', 0, ... % (unsigned short) sort code for snip data. See also OpenSorter .SortResult file.
        'timestamp', 0, ... % (double) time offset when even occurred
        'ev_offset', 0, ... % (int64) data offset in the TEV file OR (double) strobe data value
        'format', 0, ... % (long) data format of event: byte, short, float (typical), or double
        'frequency', 0 ... % (float) sampling frequency
    );
    """
    def get_event_data_by_name(name: str, event_typs: Literal["epocs", "snips", "streams", "scalars"]) -> Union[TDTEpoc, TDTSnip, TDTStream, TDTScalar, None]:
        if event_typs == "epocs":
            for epoc in data.epocs:
                if epoc.header.name == name:
                    return epoc
        elif event_typs == "snips":
            for snip in data.snips:
                if snip.header.name == name:
                    return snip
        elif event_typs == "streams":
            for stream in data.streams:
                if stream.header.name == name:
                    return stream
        elif event_typs == "scalars":
            for scalar in data.scalars:
                if scalar.header.name == name:
                    return scalar
        else:
            raise ValueError(f"Eevent type {event_typs} not recognized")
        return None
    
    if not use_outside_headers:
        tsq.seek(40, os.SEEK_SET)

        if t2 > 0:
            # make the reads shorter if we are stopping early
            read_size = 10000000
        else:
            read_size = 50000000

        code_ct = 0
        while True:
            # read all headers into one giant array
            heads = np.frombuffer(tsq.read(read_size * 4), dtype=np.uint32)
            if len(heads) == 0:
                continue

            rem = len(heads) % 10
            if rem != 0:
                warnings.warn(
                    "Block did not end cleanly, removing last {0} headers".format(rem),
                    Warning,
                    stacklevel=2,
                )
                heads = heads[:-rem]

            # reshape so each column is one header
            heads = heads.reshape((-1, 10)).T
            # check the codes first and build store maps and note arrays
            codes = heads[2, :]

            good_codes = codes > 0
            bad_codes = np.logical_not(good_codes)

            if np.sum(bad_codes) > 0:
                warnings.warn(
                    "Bad TSQ headers were written, removing {0}, keeping {1} headers".format(
                        sum(bad_codes), sum(good_codes)
                    ),
                    Warning,
                    stacklevel=2,
                )
                heads = heads[:, good_codes]
                codes = heads[2, :]

            # get set of codes but preserve order in the block
            store_codes = []
            unique_codes, unique_ind = np.unique(codes, return_index=True)

            for counter, x in enumerate(unique_codes):
                store_codes.append(
                    {
                        "code": x,
                        "type": heads[1, unique_ind[counter]],
                        "type_str": code_to_type(heads[1, unique_ind[counter]]),
                        "ucf": check_ucf(heads[1, unique_ind[counter]]),
                        "epoc_type": epoc_to_type(heads[1, unique_ind[counter]]),
                        "dform": heads[8, unique_ind[counter]],
                        "size": heads[0, unique_ind[counter]],
                        "buddy": heads[3, unique_ind[counter]],
                        "temp": heads[:, unique_ind[counter]],
                    }
                )

            # LOOP THROUGH ALL STORES AND ADD TO DATA OBJECT
            for store_code in store_codes:
                if store_code["code"] in [EventMarker.STARTBLOCK, EventMarker.STOPBLOCK]:
                    continue
                if store_code["code"] == 0:
                    warnings.warn(
                        "Skipping unknown header code 0", Warning, stacklevel=2
                    )
                    continue

                store_code["name"] = code_to_name(store_code["code"])
                valid_ind = np.where(codes == store_code["code"])[0]
                skip_disabled = False
                if len(block_notes) > 0:
                    for store in block_notes.values():
                        if store.StoreName == store_code["name"]:
                                if not store.Enabled:
                                    warnings.warn(
                                        "{0} store DISABLED".format(store.StoreName),
                                        Warning,
                                        stacklevel=2,
                                    )
                                    skip_disabled = True
                                    break
                if skip_disabled:
                    continue

                # if looking for a particular store and this isn't it, flag it
                # for now. need to keep looking for buddy epocs
                skip_by_name = False
                if speecify_store_names: # if not empty
                    if store_code["name"] not in speecify_store_names:
                        skip_by_name = True

                store_code["var_name"] = store_code["name"]
                var_name = store_code["var_name"]
                temp = heads[3, valid_ind].view(np.uint16)

                if store_code["type_str"] not in evtype:
                    logger.info(f"Skipping {store_code['name']} store of type {store_code['type_str']} as it is not in the specified evtype list: {evtype}")
                    continue
                # GET DATA FOR EACH STORE AND ADD TO DATA OBJECT
                if store_code["type_str"] == "epocs":
                    buddy = "".join(
                        [
                            str(chr(c))
                            for c in np.array([store_code["buddy"]]).view(np.uint8)
                        ]
                    )
                    buddy = buddy.replace("\x00", " ")
                    # if this store set to be skipped but is a buddy of a store that is set to be included, include it
                    if skip_by_name:
                        if speecify_store_names:
                            if buddy in speecify_store_names:
                                skip_by_name = False
                    if skip_by_name:
                        continue
                    if not store_code["name"] in [x.header.name for x in data.epocs]:
                        if skip_by_name:
                            continue
                        curr_epoc = TDTEpoc()
                        curr_epoc.header.name = store_code["name"]
                        curr_epoc.buddy = buddy
                        curr_epoc.code = store_code["code"]
                        curr_epoc.header.type = store_code["epoc_type"]
                        curr_epoc.header.type_str = store_code["type_str"]
                        dform_num = store_code["dform"]
                        curr_epoc.dform = DataFormatEnum(dform_num)
                        # if data.header.start_time is not None:
                        #         note = get_store_notes(
                        #             user_notes = heads[9, valid_ind].view(np.uint32),
                        #             store_name = curr_epoc.header.name,
                        #             heads = heads,
                        #             header_start_time = data.header.start_time,
                        #             valid_ind = valid_ind
                        #         )
                        # else:
                        #     warnings.warn(
                        #         f"Start time not found, skipping note extraction for {curr_epoc.header.name}",
                        #         Warning,
                        #         stacklevel=2,
                        #     )
                        curr_epoc.ts = np.append(curr_epoc.ts, time2sample((np.reshape(heads[[[4], [5]], valid_ind].T, (-1, 1)).T.view(np.float64)-data.header.start_time), to_time=True))
                        if curr_epoc.header.type == "onset":
                            curr_epoc.header.size = 10
                            # if ts is 2d or greater, flatten it
                            if len(curr_epoc.ts.shape) > 1:                                                       
                                curr_epoc.ts = np.concatenate(curr_epoc.ts, axis=1)[0]
                            curr_epoc.onset = curr_epoc.ts
                            curr_epoc.offset = np.append(curr_epoc.ts[1:], np.inf)

                        if curr_epoc.header.type == "offset":
                            var_name = curr_epoc.buddy
                            if var_name not in [x.header.name for x in data.epocs]:
                                warnings.warn(f"Offset buddy `{var_name}` not found", Warning, stacklevel=2)
                                continue

                            onset_buddy = get_event_data_by_name(var_name, "epocs") # this is the onset event corresponding to this offset event
                            if onset_buddy is None:
                                # handle odd case where there is a single offset event and no onset events
                                if "onset" in [x.header.type for x in data.epocs]:
                                    raise ValueError(f"There is an onset event in the data yet this offset could not find its buddy")
                                onset_buddy = TDTEpoc()
                                onset_buddy.header.name = curr_epoc.buddy
                                onset_buddy.onset = 0
                                onset_buddy.header.type_str = "epocs"
                                onset_buddy.header.type = "onset"
                                onset_buddy.data = 0
                                onset_buddy.dform = DataFormatEnum.DOUBLE
                                onset_buddy.header.size = 10
                                data.epocs.append(onset_buddy)

                            onset_buddy.offset = curr_epoc.ts

                            # fix time ranges
                            if onset_buddy.offset[0] < onset_buddy.onset[0]:
                                onset_buddy.onset = np.append(0, onset_buddy.onset)
                                onset_buddy.data = np.append(onset_buddy.data[0], onset_buddy.data)
                            if (onset_buddy.onset[-1] > onset_buddy.offset[-1]):
                                onset_buddy.offset = np.append(onset_buddy.offset, np.inf)
                        curr_epoc.data = []
                        curr_epoc.data.append(np.reshape(heads[[[6], [7]], valid_ind].T, (-1, 1)).T.view(np.float64))
                        curr_epoc.data = np.concatenate(curr_epoc.data, axis=1)[0]
                        curr_epoc.data = curr_epoc.data.view(curr_epoc.dform.to_np())
                        # if note is not None:
                        #     curr_epoc.notes.append(note)
                        data.epocs.append(curr_epoc)
                elif store_code["type_str"] == "streams":
                    if not store_code["name"] in [x.header.name for x in data.streams]:
                        curr_stream = TDTStream()
                        curr_stream.header.name = store_code["name"]
                        curr_stream.code = store_code["code"]
                        curr_stream.header.size = store_code["size"]
                        curr_stream.header.type = store_code["type"]
                        curr_stream.header.type_str = store_code["type_str"]
                        curr_stream.ucf = store_code["ucf"]
                        curr_stream.fs = np.double(
                            np.array([store_code["temp"][9]]).view(np.float32)[0]
                        )
                        dform_num = store_code["dform"]
                        curr_stream.dform = DataFormatEnum(dform_num)

                        curr_stream.ts = np.append(curr_stream.ts, time2sample((np.reshape(heads[[[4], [5]], valid_ind].T, (-1, 1)).T.view(np.float64)- data.header.start_time), to_time=True))
                        # we ignore nodata flag for streams
                        curr_stream.data = []
                        curr_stream.data.append(np.reshape(heads[[[6], [7]], valid_ind].T, (-1, 1)).T.view(np.float64))
                        curr_stream.data = np.concatenate(curr_stream.data, axis=1)[0]
                        curr_stream.data = curr_stream.data.view(np.uint64)
                        # FIXME: I don't think chan will ever have more than one element so no need to concat
                        curr_stream.chan = []
                        curr_stream.chan = np.append(curr_stream.chan, temp[::2])
                        if len(curr_stream.chan.shape) > 1:
                            curr_stream.chan = np.concatenate(curr_stream.chan)
                        if np.max(curr_stream.chan) == 1:
                            curr_stream.chan = [1]
                        data.streams.append(curr_stream)
                elif store_code["type_str"] == "snips":
                    if not store_code["name"] in [x.header.name for x in data.snips]:
                        curr_snip = TDTSnip()
                        curr_snip.header.name = store_code["name"]
                        curr_snip.code = store_code["code"]
                        curr_snip.header.size = store_code["size"]
                        curr_snip.header.type = store_code["type"]
                        curr_snip.header.type_str = store_code["type_str"]
                        curr_snip.fs = np.double(
                            np.array([store_code["temp"][9]]).view(np.float32)[0]
                        )
                        curr_snip.dform = store_code["dform"]
                        curr_snip.ts = []
                        curr_snip.ts.append(time2sample((np.reshape(heads[[[4], [5]], valid_ind].T, (-1, 1)).T.view(np.float64)- data.header.start_time), to_time=True))
                        if not nodata:
                            curr_snip.data = []
                            curr_snip.data.append(np.reshape(heads[[[6], [7]], valid_ind].T, (-1, 1)).T.view(np.float64))
                        curr_snip.data = np.concatenate(curr_snip.data, axis=1)[0].view(np.uint64)
                        curr_snip.chan = []
                        curr_snip.chan.append(temp[::2])
                        curr_snip.chan = np.concatenate(curr_snip.chan)
                        curr_snip.sortcode = []
                        if np.max(curr_snip.chan) == 1:
                            curr_snip.chan = [1]
                        if len(custom_sort_codes) > 0 and var_name in custom_sort_event:
                            # apply custom sort codes
                            for tempp in range(len(custom_sort_event)):
                                if (
                                    type(custom_sort_codes[tempp]) == np.ndarray
                                    and curr_snip.header.name
                                    == custom_sort_event[tempp]
                                ):
                                    sortchannels = np.where(
                                        custom_sort_channel_map[tempp]
                                    )[0]
                                    curr_snip.sortcode.append(
                                        custom_sort_codes[tempp][valid_ind + code_ct]
                                    )
                                    code_ct += len(codes)
                                    curr_snip.sortname = sortname
                                    curr_snip.sortchannels = sortchannels
                                elif (
                                    type(custom_sort_codes[tempp]) != np.ndarray
                                    and curr_snip.header.name
                                    == custom_sort_event[tempp]
                                ):
                                    curr_snip.sortcode.append(temp[1::2])
                                    curr_snip.sortname = "TankSort"
                                else:
                                    continue
                        else:
                            curr_snip.sortcode.append(temp[1::2])
                            curr_snip.sortname = "TankSort"
                        curr_snip.sortcode = np.concatenate(curr_snip.sortcode)
                        data.snips.append(curr_snip)
                elif store_code["type_str"] == "scalars":
                    if not store_code["name"] in [x.header.name for x in data.scalars]:
                        curr_scalar = TDTScalar()
                        curr_scalar.header.name = store_code["name"]
                        curr_scalar.code = store_code["code"]
                        curr_scalar.header.size = store_code["size"]
                        curr_scalar.header.type = store_code["type"]
                        curr_scalar.header.type_str = store_code["type_str"]
                        dform_num = store_code["dform"]
                        curr_scalar.dform = DataFormatEnum(dform_num)
                        # if data.header.start_time is not None:
                        #     note = get_store_notes(
                        #         user_notes = heads[9, valid_ind].view(np.uint32),
                        #         store_name = curr_scalar.header.name,
                        #         heads = heads,
                        #         header_start_time = data.header.start_time,
                        #         valid_ind = valid_ind
                        #     )
                        # else:
                        #     warnings.warn(
                        #         f"Start time not found, skipping note extraction for {curr_scalar.header.name}",
                        #         Warning,
                        #         stacklevel=2,
                        #     )
                        # if note is not None:
                        #     curr_scalar.notes.append(note)
                        curr_scalar.ts = np.append(curr_scalar.ts, time2sample((np.reshape(heads[[[4], [5]], valid_ind].T, (-1, 1)).T.view(np.float64)- data.header.start_time), to_time=True))
                        if not nodata:
                            curr_scalar.data = np.append(curr_scalar.data, np.reshape(heads[[[6], [7]], valid_ind].T, (-1, 1)).T.view(np.float64))
                        if len(curr_scalar.data.shape) > 1:
                            curr_scalar.data = np.concatenate(curr_scalar.data, axis=1)[0]
                        curr_scalar.chan.append(temp[::2])
                        curr_scalar.chan = np.concatenate(curr_scalar.chan)
                        if np.max(curr_scalar.chan) == 1:
                            curr_scalar.chan = [1]
                        data.scalars.append(curr_scalar)
                else:
                    warnings.warn(
                        f"Unknown store type {store_code['type_str']}, skipping", Warning, stacklevel=2
                    )
                    return None
            del temp
            del codes

            last_ts = heads[[4, 5], -1].view(np.float64) - data.header.start_time
            last_ts = last_ts[0]

            # break early if time filter
            if t2 > 0 and last_ts > t2:
                break

            # eof reached
            if heads.size < read_size:
                break

        # make fake Note epoc if it doesn't exist already
        # if len(note_text) > 0 and "Note" not in [x.header.name for x in data.epocs]:
        #     note_epoc = TDTEpoc()
        #     note_epoc.header.name="Note"
        #     note_epoc.buddy = ("    ")
        #     note_epoc.code = np.array([ord(x) for x in "Note"], dtype=np.uint8).view(np.uint32)[0]
        #     note_epoc.ts = np.append(note_epoc.ts, np.array(note_ts, dtype=np.float64))
        #     note_epoc.header.type ="onset"
        #     note_epoc.header.type_str = "epocs"
        #     # note_epoc.typeNum = 2 # this is never used?
        #     note_epoc.data = np.append(note_epoc.data, np.arange(1, len(note_ts) + 1))
        #     note_epoc.dform = DataFormatEnum.DOUBLE

        # IDK WHAT THIS DOES BUT THE SAY: "fix[es] secondary epoc offsets"
        if len(block_notes) > 0:
            for epoc in data.epocs:
                if epoc.header.type == "onset":
                    var_name = epoc.header.name
                    epoc_of_intrest = get_event_data_by_name(var_name, "epocs")
                    for store in block_notes.values():
                        if store.StoreName == epoc_of_intrest.header.name:
                            head_name = store.HeadName
                            if "|" in head_name:
                                primary = head_name[-4:]
                                if primary in [x.header.name for x in data.epocs]:
                                    epoc_of_intrest.offset = get_event_data_by_name(primary, "epocs").offset

        # if there is a custom sort name but this store ID isn't included, ignore it altogether
        for snip in data.snips:
            if "snips" in evtype and sortname != "TankSort":
                if not snip.sortcode: # if empty
                    data.snips.remove(snip)

    del heads

    if headers_only:
        try:
            tsq.close()
        except:
            pass
        return data

    if t2 > 0:
        valid_time_range = np.array([[t1], [t2]])
    else:
        valid_time_range = np.array([[t1], [np.inf]])

    if hasattr(ranges, "__len__"):
        valid_time_range = ranges

    num_ranges = valid_time_range.shape[1]
    if num_ranges > 0:
        data.time_ranges = valid_time_range

    # loop through all event stores and do a time filter
    for epoc in data.epocs:
        start_time, filtered_data, filtered_ts, filtered_chan, filtered_sort_code, filtered_onset, filtered_offset = _time_filter(event=epoc, valid_time_range=valid_time_range, num_ranges=num_ranges)
        epoc.data = filtered_data
        epoc.ts = filtered_ts
        epoc.onset = filtered_onset
        epoc.offset = filtered_offset
    for snip in data.snips:
        start_time, filtered_data, filtered_ts, filtered_chan, filtered_sort_code, filtered_onset, filtered_offset = _time_filter(event=snip, valid_time_range=valid_time_range, num_ranges=num_ranges)
        snip.data = filtered_data
        snip.ts = filtered_ts
        snip.chan = filtered_chan
        snip.sortcode = filtered_sort_code
    for stream in data.streams:
        start_time, filtered_data, filtered_ts, filtered_chan, filtered_sort_code, filtered_onset, filtered_offset = _time_filter(event=stream, valid_time_range=valid_time_range, num_ranges=num_ranges)
        stream.data = filtered_data
        stream.ts = filtered_ts
        stream.chan = filtered_chan
    for scalar in data.scalars:
        start_time, filtered_data, filtered_ts, filtered_chan, filtered_sort_code, filtered_onset, filtered_offset = _time_filter(event=scalar, valid_time_range=valid_time_range, num_ranges=num_ranges)
        scalar.data = filtered_data
        scalar.ts = filtered_ts
        scalar.chan = filtered_chan


        
    # see which stores might be in SEV files
    # sev_names = read_sev(block_path, just_names=True)

    for scalar in data.scalars:
        if len(scalar.chan) > 0:
            nchan = int(np.max(scalar.chan))
        else:
            nchan = 0
        if nchan > 1:
            # organize data by sample
            # find channels with most and least amount of data
            ind = []
            min_length = np.inf
            max_length = 0
            for xx in range(nchan):
                ind.append(
                    np.where(scalar.chan == xx + 1)[0]
                )
                min_length = min(len(ind[-1]), min_length)
                max_length = max(len(ind[-1]), max_length)
            if min_length != max_length:
                warnings.warn(
                    "Truncating store {0} to {1} values (from {2})".format(
                        scalar.header.name, min_length, max_length
                    ),
                    Warning,
                )
                ind = [ind[xx][:min_length] for xx in range(nchan)]
            if not nodata:
                scalar.data = (
                    scalar.data[np.concatenate(ind)]
                    .reshape(nchan, -1)
                )

            # only use timestamps from first channel
            scalar.ts = scalar.ts[ind[0]]

        # if (
        #     hasattr(data[current_type_str][current_name], "notes")
        #     and current_name != "Note"
        # ):
        #     data[current_type_str][current_name].notes.notes = (
        #         data[current_type_str][current_name].notes.notes[np.newaxis].T
        #     )
        #     data[current_type_str][current_name].notes.index = (
        #         data[current_type_str][current_name].notes.index[np.newaxis].T
        #     )
        #     data[current_type_str][current_name].notes.ts = (
        #         data[current_type_str][current_name].notes.ts[np.newaxis].T
        #     )

    for snip in data.snips:
            # data[current_type_str][current_name].name = current_name
            # data[current_type_str][current_name].fs = stream.fs

            all_ch = set(snip.chan)

            # make channel filter a list
            if type(channel) is not list:
                channels = [channel]
            else:
                channels = channel
            if 0 in channels:
                channels = list(all_ch)
                use_all_known = True
            channels = sorted(list(set(channels)))
            use_all_known = len(all_ch.intersection(set(channels))) == len(all_ch)

            if not use_all_known:
                # find valid indicies that match our channels
                valid_ind = [
                    i
                    for i, x in enumerate(snip.chan)
                    if x in channels
                ]
                if len(valid_ind) == 0:
                    raise Exception("channels {0} not found".format(repr(channels)))
                if not nodata:
                    all_offsets = snip.data[valid_ind]
                snip.chan = snip.chan[valid_ind]
                snip.sortcode = snip.sortcode[valid_ind]
                snip.ts = snip.ts[valid_ind]
            else:
                if not nodata:
                    all_offsets = snip.data

            if len(snip.chan) > 1:
                snip.chan = (
                    snip.chan[np.newaxis].T
                )
            snip.sortcode = (snip.sortcode[np.newaxis].T)
            snip.ts = (snip.ts[np.newaxis].T)

            if not nodata:
                # try to optimally read data from disk in bigger chunks
                max_read_size = 10000000
                iter = 2048
                arr = np.array(range(0, len(all_offsets), iter))
                if len(arr) > 0:
                    markers = all_offsets[arr]
                else:
                    markers = np.array([])

                while (
                    len(markers) > 1
                    and max(np.diff(markers)) > max_read_size
                    and iter > 1
                ):
                    iter = max(iter // 2, 1)
                    markers = all_offsets[range(0, len(all_offsets), iter)]

                arr = range(0, len(all_offsets), iter)

                snip.data = []
                npts = (snip.header.size - np.uint32(10)) * np.uint32(4) // sz
                event_count = 0
                for f in range(len(arr)):
                    tev.seek(markers[f], os.SEEK_SET)

                    # do big-ish read
                    if f == len(arr) - 1:
                        read_size = (all_offsets[-1] - markers[f]) // sz + npts
                    else:
                        read_size = (markers[f + 1] - markers[f]) // sz + npts

                    tev_data = np.frombuffer(
                        tev.read(read_size * sz), snip.dform.to_np()
                    )

                    # we are covering these offsets
                    start = arr[f]
                    stop = min(arr[f] + iter, len(all_offsets))
                    xxx = all_offsets[start:stop]

                    # convert offsets from bytes to indices in data array
                    relative_offsets = ((xxx - min(xxx)) // sz)[np.newaxis].T
                    ind = relative_offsets + np.tile(
                        range(npts), [len(relative_offsets), 1]
                    )

                    # if we are missing data, there will be duplicates in the ind array
                    _, unique_ind = np.unique(ind[:, 0], return_index=True)
                    if len(unique_ind) != len(ind[:, 0]):
                        # only keep uniques
                        ind = ind[unique_ind, :]
                        # remove last row
                        ind = ind[:-1, :]
                        warnings.warn(
                            "data missing from tev file for store:{0} time:{1}s".format(
                                snip.header.name,
                                np.round(
                                    snip.ts[
                                        event_count + len(ind)
                                    ][0],
                                    3,
                                ),
                            ),
                            Warning,
                        )
                        if len(ind) == 0:
                            continue

                    # add data to big array
                    snip.data.append(tev_data[ind.flatten().astype(np.uint64)].reshape((-1, npts)))
                    event_count += len(ind)

                # convert cell array for output
                if len(snip.data) > 1:
                    snip.data = np.concatenate(snip.data
                    )
                elif np.size(snip.data) == 0:
                    snip.data = []
                else:
                    snip.data = snip.data[0]

                totalEvents = len(snip.data)
                if len(snip.chan) > 1:
                    snip.chan = snip.chan[:totalEvents]
                snip.sortcode = snip.sortcode[:totalEvents]
                snip.ts = snip.ts[:totalEvents]

    for stream in data.streams:
        # data[current_type_str][current_name].name = current_name
        # data[current_type_str][current_name].fs = stream.fs
        sz = np.uint64(np.dtype(stream.dform.to_np()).itemsize)
        # make sure SEV files are there if they are supposed to be
        if stream.ucf == 1:
            warnings.warn(
                "Expecting SEV files for {0} but none were found, skipping...".format(
                    stream.header.name
                ),
                Warning,
            )
            continue

        filtered_data = stream.data
        stream.data = [
            [] for i in range(num_ranges)
        ]
        for jj in range(num_ranges):
            fc = stream.chan[jj]
            if len(fc) < 1:
                continue

            # make channel filter into a list
            if type(channel) is not list:
                channels = [channel]
            else:
                channels = channel
            if 0 in channels:
                channels = fc
            channels = sorted(list(set(channels)))

            if len(fc) == 1:
                # there is only one channel here, use them all
                valid_ind = np.arange(
                    len(filtered_data[jj])
                )
                nchan = np.uint64(1)
            elif len(channels) == 1:
                valid_ind = fc == channels[0]
                if not np.any(valid_ind):
                    raise Exception(
                        "channel {0} not found in store {1}".format(
                            channels[0], stream.header.name
                        )
                    )
                fc = fc[valid_ind]
                nchan = np.uint64(1)
            else:
                # use selected channels only
                valid_ind = [i for i, x in enumerate(fc) if x in channels]
                fc = fc[valid_ind]
                nchan = np.uint64(len(list(set(fc))))
            logger.info(f"Reading {nchan} channels from {stream.header.name}")
            chan_index = np.zeros(nchan, dtype=np.uint64)
            these_offsets = np.array(filtered_data[jj])[valid_ind]

            # preallocate data array
            npts = (stream.header.size - np.uint32(10)) * np.uint32(4) // sz 
            # size_in_bytes = current_data_format(1).itemsize * nchan * npts * np.uint64(len(these_offsets)) // nchan
            stream.data[jj] = np.zeros(
                [nchan, npts * np.uint64(len(these_offsets)) // nchan],
                dtype=stream.dform.to_np(),
            )
            max_read_size = 10000000
            iter = max(min(8192, len(these_offsets) - 1), 1)
            arr = np.array(range(0, len(these_offsets), iter))
            if len(arr) > 0:
                markers = these_offsets[arr]
            else:
                markers = np.array([])
            while (
                len(markers) > 1
                and max(np.diff(markers)) > max_read_size
                and iter > 1
            ):
                iter = max(iter // 2, 1)
                markers = these_offsets[range(0, len(these_offsets), iter)]

            arr = range(0, len(these_offsets), iter)

            # create export file handles
            channel_offset = 0
            for f in range(len(arr)):
                tev.seek(np.int64(markers[f]), os.SEEK_SET)
                
                # do big-ish read
                if f == len(arr) - 1:
                    read_size = (these_offsets[-1] - markers[f]) // sz + npts
                else:
                    read_size = (markers[f+1] - markers[f]) // sz + npts
                
                tev_data = np.frombuffer(tev.read(np.int64(read_size * sz)), stream.dform.to_np())
                
                # we are covering these offsets
                start = arr[f]
                stop = min(arr[f]+iter, len(these_offsets))
                xxx = these_offsets[start:stop]
            
                # convert offsets from bytes to indices in data array
                relative_offsets = ((xxx - min(xxx))/sz)[np.newaxis].T
                ind = relative_offsets + np.tile(range(npts), [len(relative_offsets), 1])
                
                # loop through values, filling array
                found_empty = False
                for kk in range(len(relative_offsets)):
                    if nchan > 1:
                        arr_index = channels.index(fc[channel_offset])
                    else:
                        arr_index = 0
                    channel_offset += 1
                    if not np.any(ind[kk,:] <= len(tev_data)):
                        if not found_empty:
                            warnings.warn(f'Data missing from tev file for store {stream.header.name} time {stream.ts[jj][kk]}s', Warning)
                            found_empty = True
                        chan_index[arr_index] += npts
                        continue
                    found_empty = False
                    stream.data[jj][arr_index, chan_index[arr_index]:(chan_index[arr_index] + npts)] = tev_data[ind[kk,:].flatten().astype(np.uint64)]
                    chan_index[arr_index] += npts

            # add data to big cell array
            # convert cell array for output
            # be more exact with streams time range filter.
            # keep timestamps >= valid_time_range(1) and < valid_time_range(2)
            # index 1 is at header.stores.(current_name).start_time
            # round timestamps to the nearest sample
            
            # actual time the segment starts on
            td_time = time2sample(stream.header.start_time[jj], stream.fs, to_time=True)
            tdSample = time2sample(td_time, stream.fs, t1=1)
            ltSample = time2sample(valid_time_range[0,jj], stream.fs, t1=1)
            minSample = ltSample - tdSample
            if np.isinf(valid_time_range[1,jj]):
                maxSample = MAX_UINT64
            else:
                etSample = time2sample(valid_time_range[1,jj], stream.fs, t1=1)
                maxSample = etSample - tdSample - 1
            stream.data[jj] = stream.data[jj][:,minSample:int(maxSample+1)]
            stream.header.start_time[jj] = ltSample / stream.fs
    
        stream.chan = channels

    stream.data = np.array(stream.data)
    if len(stream.data) == 1:
        stream.data = stream.data[0]
        if len(stream.data) > 0:
            if stream.data.shape[0] == 1:
                stream.data = stream.data[0]
        stream.header.start_time = stream.header.start_time[0]
    

    if not use_outside_headers:
        try:
            tsq.close()
        except:
            pass
    try:
        tev.close()
    except:
        pass

    # find SEV files that weren't in TSQ file
    # if "streams" in evtype:
    #     for sev_name in sev_names:
    #         # if looking for a particular store and this isn't it, skip it
    #         if store:
    #             if isinstance(store, str):
    #                 if sev_name != store:
    #                     continue
    #             elif isinstance(store, list):
    #                 if sev_name not in store:
    #                     continue
    #         if sev_name not in [data.streams[fff].name for fff in data.streams.keys()]:
    #             d = read_sev(
    #                 block_path,
    #                 event_name=sev_name,
    #                 channel=channel,
    #                 verbose=verbose,
    #                 t1=t1,
    #                 t2=t2,
    #                 ranges=ranges,
    #                 export=export,
    #                 scale=scale,
    #             )
    #             if d is not None:  # exporting returns None
    #                 data.streams[sev_name] = d[sev_name]
    #                 data.streams[sev_name].start_time = time2sample(
    #                     t1, t1=True, to_time=True
    #                 )

    # if hasattr(combine, "__len__"):
    #     for store in combine:
    #         if not hasattr(data.snips, store):
    #             raise Exception(
    #                 "specified combine store name {0} is not in snips structure".format(
    #                     store
    #                 )
    #             )
    #         data.snips[store] = snip_maker(data.snips[store])

    # if bitwise != "":
    #     if not (hasattr(data.epocs, bitwise) or hasattr(data.scalars, bitwise)):
    #         raise Exception(
    #             "specified bitwise store name {0} is not in epocs or scalars".format(
    #                 bitwise
    #             )
    #         )

    #     nbits = 32
    #     if hasattr(data.epocs, bitwise):
    #         bitwisetype = "epocs"
    #     else:
    #         bitwisetype = "scalars"

    #     if not hasattr(data[bitwisetype][bitwise], "data"):
    #         raise Exception("data field not found")

    #     # create big array of all states
    #     sz = len(data[bitwisetype][bitwise].data)
    #     big_array = np.zeros((nbits + 1, sz))
    #     if bitwisetype == "epocs":
    #         big_array[0, :] = data[bitwisetype][bitwise].onset
    #     else:
    #         big_array[0, :] = data[bitwisetype][bitwise].ts

    #     data[bitwisetype][bitwise].bitwise = tdt.StructType()

    #     # loop through all states
    #     prev_state = np.zeros(nbits)
    #     for i in range(sz):
    #         xxx = np.array([data[bitwisetype][bitwise].data[i]], dtype=np.uint32)

    #         curr_state = np.unpackbits(xxx.byteswap().view(np.uint8))
    #         big_array[1 : nbits + 1, i] = curr_state

    #         # look for changes from previous state
    #         changes = np.where(np.logical_xor(curr_state, prev_state))[0]

    #         # add timestamp to onset or offset depending on type of state change
    #         for ind in changes:
    #             ffield = "bit" + str(nbits - ind - 1)
    #             if curr_state[ind]:
    #                 # nbits-ind reverses it so b0 is bbb(end)
    #                 if hasattr(data[bitwisetype][bitwise].bitwise, ffield):
    #                     data[bitwisetype][bitwise].bitwise[
    #                         ffield
    #                     ].onset = np.concatenate(
    #                         [
    #                             data[bitwisetype][bitwise].bitwise[ffield].onset,
    #                             [big_array[0, i]],
    #                         ]
    #                     )
    #                 else:
    #                     data[bitwisetype][bitwise].bitwise[ffield] = tdt.StructType()
    #                     data[bitwisetype][bitwise].bitwise[ffield].onset = np.array(
    #                         [big_array[0, i]]
    #                     )
    #                     data[bitwisetype][bitwise].bitwise[ffield].offset = np.array([])
    #             else:
    #                 data[bitwisetype][bitwise].bitwise[ffield].offset = np.concatenate(
    #                     [
    #                         data[bitwisetype][bitwise].bitwise[ffield].offset,
    #                         [big_array[0, i]],
    #                     ]
    #                 )
    #         prev_state = curr_state

    #     # add 'inf' to offsets that need them
    #     for i in range(nbits):
    #         ffield = "bit" + str(i)
    #         if hasattr(data[bitwisetype][bitwise].bitwise, ffield):
    #             if len(data[bitwisetype][bitwise].bitwise[ffield].onset) - 1 == len(
    #                 data[bitwisetype][bitwise].bitwise[ffield].offset
    #             ):
    #                 data[bitwisetype][bitwise].bitwise[ffield].offset = np.concatenate(
    #                     [data[bitwisetype][bitwise].bitwise[ffield].offset, [np.inf]]
    #                 )

    return data


class TDTLoader:
    def __init__(self, path: str):
        super(TDTLoader, self).__init__()
        self.path = path
        self.max = None
        self.block = None

    def load_block(self):
        self.block = read_block(self.path, evtype=["epocs"])

    def run(self):
        self.load_block()

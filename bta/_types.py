from re import A
from pydantic import BaseModel
from enum import StrEnum, Enum, IntEnum
from typing import Union, List, Dict, Tuple, Literal, TypeAlias, Any
from datetime import datetime
from datetime import timedelta
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel

class TankEventType(IntEnum):
    UNKNOWN = int('00000000', 16)
    STRON = int('00000101', 16)
    STROFF = int('00000102', 16)
    SCALAR = int('00000201', 16)
    STREAM = int('00008101', 16)
    SNIP = int('00008201', 16)
    MARK = int('00008801', 16)
    HASDATA = int('00008000', 16)
    UCF = int('00000010', 16)
    PHANTOM = int('00000020', 16)
    MASK = int('0000FF0F', 16)
    INVALID_MASK = int('FFFF0000', 16)

class EventMarker(IntEnum):
    STARTBLOCK = int('0001', 16)
    STOPBLOCK = int('0002', 16)

class AllowedFormats(Enum):
    FLOAT = np.float32
    LONG = np.int32
    SHORT = np.int16
    BYTE = np.int8
    DOUBLE = np.float64
    QWORD = np.int64

class DataFormatEnum(IntEnum):
    FLOAT = 0
    LONG = 1
    SHORT = 2
    BYTE = 3
    DOUBLE = 4
    QWORD = 5
    TYPE_COUNT = 6

    def to_np(self):
        if self.name in [x.name for x in AllowedFormats]:
            return AllowedFormats[self.name].value
        else:
            raise ValueError(f"DataFormatEnum {self.name} not in AllowedFormats")
        
class AllowedEvtypes(StrEnum):
    ALL = "all"
    EPOCS = "epocs"
    SNIPS = "snips"
    STREAMS = "streams"
    SCALARS = "scalars"
    
@dataclass
class BlockInfo:
    tankpath: str
    blockname: str
    blockpath: str
    start_date: str
    utc_start_time: str
    stop_date: str
    utc_stop_time: str
    duration: str
    video_path: str


class Block:
    epocs: Dict[str, Any]
    streams: Dict[str, Any]
    info: BlockInfo

class TDTData(BaseModel):
    pass
    
class SampleInfo(BaseModel):
    name: str = ""
    start_sample: int = 0
    hour: int = 0
    gaps: list[Tuple[int, int]] = []
    gap_text: str = ""

class StreamHeader(BaseModel):
    size_bytes: int = 0
    file_type: str = ""
    file_version: int = 0
    event_name: str = ""
    channel_num: int = 0
    total_num_channels: int = 0
    sample_width_bytes: int = 0
    data_format: str = ""
    decimate: int = 0
    rate: int = 0
    fs: float = 0

class StoreType(BaseModel):
    CircType: int = 0
    DataFormat: DataFormatEnum = DataFormatEnum.DOUBLE
    Enabled: bool = False
    HeadName: str = ""
    NumChan: int = 0
    NumPoints: int = 0
    SampleFreq: float = 0
    SecTag: str = ""
    StoreName: str = ""
    StrobeBuddy: str = ""
    StrobeMode: int = 0
    TankEvType: int = 0

class EventHeader(BaseModel):
    name: str = ""
    type: str = ""
    start_time: List[float] = []
    type_str: AllowedEvtypes = AllowedEvtypes.EPOCS
    size: int = 0

    def __repr__(self):
        return f"""
name: {self.name}
type: {self.type}
start_time: {self.start_time}
type_str: {self.type_str}
size: {self.size}
"""

class TDTNote(BaseModel):
    name: List = []
    index: List = []
    text: List = []
    ts: List = []

    
class TDTDataHeader(BaseModel):
    tev_path: str = ""
    start_time: Union[float, None] = None
    stop_time: Union[float, None] = None
    
class Event(BaseModel):
    header: EventHeader = EventHeader()
    ts: np.ndarray = np.array([])
    data: np.ndarray = np.array([])
    code: int = 0
    dform: DataFormatEnum = DataFormatEnum.DOUBLE

    class Config:
        arbitrary_types_allowed = True

    def __repr__(self):
        return str(self.header) + f"""
ts: {self.ts}
data: {self.data}
code: {self.code}
dform: {self.dform}
    """

    def __str__(self):
        return self.__repr__()

class TDTEpoc(Event):
    buddy: str = ""
    onset: np.ndarray = np.array([])
    offset: np.ndarray = np.array([])
    notes: List[TDTNote] = []

    def __repr__(self):
        super().__repr__()
        return str(self.header) + f"""
onset: {self.onset}
offset: {self.offset}
    """

class TDTSnip(Event):
    fs: np.double = 0.0
    chan: np.ndarray = np.array([])
    sortcode: np.ndarray = np.array([])
    sortname: str = ""
    sortchannels: np.ndarray = np.array([])

class TDTStream(Event):
    ucf: bool = False
    fs: np.double = 0.0
    chan: np.ndarray = np.array([])

class TDTScalar(Event):
    chan: List = []
    notes: List[TDTNote] = []
    
class TDTInfo(Event):
    tankpath: str = ""
    blockname: str = ""
    start_date: datetime = datetime.now()
    utc_start_time: Union[str, None] = None
    stop_date: Union[datetime, None] = None
    utc_stop_time: Union[str, None] = None
    duration: Union[timedelta, None] = None
    stream_channel: int = 0
    snip_channel: int = 0
    experiment: str = ""
    subject: str = ""
    user: str = ""
    start: str = ""
    stop: str = ""
     


class TDTData(BaseModel):
    header: TDTDataHeader = TDTDataHeader()
    info: TDTInfo = TDTInfo()
    time_ranges: np.ndarray = np.array([]) # represents the time ranges of the data, can be [[0], [np.inf]] for all data or [[start], [stop]] for a specific range
    epocs: List[TDTEpoc] = []
    snips: List[TDTSnip] = []
    streams: List[TDTStream] = []
    scalars: List[TDTScalar] = []

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist()
        }
        json_decoders = {
            np.ndarray: lambda v: np.array(v)
        }

    # repr as a list of the names of the events if they are not empty
    def __repr__(self):
        res = []
        if self.epocs:
            for idx, e in enumerate(self.epocs):
                if idx == 0:
                   x = f"Epocs:   {e.header.name}\n"
                else:
                    x += f"         {e.header.name}\n"
            # remove the last newline from the string
            x = x.split("\n")[:-1]
            x = "\n".join(x)
            res.append(x)
        if self.snips:
            for idx, s in enumerate(self.snips):
                if idx == 0:
                    x = f"Snips:  {s.header.name}\n"
                else: x += f"         {s.header.name}\n"
            x = x[:-1]
            res.append(x)
        if self.streams:

            for idx, s in enumerate(self.streams):
                if idx == 0:
                    x = f"Streams: {s.header.name}\n"
                else: x += f"         {s.header.name}\n"
            x = x[:-1]
            res.append(x)
        if self.scalars:
            for idx, s in enumerate(self.scalars):
                if idx == 0:
                    x = f"Scalars: {s.header.name}\n"
                else: x += f"         {s.header.name}\n"
            x = x[:-1]
            res.append(x)

        return "\n".join(res)

    def __str__(self):
        return self.__repr__()

    def get_epoc(self, name: str) -> Union[TDTEpoc, None]:
        """
        Get an epoc by name

        Parameters
        ----------
        name : str
            The name of the epoc to get

        Returns
        -------
        Union[TDTEpoc, None]
            The epoc if it exists, otherwise None
        """
        for e in self.epocs:
            if e.header.name == name:
                return e
        return None

    def get_snip(self, name: str) -> Union[TDTSnip, None]:
        """
        Get a snip by name

        Parameters
        ----------
        name : str
            The name of the snip to get

        Returns
        -------
        Union[TDTSnip, None]
            The snip if it exists, otherwise None
        """
        for s in self.snips:
            if s.header.name == name:
                return s
        return None

    def get_stream(self, name: str) -> Union[TDTStream, None]:
        """
        Get a stream by name

        Parameters
        ----------
        name : str
            The name of the stream to get

        Returns
        -------
        Union[TDTStream, None]
            The stream if it exists, otherwise None
        """
        for s in self.streams:
            if s.header.name == name:
                return s
        return None

    def get_scalar(self, name: str) -> Union[TDTScalar, None]:
        """
        Get a scalar by name

        Parameters
        ----------
        name : str
            The name of the scalar to get

        Returns
        -------
        Union[TDTScalar, None]
            The scalar if it exists, otherwise None
        """
        for s in self.scalars:
            if s.header.name == name:
                return s
        return None

    def get_event(self, name: str) -> Union[TDTEpoc, TDTSnip, TDTStream, TDTScalar, None]:
        """
        Get an event by name

        Parameters
        ----------
        name : str
            The name of the event to get

        Returns
        -------
        Union[TDTEpoc, TDTSnip, TDTStream, TDTScalar, None]
            The event if it exists, otherwise None
        """
        e = self.get_epoc(name)
        if e:
            return e
        s = self.get_snip(name)
        if s:
            return s
        st = self.get_stream(name)
        if st:
            return st
        sc = self.get_scalar(name)
        if sc:
            return sc
        return None

    

    
    
    
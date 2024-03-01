from cProfile import Profile
import os
import logging
import numpy as np
import re
from bta import TDTData, AllowedEvtypes
from typing import List

log_format = '[%(asctime)s] %(levelname)s: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logger = logging.getLogger(__name__)
for handler in logging.root.handlers:
    handler.setFormatter(logging.Formatter(log_format))

def get_files(dir: str, ext: str, ignore_mac: bool = False) -> List[str]:
    result = []
    for file in os.listdir(dir):
        if file.endswith(ext):
            if ignore_mac and file.startswith('._'):
                continue
            result.append(os.path.join(dir, file))
    return result

def read_sev(sev_dir, *, channel=0, event_name='', t1=0, t2=0, fs=0, ranges=None,
           verbose=0, just_names=0, export=None, scale=1, dtype=None, outdir=None,
           prefix=None):
    
    
    if outdir is None and export is not None:
        outdir = sev_dir
    if outdir is not None and export is None:
        logger.warning('outdir specified without export, ignoring')
    
    data = TDTData()
    sample_info = []
    
    if os.path.isfile(sev_dir):
        # treat as single file only
        sev_files = [sev_dir]
    elif os.path.isdir(sev_dir):
        # treat as directory 
        sev_dir = os.path.join(sev_dir, '')
        sev_files = get_files(sev_dir, '.sev', ignore_mac=True)
        
        # parse log files
        if just_names == 0:
            txt_file_list = get_files(sev_dir, '_log.txt', ignore_mac=True)
            
            n_txtfiles = len(txt_file_list)
            if n_txtfiles < 1 and verbose:
                #fprintf('info: no log files in %s\n', SEV_DIR);
                pass
            else:
                start_search = re.compile('recording started at sample: (\d*)')
                gap_search = re.compile('gap detected. last saved sample: (\d*), new saved sample: (\d*)')
                hour_search = re.compile('-(\d)h')
                for txt_path in txt_file_list:
                    if verbose:
                        print('info: log file', txt_path)
                    
                    # get store name
                    temp_sample_info = {'hour':0}
                    temp_sample_info['name'] = os.path.split(txt_path)[-1][:4]
                    with open(txt_path, 'r') as f:
                        log_text = f.read()
                        if verbose:
                            print(log_text)
                    
                    temp_start_sample = start_search.findall(log_text)
                    temp_sample_info['start_sample'] = int(temp_start_sample[0])
                    temp_hour = hour_search.findall(txt_path)
                    if len(temp_hour) > 0:
                        temp_sample_info['hour'] = int(temp_hour[-1])
                    if temp_sample_info['start_sample'] > 2 and temp_sample_info['hour'] == 0:
                        logger.warning(f'{temp_sample_info["name"]} store starts on sample {temp_sample_info["start_sample"]}')
                    # look for gap info
                    temp_sample_info['gaps'] = []
                    temp_sample_info['gap_text'] = ''
                    gap_text = gap_search.findall(log_text)
                    if len(gap_text):
                        temp_sample_info['gaps'] = np.array(gap_text, dtype=np.int64)
                        temp_sample_info['gap_text'] = '\n   '.join([x.group() for x in gap_search.finditer(log_text)])
                        
                        if temp_sample_info['hour'] > 0:
                            logger.warning(f'gaps detected in data set for {temp_sample_info["name"]}-{temp_sample_info["hour"]}h!\n   {temp_sample_info["gap_text"]}')
                        else:
                            logger.warning(f'gaps detected in data set for {temp_sample_info["name"]}!\n   {temp_sample_info["gap_text"]}\nContact TDT for assistance.')
                    sample_info.append(temp_sample_info)
    else:
        raise FileNotFoundError(f'Unable to find sev file or directory:\n\t{sev_dir}')
        
    nfiles = len(sev_files)
    if nfiles < 1:
        if just_names:
            return []
        else:
            raise FileNotFoundError(f'No .sev files found in {sev_dir}')
    if fs > 0:
        logger.info(f'Using {fs:.4f} Hz as SEV sampling rate for {event_name}')
    
    file_list = []
    for file in sev_files:
        [filename, _] = os.path.splitext(file)
        [path, name] = os.path.split(filename)
        if name.startswith('._'):
            continue
        file_list.append({'fullname':file,
                          'folder':path,
                          'name':name})
    print(file_list)

def read_block(block_path, *, bitwise='', channel=0, combine=None, headers=0,
              nodata=False, ranges=None, store='', t1=0, t2=0, evtype=None,
              verbose=0, sortname='TankSort', export=None, scale=1, dtype=None,
              outdir=None, prefix=None, outfile=None, dmy=False):
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
        store       string or list, specify a single store or array of stores
                        to extract.
        channel     integer or list, choose a single channel or array of channels
                        to extract from stream or snippet events. Default is 0,
                        to extract all channels.
        bitwise     string, specify an epoc store or scalar store that 
                        contains individual bits packed into a 32-bit 
                        integer. Onsets/offsets from individual bits will
                        be extracted.
        headers     var, set to 1 to return only the headers for this
                        block, so that you can make repeated calls to read
                        data without having to parse the TSQ file every
                        time, for faster consecutive reads. Once created,
                        pass in the headers using this parameter.
                      example:
                        heads = read_block(block_path, headers=1)
                        data = read_block(block_path, headers=heads, evtype=['snips'])
                        data = read_block(block_path, headers=heads, evtype=['streams'])
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
        dtype       string, data type for exported binary data files
                        None: Uses the format the data was stored in (default)
                        'i16': Converts all data to 16-bit integer format
                        'f32': Converts all data to 32-bit integer format
        outdir      string, output directory for exported files. Defaults to current
                        block folder if not specified
        prefix      string, prefix for output file name. Defaults to None
        outfile     string, output file name for exported files. Defaults to 'export' 
                        if not specified
        dmy         boolean, force dd/MM/yyyy date format for Notes file (Synapse
                        preference)
    """
    
    # if export:
    #     start_time = time.time()
    
    if outdir is None and export is not None:
        outdir = block_path
    if outdir is not None and export is None:
        logger.warn('No export is selected, outdir parameter is ignored')
    if outfile is None and export is not None:
        outfile = 'export.' + export
    if outfile is not None and export is None:
        logger.warn('No export is selected, outfile parameter is ignored')
    
    if not hasattr(evtype, "__len__"):
        evtype = ['all']

    try:
        evtype = [t.lower() for t in evtype]
    except:
        raise ValueError('evtype must be a list of strings')
    
    if 'all' in evtype:
        evtype = ['epocs','snips','streams','scalars']
    else:
        for given_type in evtype:
            if given_type not in [x.value for x in AllowedEvtypes]:
                # print('Unrecognized type: {0}\nAllowed types are: {1}'.format(given_type, tdt.ALLOWED_EVTYPES))
                logger.error(f'Unrecognized type: {given_type}\nAllowed types are: {[x.value for x in AllowedEvtypes]}')
                return None
    
    evtype = list(set(evtype))
    
    use_outside_headers = False
    do_headers_only = False
    if isinstance(headers, TDTData) or isinstance(headers, dict):
        use_outside_headers = True
        header = headers
    else:
        header = TDTData()
        if headers == 1:
            do_headers_only = True

    block_path = os.path.join(block_path, '')
        
if __name__ == "__main__":
    # profile = Profile()
    read_sev(r"H:\D-E mice Recording Backup\Reward\FiPho-230314\d101")
    # profile.runcall(read_block, r"H:\D-E mice Recording Backup\Reward\FiPho-230314\d101")
    # # save the profile results
    # profile.dump_stats("tdt_read_block.prof")
    


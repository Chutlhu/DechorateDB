import h5py
import argparse
import platform

import sofar as sf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from dechorate import constants

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", help="Path to output files", type=str)
    parser.add_argument("--echo", help="Path to dEchorate csv echo annotation", type=str)
    parser.add_argument("--csv", help="Path to dEchorate csv database", type=str)
    parser.add_argument("--hdf", help="Path to dEchorate rir hdf5 dataset", type=str)
    args = parser.parse_args()

    path_to_outdir = Path(args.outdir)
    assert path_to_outdir.exists()

    path_to_hdf = Path(args.hdf)
    dset = h5py.File(path_to_hdf, mode='r')

    path_to_echo_h5 = Path(args.echo)
    note_dset = h5py.File(path_to_echo_h5, 'r')

    src_dir_pos = note_dset['sources_directional_position']
    src_omn_pos = note_dset['sources_omnidirection_position']
    src_pos = np.concatenate([src_dir_pos, src_omn_pos], axis=1)
    assert src_pos.shape == (3,9)
    src_dir_rot = note_dset['sources_directional_direction']
    src_omn_rot = np.zeros((3,3))
    src_rot = np.concatenate([src_dir_rot, src_omn_rot], axis=1)
    assert src_rot.shape == (3,9)

    arr_pos = note_dset['arrays_position']
    arr_rot = note_dset['arrays_direction']
    mic_pos = note_dset['microphones']
    room_size = note_dset['room_size']

    echo_toa = note_dset['echo_toa']
    echo_wall = note_dset['echo_wall']

    signals = list(dset.keys())
    signal = signals[0]
    print('Signals:\t', signals)

    rooms = list(dset[signals[0]].keys())
    print('Rooms:\t', rooms)

    sources = list(dset[signals[0]][rooms[0]].keys())
    print('Sources:\t', sources)
    sources_description = ['Directional Avanton MIXCUBE']*6 + ['Omnidirectional B&G']*3

    path_to_db = Path(args.csv)
    df = pd.read_csv(path_to_db)


    ### RIR MATRIX ###    
    print('Building RIR matrix...', end='')
    D = len(rooms)
    J = len(sources)
    I = 30 # do not count the loopback
    Fs = dset.attrs['sampling_rate']
    L = dset.attrs['n_samples']

    rirs = np.zeros([L, I, J, D])

    for r, room in enumerate(rooms):

        for j, src in enumerate(sources):

            group = f'/{signal}/{room}/{src}'
            data = np.asarray(dset[group])
            
            rirs[:,:,j,r] = data[:L,:I]
    
    print(' Done.')
    print('RIR matrix shape', rirs.shape)

    ### RIRs to SOFAR's DRIRS

    for room_idx in tqdm(range(D), desc='room'):

        for src_idx in tqdm(range(J), desc='src'):
            
            for arr_idx in range(6):

                sofa = sf.Sofa('SingleRoomSRIR')

                # get rirs
                mic_idxs = np.arange(5*arr_idx,5*(arr_idx+1))
                mimo_srir = rirs[:, mic_idxs, src_idx, room_idx]
                
                sofa.GLOBAL_DatabaseName = 'dEchorate'

                title_suffix = f'room{rooms[room_idx]}_src{src_idx+1}_arr{arr_idx+1}_mics{mic_idxs.min()+1}-{mic_idxs.max()+1}'
                sofa.GLOBAL_Title = f'dEchorate_{title_suffix}'

                sofa.GLOBAL_DateCreated = '2024-03-24'
                sofa.GLOBAL_DateModified = sofa.GLOBAL_DateCreated

                sofa.GLOBAL_AuthorContact = 'diego.dicarlo89@gmail.com'
                sofa.GLOBAL_Organization = 'INRIA Rennes (Fr)'
                sofa.GLOBAL_License = 'The MIT License (MIT) Copyright (c) 2019, Diego Di Carlo. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.'
                sofa.GLOBAL_ApplicationName = 'Python'
                sofa.GLOBAL_ApplicationVersion = platform.python_version()


                sofa.GLOBAL_Comment = """- website: 'http://diegodicarlo.com/datasets/dechorate';
                \n- code: 'https://github.com/Chutlhu/DechorateDB'; 
                \n- paper: 'https://doi.org/10.1186/s13636-021-00229-0'"""


                sofa.GLOBAL_History = \
                """- 2019-01-20: Recorded raw data
                \n- 2019-06-17: Processed annotation
                \n- 2021-03-22: Publication on Zenodo v0
                \n- 2022-04-21: Creation SOFA
                \n- 2023-01-26: New Sofa
                \n- 2024-03-24: Updating Sofa
                \n- 2024-07-07: Updating Sofa
                """


                sofa.GLOBAL_References = \
                """@article{dicarlo2021dechorate,
                \ntitle={dEchorate: a calibrated room impulse response dataset for echo-aware signal processing},
                \nauthor={Di Carlo, Diego and Tandeitnik, Pinchas and Foy, Cedri{\'c} and Bertin, Nancy and Deleforge, Antoine and Gannot, Sharon},
                \njournal={EURASIP Journal on Audio, Speech, and Music Processing},
                \nvolume={2021},
                \nnumber={1},
                \npages={1--15},
                \nyear={2021},
                \npublisher={Springer}
                \n}"""

                ## --- RIRS --- ##
                sofa.Data_IR = mimo_srir.T[None,:,:] 
                # must be in M x R x N x E (measurements, receivers, samples, emitters)
                
                
                assert sofa.Data_IR.shape == (1,5,L)
                sofa.Data_SamplingRate = Fs
                sofa.Data_SamplingRate_Units = 'hertz'
                sofa.Data_Delay = [0] * 5

                ## --- ARRAY --- ##

                # Listener = compact microphone array (it may contain many receiver, i.e. mics)
                sofa.GLOBAL_ListenerShortName = f'array_id:{arr_idx}'

                sofa.GLOBAL_ListenerDescription = 'handcrafted 5-mics non-Uniform Linear Array'


                sofa.ListenerPosition = arr_pos[:,arr_idx]
                sofa.ListenerPosition_Type = 'cartesian'
                sofa.ListenerPosition_Units = 'metre'
                sofa.ListenerView = arr_rot[:,arr_idx]
                sofa.ListenerUp = [0,0,1]
                sofa.ListenerView_Type = 'cartesian'
                sofa.ListenerView_Units = 'metre'

                ## --- MICROPHONES --- ##

                # Receiver = MICROPHONE
                sofa.GLOBAL_ReceiverShortName = 'capsule'
                sofa.GLOBAL_ReceiverDescription = ['AKG CK32']
                sofa.ReceiverDescriptions = [['']]*5
                sofa.ReceiverPosition = mic_pos[:,mic_idxs].T
                sofa.ReceiverPosition_Type = 'cartesian'
                sofa.ReceiverPosition_Units = 'metre'
                sofa.ReceiverView = [[1,0,0]]*5
                sofa.ReceiverUp = [[0,0,1]]*5
                sofa.ReceiverView_Type = 'cartesian'
                sofa.ReceiverView_Units = 'metre'

                # --- SOURCES --- ## 

                # Source = compact sound source (it may contain more emitters, i.e. speakers)
                sofa.GLOBAL_SourceShortName = f'loudspeaker_id:{src_idx+1}'
                sofa.GLOBAL_SourceDescription = sources_description[src_idx]
                sofa.SourcePosition = src_pos[:,src_idx]
                sofa.SourcePosition_Type = 'cartesian'
                sofa.SourcePosition_Units = 'metre'
                sofa.SourceView = src_rot[:,src_idx]
                sofa.SourceUp = [0,0,1]
                sofa.SourceView_Type = 'cartesian'
                sofa.SourceView_Units = 'metre'

                ## --- ROOM --- ##
                sofa.GLOBAL_RoomType = 'shoebox'
                room_description = f'room_code:{room}->is reflective?' \
                    + f'floor:{int(int(room[0]) > 0)}_' \
                    + f'ceiling:{int(int(room[1]) > 0)}_' \
                    + f'west:{int(int(room[2]) > 0)}_' \
                    + f'south:{int(int(room[3]) > 0)}_' \
                    + f'east:{int(int(room[4]) > 0)}_' \
                    + f'north:{int(int(room[5]) > 0)}_' \
                    + f'hasFornituer?:{int(int(room[1]) > 1)}'
                
                sofa.GLOBAL_RoomDescription = room_description
                sofa.RoomCornerA = [0,0,0]
                sofa.RoomCornerB = room_size
                sofa.RoomCorners_Type = 'cartesian'
                sofa.RoomCorners_Units = 'metre'

                sofa.verify()

                filename = sofa.GLOBAL_Title
                path_to_out = path_to_outdir / Path(f'{filename}.sofa')
                if not path_to_out.exists():
                    sf.write_sofa(str(path_to_out), sofa, compression=7)

                # testing
                print(sf.read_sofa(str(path_to_out), verify=True, verbose=True))
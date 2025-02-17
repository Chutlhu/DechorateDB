import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf

from tqdm import tqdm
from datetime import date

from dechorate import constants
from dechorate.dataset import DechorateDataset, SyntheticDataset
from dechorate.utils.file_utils import save_to_pickle, load_from_pickle, save_to_matlab
from dechorate.utils.dsp_utils import normalize, resample, envelope, todB, rake_filter
from dechorate.utils.viz_utils import plt_time_signal
from dechorate.utils.evl_utils import snr_dB

from risotto.rtf import estimate_rtf, estimates_PSDs_PSDr_from_RTF
from risotto.utils.dsp_utils import stft, istft

from brioche.beamformer import DS, MVDR, LCMV, GSC
from brioche.utils.dsp_utils import diffuse_noise

import speechmetrics
metrics = speechmetrics.load('pesq', 4)

curr_dir = './recipes/echo_aware_processing/'
data_filename = curr_dir + 'data_notebook.pkl'

dataset_dir = './data/dECHORATE/'
path_to_processed = './data/processed/'
path_to_note_csv = dataset_dir + 'annotations/dECHORATE_database.csv'
path_to_after_calibration = path_to_processed + \
    'post2_calibration/calib_output_mics_srcs_pos.pkl'

# Annotation, RIRs from measurements, 'equivalent' synthetic RIRs
note_dict = load_from_pickle(path_to_after_calibration)
rdset = DechorateDataset(path_to_processed, path_to_note_csv)
sdset = SyntheticDataset()
note_dict.keys()


def main(arr_idx, dataset_idx, target_idx, interf_idx, sir, snr, data_kind, spk_comb, ref_mic=0, render=False):

    print('arr_idx', arr_idx)
    print('dataset_idx', dataset_idx)
    print('target_idx', target_idx)
    print('interf_idx', interf_idx)
    print('sir', sir)
    print('snr', snr)
    print('data_kind', data_kind)
    print('spk_comb', spk_comb)

    # Some constant of the dataset
    L = constants['rir_length']
    Fs = constants['Fs']
    c = constants['speed_of_sound']
    L = constants['rir_length']
    datasets_name = constants['datasets'][:6]
    D = len(datasets_name)

    # which dataset?
    d = dataset_idx

    # which array?
    print(':: Array', arr_idx)

    mics_idxs = [(5*arr_idx + i) for i in range(5)]
    print(':: Mics index', mics_idxs)
    I = len(mics_idxs)
    # which one is the reference mic?
    r = ref_mic
    print(':: Ref mics', mics_idxs[ref_mic])

    # which source?
    if interf_idx == target_idx:
        raise ValueError('Same interf')

    srcs_idxs = [target_idx, interf_idx]
    print(':: Srcs index', srcs_idxs)

    J = len(srcs_idxs)
    # which src is the one to enhance?
    t = 0  # the target
    q = 1  # the interf

    # how many echoes to rake?
    K = 7  # all the first 7
    # in which order?
    ordering = 'strongest'  # earliest, strongest, order

    # get tdoa
    dset = datasets_name[d]
    print(':: Dataset code', dset)

    rirs_real = np.zeros([L, I, J])
    rirs_synt = np.zeros([L, I, J])
    mics = np.zeros([3, I])
    srcs = np.zeros([3, J])
    toas = np.zeros([K, I, J])
    toas_peak = np.zeros([K, I, J])
    toas_cmds = np.zeros([K, I, J])
    amps_cmds = np.zeros([K, I, J])

    for i, m in enumerate(mics_idxs):
        for j, s in enumerate(srcs_idxs):

            # get rir from the recondings
            rdset.set_dataset(dset)
            rdset.set_entry(m, s)
            mic, src = rdset.get_mic_and_src_pos()
            _, rrir = rdset.get_rir()
            diffuse_noise = rdset.get_diffuse_noise(20)

            # measure after calibration
            mics[:, i] = note_dict['mics'][:, m]
            srcs[:, j] = note_dict['srcs'][:, s]

            # get synthetic rir
            sdset = SyntheticDataset()
            sdset.set_room_size(constants['room_size'])
            sdset.set_dataset(dset, absb=0.90, refl=0.2)
            sdset.set_c(c)
            sdset.set_fs(Fs)
            sdset.set_k_order(3)
            sdset.set_k_reflc(1000)
            sdset.set_mic(mics[0, i], mics[1, i], mics[2, i])
            sdset.set_src(srcs[0, j], srcs[1, j], srcs[2, j])
            tk, ak = sdset.get_note(ak_normalize=False, tk_order=ordering)

            ak = ak / (4 * np.pi)

            _, srir = sdset.get_rir(normalize=False)
            srir = srir / (4 * np.pi)

            Ls = min(len(srir), L)
            Lr = min(len(rrir), L)

            # measure after calibration
            rirs_real[:Lr, i, j] = rrir[:Lr]
            rirs_synt[:Ls, i, j] = srir[:Ls]

            toas_peak[:7, i, j] = note_dict['toa_pck'][:7, m, s]
            toas_cmds[:K, i, j] = tk[:K]
            amps_cmds[:K, i, j] = ak[:K]

    print('done with the extraction')
    rirs_real = np.squeeze(rirs_real)
    rirs_synt = np.squeeze(rirs_synt)

    data = {
        'rirs_real': rirs_real,
        'rirs_synt': rirs_synt,
        'mics': mics,
        'srcs': srcs,
        'toas_synt': toas_cmds,
        'toas_peak': toas_peak,
        'amps_synt': amps_cmds,
        'Fs': Fs,
    }
    mic_pos = mics
    arr_pos = np.mean(mics, 1)
    tgt_pos = srcs[:, 0:1]
    itf_pos = srcs[:, 1:2]

    save_to_pickle(data_filename, data)
    print('Saved.')

    data_dir = curr_dir + 'TIMIT_long_nili/'
    s1m = data_dir + 'DR5_MHMG0_SX195_SX285_SX375_7s.wav'
    s2m = data_dir + 'DR7_MGAR0_SX312_SX402_7s.wav'
    s1f = data_dir + 'DR1_FTBR0_SX201_SI921_7s.wav'
    s2f = data_dir + 'DR4_FKLC0_SI985_SI2245_7s.wav'

    files = [s1m, s2m, s1f, s2f]
    N = len(files)

    wavs = []
    for file in files:
        wav, fs = sf.read(file)
        assert len(wav.shape) == 1
        wavs.append(wav[:7*fs])
    print('done.')
    print('Audio rate is', fs)


    if data_kind == 'synt':
        # which rirs?
        h_ = rirs_synt
        # with annotation?
        toas = toas_cmds
        amps = amps_cmds
    if data_kind == 'real':
        h_ = rirs_real

        # restore amps based on direct path eight
        amps_rirs = np.zeros_like(toas_cmds)
        for j in range(J):
            for i in range(I):
                print(i, j)
                for k in range(K):
                    t = int(toas_peak[k, i, j]*Fs)
                    a = np.max(np.abs(rirs_real[t-10:t+10, i, j]))
                    amps_rirs[k, i, j] = a

        amps = amps_rirs
        toas = toas_peak

    r = ref_mic  # reference mic

    print('Ref mic', r)
    print('input SIR', sir)
    print('input SNR', snr)

    s1 = wavs[spk_comb[0]]
    s2 = wavs[spk_comb[1]]

    # center and scale for unit variance
    ss1 = (s1-np.mean(s1))/np.std(s1)
    ss2 = (s2-np.mean(s2))/np.std(s2)
    assert len(ss1) == len(ss2)


    # Upsampling and stacking
    print('Upsampling for convolution', fs, '-->', Fs)
    s_ = np.concatenate([resample(ss1, fs, Fs)[:, None],
                         resample(ss2, fs, Fs)[:, None]], axis=1)
    print(s_.shape)

    Lc = 10*fs
    c_ = np.zeros([Lc, I, J])


    # Convolution, downsampling and stacking
    print('Convolution and downsampling', Fs, '-->', fs)
    for i in range(I):
        for j in range(J):
            cs = np.convolve(h_[:, i, j], s_[:, j], 'full')
            cs = resample(cs, Fs, fs)
            L = len(cs)
            print(i, j, L)
            c_[:L, i, j] = cs

    save_to_pickle(curr_dir + 'cs_pkl', c_)
    c_ = load_from_pickle(curr_dir + 'cs_pkl')

    print('Done.')

    # Standardization wtr reference microphone
    sigma_target = np.std(c_[:7*fs, r, 0])
    sigma_interf = np.std(c_[:7*fs, r, 1])

    c_[:, :, 0] = c_[:, :, 0] / sigma_target
    c_[:, :, 1] = c_[:, :, 1] / sigma_interf

    # hereafter we assume that the two images have unit-variance at the reference microphone

    # lets add some silence and shift the source such that there is overlap
    cs1 = np.concatenate([np.zeros([2*fs, I]), c_[:, :, 0],
                        np.zeros([4*fs, I]), np.zeros([2*fs, I])], axis=0)
    cs2 = np.concatenate([np.zeros([2*fs, I]), np.zeros([4*fs, I]),
                        c_[:, :, 1], np.zeros([2*fs, I])], axis=0)
    # diffuse noise field simulation given the array geometry

    dn_name = curr_dir + 'diffuse.npy'
    try:
        dn = np.load(dn_name)
    except:
        dn = diffuse_noise(mic_pos, cs1.shape[0], fs, c=343, N=32, mode='sphere').T
        np.save(dn_name, dn)
    assert dn.shape == cs1.shape
    # and unit-variance with respect to the ref mic
    dn = dn / np.std(dn[:, r])


    sigma_n = np.sqrt(10 ** (- snr / 10))
    sigma_i = np.sqrt(10 ** (- sir / 10))

    cs1 = cs1
    cdn = sigma_n * dn
    cs2 = sigma_i * cs2


    # mixing all together
    x = cs1 + cs2 + cdn

    vad = {
        'target': (int(2*fs), int(4.5*fs)),
        'interf': (int(10*fs), int(12.5*fs)),
        'noise':  (int(0.5*fs), int(1.5*fs)),
    }

    x = cs1 + cs2 + cdn

    assert fs == 16000
    nfft = 1024
    hop = 512
    nrfft = nfft+1
    F = nrfft
    fstart = 100  # Hz
    fend = 7500  # Hz
    assert r == ref_mic


    # stft of the spatial images
    CS1 = stft(cs1.T, Fs=Fs, nfft=nfft, hop=hop)[-1]
    CS2 = stft(cs2.T, Fs=Fs, nfft=nfft, hop=hop)[-1]
    CDN = stft(cdn.T, Fs=Fs, nfft=nfft, hop=hop)[-1]
    X = stft(x.T, Fs=Fs, nfft=nfft, hop=hop)[-1]
    CS1 = CS1.transpose([1, 2, 0])
    CS2 = CS2.transpose([1, 2, 0])
    CDN = CDN.transpose([1, 2, 0])
    X = X.transpose([1, 2, 0])
    assert np.allclose(X, CS1+CS2+CDN)

    xin = istft(X[:, :, r], Fs=Fs, nfft=nfft, hop=hop)[-1].real
    cs1in = istft(CS1[:, :, r], Fs=Fs, nfft=nfft, hop=hop)[-1].real
    cs2in = istft(CS2[:, :, r], Fs=Fs, nfft=nfft, hop=hop)[-1].real
    cdnin = istft(CDN[:, :, r], Fs=Fs, nfft=nfft, hop=hop)[-1].real
    assert np.allclose(xin, cs1in + cs2in + cdnin)

    assert Fs == 48000
    assert fs == 16000
    freqs = np.linspace(0, fs//2, F)
    omegas = 2*np.pi*freqs

    print('full measured and synthetic RTF')
    gevdRTF = np.zeros([nrfft, I, J], dtype=np.complex)
    syntRTF = np.zeros_like(gevdRTF)
    rakeRTF = np.zeros_like(gevdRTF)
    for j, src in enumerate(['target', 'interf']):

        Hr = rake_filter(amps[:, r, j], toas[:, r, j], omegas)
        hr = resample(rirs_synt[:, r, j], Fs, fs)

        for i in range(I):

            if i == r:
                gevdRTF[:, r, j] = np.ones(nrfft, dtype=np.complex)
                syntRTF[:, r, j] = np.ones(nrfft, dtype=np.complex)
                rakeRTF[:, r, j] = np.ones(nrfft, dtype=np.complex)

            else:
                # measured RTF
                mi = x[vad[src][0]:vad[src][1], i]
                mr = x[vad[src][0]:vad[src][1], r]
                nd = x[vad['noise'][0]:vad['noise'][1], [r, i]]
                gevdRTF[:, i, j] = estimate_rtf(mi, mr, 'gevdRTF', 'full', Lh=None, n=nd, Fs=fs, nfft=nfft, hop=hop)
                # synthetic RTF
                hi = resample(rirs_synt[:, i, j], Fs, fs)
                syntRTF[:, i, j] = estimate_rtf(hi, hr, 'xcrsRTF', 'full', Lh=None, Fs=fs, nfft=nfft, hop=hop)
                # early closed RTF
                Hi = rake_filter(amps[:, i, j], toas[:, i, j], omegas)
                rakeRTF[:, i, j] = Hi / Hr

    print('... done.')

    # print('gved-based RTF.')
    # dRTF = np.zeros([nrfft, I, J], dtype=np.complex)
    # for j, src in enumerate(['target', 'interf']):
    #     for i in range(I):
    #         if i == r:
    #             dRTF[:, r, j] = np.ones(nrfft)
    #         else:
    #             mi = x[vad[src][0]:vad[src][1], i]
    #             mr = x[vad[src][0]:vad[src][1], r]
    #             nd = x[vad['noise'][0]:vad['noise'][1], [r, i]]
    #             dRTF[:, i, j] = estimate_rtf(mi, mr, 'gevdRTF', 'full', Lh=None, n=nd,
    #                                         Fs=fs, nfft=nfft, hop=hop)
    # print('... done.')

    # print('echo-based RTF:')
    # eRTF = np.zeros([F, I, J], dtype=np.complex)
    # for j in range(J):
    #     for i in range(I):
    #         if i == r:
    #             eRTF[:, r, j] = np.ones(nrfft)
    #         else:
    #             assert len(amps[:, i, j]) == K
    #             assert len(toas[:, i, j]) == K

    #             Hr = rake_filter(amps[:, r, j], toas[:, r, j], omegas)
    #             Hi = rake_filter(amps[:, i, j], toas[:, i, j], omegas)
    #             eRTF[:, i, j] = Hi / Hr
    # print('... done.')


    print('direct-path-based RTF:')
    dpTF = np.zeros([F, I, J], dtype=np.complex)
    for j in range(J):
        for i in range(I):
            Hr = rake_filter(amps[:1, r, j], toas[:1, r, j], omegas)
            if i == r:
                dpTF[:, r, j] = np.ones(nrfft)
            else:
                Hi = rake_filter(amps[:1, i, j], toas[:1, i, j], omegas)
                dpTF[:, i, j] = Hi / Hr
    print('... done.')

    # mix with noise only
    xn = x[vad['noise'][0]:vad['noise'][1], :]
    # mix with target only
    xs = x[vad['target'][0]:vad['target'][1], :]
    # mix with interf only
    xq = x[vad['interf'][0]:vad['interf'][1], :]

    # computed Sigma_n from noise-only
    Sigma_n = np.zeros([F, I, I], dtype=np.complex64)
    XN = stft(xn.T, Fs=Fs, nfft=nfft, hop=hop)[-1]
    XN = XN.transpose([1, 2, 0])
    for f in range(F):
        Sigma_n[f, :, :] = np.cov(XN[f, :, :].T)
    print('Done with noise covariance.')

    # computed Sigma_ni from interf-only
    Sigma_nq = np.zeros([F, I, I], dtype=np.complex64)
    XNQ = stft(xq.T, Fs=Fs, nfft=nfft, hop=hop)[-1]
    XNQ = XNQ.transpose([1, 2, 0])
    for f in range(F):
        Sigma_nq[f, :, :] = np.cov(XNQ[f, :, :].T)
    print('Done with noise covariance.')

    # compute early and late PSD
    PSDs1, PSDr1, PSDl1, COVn = estimates_PSDs_PSDr_from_RTF(
        rakeRTF[:, :, 0],
        xs, xn,
        mic_pos, ref_mic = ref_mic,
        Fs=fs, nrfft=F, hop=hop, fstart=fstart, fend=fend, speed_of_sound=constants['speed_of_sound'])

    assert np.allclose(COVn[200:800, :, :],  Sigma_n[200:800, :, :])

    Sigma_ln = np.zeros_like(Sigma_n)
    Sigma_lnq = np.zeros_like(Sigma_nq)
    for f in range(F):
        Sigma_ln[f, :, :] = Sigma_n[f, :, :] + PSDl1[f, :, :]
        Sigma_lnq[f, :, :] = Sigma_nq[f, :, :] + PSDl1[f, :, :]

    bfs = [
        # DS
        (DS(name='dpDS', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(dpTF[:, :, 0]), dpTF),
        # MVDR
        (MVDR(name='gevdMVDR', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(gevdRTF[:, :, 0], Sigma_n), gevdRTF),
        (MVDR(name='syntMVDR', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(syntRTF[:, :, 0], Sigma_n), syntRTF),
        (MVDR(name='rakeMVDR', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(rakeRTF[:, :, 0], Sigma_n), rakeRTF),
        (MVDR(name='lateMVDR', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(rakeRTF[:, :, 0], Sigma_ln), rakeRTF),
        (MVDR(name='minrMVDR', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(rakeRTF[:, :, 0], Sigma_nq), rakeRTF),
        (MVDR(name='linrMVDR', fstart=fstart, fend=fend, Fs=fs, nrfft=F).compute_weights(rakeRTF[:, :, 0], Sigma_lnq), rakeRTF),

    ]

    results = []

    for (bf, RTF) in bfs:

        print(bf)

        print('TARGET', np.mean(np.abs(bf.enhance(RTF[:, :, 0]))))
        print('INTERF', np.mean(np.abs(bf.enhance(RTF[:, :, 1]))))

        # separation
        Xout = bf.enhance(X.copy())
        CS1out = bf.enhance(CS1.copy())
        CS2out = bf.enhance(CS2.copy())
        CDNout = bf.enhance(CDN.copy())

        xout = istft(Xout, Fs=fs, nfft=nfft, hop=hop)[-1].real
        cs1out = istft(CS1out, Fs=fs, nfft=nfft, hop=hop)[-1].real
        cs2out = istft(CS2out, Fs=fs, nfft=nfft, hop=hop)[-1].real
        cdnout = istft(CDNout, Fs=fs, nfft=nfft, hop=hop)[-1].real

        # metrics
        time = np.arange(2*fs, 9*fs)
        snr_in = snr_dB(cs1in[time], cdnin[time])
        snr_out = snr_dB(cs1out[time], cdnout[time])
        print('SNR', snr_in, '-->', snr_out, ':', snr_out - snr_in)

        sir_in = snr_dB(cs1in[time], cs2in[time])
        sir_out = snr_dB(cs1out[time], cs2out[time])
        print('SIR', sir_in, '-->', sir_out, ':', sir_out - sir_in)

        sdr_in = snr_dB(cs1in[time], xin[time] - cs1in[time])
        sdr_out = snr_dB(cs1out[time], xout[time] - cs1out[time])
        print('SDR', sdr_in, '-->', sdr_out, ':', sdr_out - sdr_in)

        pesq_in = metrics(xin[time], cs1in[time], rate=fs)['pesq'][0]
        pesq_out = metrics(xout[time], cs1in[time], rate=fs)['pesq'][0]
        print('PESQ', pesq_in, '-->', pesq_out, ':', pesq_out - pesq_in)

        prefix = 'data/interim/wav/'
        suffix = '_data-%s_bf-%s' % (data_kind, bf)

        gin = np.abs(np.max(xin))
        gout = np.abs(np.max(xout))


        if render:
            sf.write(curr_dir + prefix + 'x_out' + suffix + '.wav', xout/gout, fs)
            sf.write(curr_dir + prefix + 'cs1_out' + suffix + '.wav', cs1out/gout, fs)
            sf.write(curr_dir + prefix + 'cs2_out' + suffix + '.wav', cs2out/gout, fs)
            sf.write(curr_dir + prefix + 'cdn_out' + suffix + '.wav', cdnout/gout, fs)
            sf.write(curr_dir + prefix + 'x_in' + suffix + '.wav', xin/gin, fs)
            sf.write(curr_dir + prefix + 'cs1_in' + suffix + '.wav', cs1in/gin, fs)
            sf.write(curr_dir + prefix + 'cs2_in' + suffix + '.wav', cs2in/gin, fs)
            sf.write(curr_dir + prefix + 'cdn_in' + suffix + '.wav', cdnin/gin, fs)


        result = {
            'bf' : str(bf),
            'sir_in': sir_in,
            'snr_in': snr_in,
            'sdr_in': sdr_in,
            'sir_out' : sir_out,
            'snr_out': snr_out,
            'sdr_out': sdr_out,
            'pesq_in': pesq_in,
            'pesq_out': pesq_out,
        }
        results.append(result)

    return results



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Run Echo-aware Beamformers')
    parser.add_argument('-a', '--array', help='Which array?', required=True, type=int)
    parser.add_argument('-d', '--data', help='Real or Synthetic?', required=True, type=str)
    parser.add_argument('-D', '--dataset', help='Which dataset? from 0 to 6', required=True, type=int)
    parser.add_argument('-R', '--render', help='render?', required=False, type=bool, default=False)
    parser.add_argument('-N', '--snr', help='SNR input [dB]', required=True, type=int)
    parser.add_argument('-I', '--sir', help='SIR input [dB]', required=True, type=int)


    args = vars(parser.parse_args())

    data = args['data']
    dataset_idx = args['dataset']
    arr_idx = args['array']
    render = args['render']
    snr = args['snr']
    sir = args['sir']

    spk_combs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

    today = date.today()
    result_dir = curr_dir + 'data/interim/'

    results = pd.DataFrame()

    # input('Data are %s\nWanna continue?' % data)

    suffix = 'arr-%d_data-%s_dataset-%d_snr-%s_sir_%d' % (arr_idx, data, dataset_idx, snr, sir)
    results.to_csv(result_dir + '%s_results_%s.csv' % (today, suffix))

    c = 0
    for target_idx in range(4):
        for interf_idx in range(4):
            for s, spk_comb in enumerate(spk_combs):

                if target_idx == interf_idx:
                    continue

                res = main(arr_idx, dataset_idx, target_idx, interf_idx, sir, snr, data, spk_comb, ref_mic=3, render=render)
                1/0

                for res_bf in res:

                    results.at[c, 'data'] = data
                    results.at[c, 'array'] = arr_idx
                    results.at[c, 'dataset'] = dataset_idx
                    results.at[c, 'target_idx'] = target_idx
                    results.at[c, 'interf_idx'] = interf_idx
                    results.at[c, 'sir'] = sir
                    results.at[c, 'snr'] = snr
                    results.at[c, 'spk_comb'] = s
                    results.at[c, 'bf'] = res_bf['bf']
                    results.at[c, 'sir_in'] = res_bf['sir_in']
                    results.at[c, 'snr_in'] = res_bf['snr_in']
                    results.at[c, 'sdr_in'] = res_bf['sdr_in']
                    results.at[c, 'sir_out'] = res_bf['sir_out']
                    results.at[c, 'snr_out'] = res_bf['snr_out']
                    results.at[c, 'sdr_out'] = res_bf['sdr_out']
                    results.at[c, 'pesq_in'] = res_bf['pesq_in']
                    results.at[c, 'pesq_out'] = res_bf['pesq_out']

                    c += 1

                results.to_csv(result_dir + '%s_results_%s.csv' % (today, suffix))
    pass

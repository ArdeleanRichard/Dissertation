import ctypes
import ctypes.util

import matplotlib.pyplot as plt
import numpy as np

import derivatives as deriv


def slt(spikes, ord, ncyc, derivatives=True):

    # superletsDLL = ctypes.WinDLL(r"D:\AC\Anul_4\Licenta\superlets_wrapper.dll")
    superletsDLL = ctypes.WinDLL("D:\\AC\\Anul_4\\Licenta\\superlets_wrapper.dll")

    superletsDLL.asrwt_alloc.restype = ctypes.POINTER(ctypes.c_int)
    superletsDLL.asrwt_alloc.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
                                         ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
    superletsDLL.asrwt_execute.restype = ctypes.c_int
    superletsDLL.asrwt_execute.argtypes = [ctypes.POINTER(ctypes.c_int),
                                           ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    superletsDLL.asrwt_free.restype = None
    superletsDLL.asrwt_free.argtypes = [ctypes.POINTER(ctypes.c_int)]

    c_float_p = ctypes.POINTER(ctypes.c_float)

    result = []
    pos = 0
    # ptr = superletsDLL.asrwt_alloc(len(spikes[0]), 1000, 20, 200, 181, 2, 10, 10, True)

    ptr = superletsDLL.asrwt_alloc(len(spikes[0]), 1000, 1, 250, 250, ncyc, ord, ord, True)

    for spike in spikes:
        # print(pos)

        # spike_arr = np.ndarray.tolist(spike)
        # spike2 = ctypes.cast(spike, ctypes.POINTER(ctypes.c_float))
        output = np.zeros((250, 80))
        output2 = output.astype(np.float32)

        spike2 = (ctypes.c_float * len(spikes[0]))(*spike)  # this is probably correct dont change!!!!
        data_p = output2.ctypes.data_as(c_float_p)

        res = superletsDLL.asrwt_execute(ptr, spike2, data_p)

        # print(res)

        # print(data_p[0])
        output_final = []
        # print(data_p[1])
        for i in np.arange(250):
            output_partial = []
            for j in np.arange(79):
                output_partial.append(data_p[i * 79 + j])
            output_final.append(output_partial)
        result.append(output_final)

        pos = pos + 1
        del data_p
        del output
        del output2

    superletsDLL.asrwt_free(ptr)

    result2 = []
    if derivatives:
        for i in np.arange(len(spikes)):
            # if i == 0:
            #     for j in np.arange(len(result[i])):
            #         first_derivative = deriv.compute_derivative5stencil(result[i][j])
            #         if first_derivative[3]!=0:
            #             plt.plot(np.arange(len(result[i][j])), np.abs(result[i][j]))
            #             plt.plot(np.arange(len(first_derivative)), first_derivative)
            #             plt.show()

            result_partial = deriv.compute_fdmethod(result[i])
            result2.append(np.ndarray.flatten(result_partial))
    else:
        for i in np.arange(len(spikes)):
            result_partial = []

            for j in np.arange(250):
                # result2.append(np.ndarray.flatten(np.array(result[i])))
                result_partial.append(max(result[i][j]))
            max1 = max(result_partial)
            maxpos = result_partial.index(max1)
            fin = []
            fin.append(max1)
            fin.append(maxpos)
            result2.append(np.array(fin))
    print("done")
    return result2


def slt_1spike(spike, spike_pos, label, ord, ncyc, freq_start, freq_end):
    superletsDLL = ctypes.WinDLL(
        "C:\\poli\\year4\\licenta\\codeGoodGit\\superletwrapper\\x64\\Release\\superlets_wrapper.dll")
    superletsDLL.asrwt_alloc.restype = ctypes.POINTER(ctypes.c_int)
    superletsDLL.asrwt_alloc.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
                                         ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
    superletsDLL.asrwt_execute.restype = ctypes.c_int
    superletsDLL.asrwt_execute.argtypes = [ctypes.POINTER(ctypes.c_int),
                                           ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    superletsDLL.asrwt_free.restype = None
    superletsDLL.asrwt_free.argtypes = [ctypes.POINTER(ctypes.c_int)]

    c_float_p = ctypes.POINTER(ctypes.c_float)

    result = []
    pos = 0
    ptr = superletsDLL.asrwt_alloc(len(spike), 1000, freq_start, freq_end, freq_end - freq_start + 1, ncyc, ord, ord,
                                   True)
    output = np.zeros((freq_end - freq_start + 1, 80))
    output2 = output.astype(np.float32)

    spike2 = (ctypes.c_float * len(spike))(*spike)  # this is probably correct dont change!!!!
    data_p = output2.ctypes.data_as(c_float_p)

    res = superletsDLL.asrwt_execute(ptr, spike2, data_p)

    output_final = []
    for i in np.arange(freq_end - freq_start + 1):
        output_partial = []
        for j in np.arange(79):
            output_partial.append(data_p[i * 79 + j])
        output_final.append(output_partial)
    result.append(output_final)

    superletsDLL.asrwt_free(ptr)
    values = np.abs(output_final)
    scales_freq = np.arange(freq_start, freq_end)
    df = scales_freq[-1] / scales_freq[-2]
    ymesh = np.concatenate([scales_freq, [scales_freq[-1] * df]])
    im = plt.pcolormesh(np.arange(len(spike)), ymesh, values, cmap="jet")
    plt.colorbar(im)
    plt.title("Sim" + str(2) + "_spike" + str(spike_pos) + "label" + str(label) + "_superlets")
    plt.xlabel("Time")
    plt.ylabel("Frequency - linear")
    plt.savefig("Sim" + str(2) + "_spike" + str(spike_pos) + "label" + str(label) + "_superlets")
    plt.show()

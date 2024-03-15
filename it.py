import numpy as np
from scipy.stats import mode
from typing import List
from tqdm import tqdm

import matplotlib.pyplot as plt

from jpype import *



## DATA-TYPE CHECKS

def setup_JArray(a: np.ndarray) -> JArray:
    
    a = (a).astype(np.float64) 

    try:
        ja = JArray(JDouble, 1)(a)
        
    except Exception: 
        ja = JArray(JDouble, 1)(a.tolist())

    return ja
        
def check_eeg_data(data: np.ndarray, epoch_avg: bool) -> np.ndarray:

    if not isinstance(data, np.ndarray):
        raise ValueError("EEG data must be a numpy array.")
    
    if data.ndim == 3 and epoch_avg:  # Average across epochs to obtain shape (n_chan, time_points)
        raise ValueError(f"epoch_avg is set to True but EEG signals passed have {data.ndim} dimensions. Expected 2 dimensions with shape (n_epo, n_timepoints)")
    elif data.ndim > 3:
        raise ValueError(f"The EEG signals passed do not have the correct shape. Expected 2 dimensions (n_chan, time_points) or 3 dimensions (n_epochs, n_chan, time_points); instead, received {data.ndim}.")
    return data



## MUTUAL INFORMATION 

def mi_ksg(s1: np.ndarray, s2: np.ndarray, epoch_avg: bool, version: int = 1, kraskov_param: int = 4) -> float:
    """Calculate mutual information between two time series using Kraskov-Stögbauer-Grassberger estimator 1 or 2

    Args:
        s1                  (np.ndarray): EEG time series 1
        s2                  (np.ndarray): EEG time series 2
        epoch_avg                 (bool): Whether the data has *already* been averaged across epochs 
        version          (int, optional): Which estimator version to use. Defaults to 1.
        kraskov_param    (int, optional): KSG parameter. Defaults to 4.

    Returns:
                                 (float): KSG mutual information between s1 and s2.
    """
    n_epo = 1 if epoch_avg else s1.shape[0]
    result = 0

    miCalcClass = JPackage("infodynamics.measures.continuous.kraskov")\
                  .MutualInfoCalculatorMultiVariateKraskov1 if version == 1 else \
                  JPackage("infodynamics.measures.continuous.kraskov")\
                  .MutualInfoCalculatorMultiVariateKraskov2
    
    miCalc = miCalcClass()
    miCalc.setProperty("k", str(kraskov_param))

    for epo_i in range(n_epo):

        X, Y = (s1, s2) if epoch_avg else (s1[epo_i, :], s2[epo_i, :])

        sig1 = setup_JArray(X)
        sig2 = setup_JArray(Y)

        miCalc.initialise()
        miCalc.setObservations(sig1, sig2)
        result += miCalc.computeAverageLocalOfObservations()

    return result / n_epo

def mi_kernel(s1: np.ndarray, s2: np.ndarray, epoch_avg: bool, kernel_width: float = 0.25) -> float:
    """Calculate mutual information between two time series using a kernel estimator
    
    Args:
        s1                  (np.ndarray): EEG time series 1
        s2                  (np.ndarray): EEG time series 2
        epoch_avg                 (bool): Whether the data has *already* been averaged across epochs
        kernel_width   (float, optional): Kernel width to use. Defaults to 0.25.

    Returns:
                                 (float): Kernel mutual information between s1 and s2.
    """
    n_epo = 1 if epoch_avg else s1.shape[0]
    result = 0

    miCalcClass = JPackage("infodynamics.measures.continuous.gaussian").MutualInfoCalculatorMultiVariateGaussian
    miCalc = miCalcClass()
    miCalc.setProperty("KERNEL_WIDTH", str(kernel_width))

    for epo_i in range(n_epo):

        X, Y = (s1, s2) if epoch_avg else (s1[epo_i, :], s2[epo_i, :])

        sig1 = setup_JArray(X)
        sig2 = setup_JArray(Y)

        miCalc.initialise()
        miCalc.setObservations(sig1, sig2)
        result += miCalc.computeAverageLocalOfObservations()

    return result / n_epo

def mi_gaussian(s1: np.ndarray, s2: np.ndarray, epoch_avg: bool) -> float:
    """Calculate mutual information between two time series using Gaussian estimator
    
    Args:
        s1            (np.ndarray): EEG time series 1
        s2            (np.ndarray): EEG time series 2
        epoch_avg           (bool): Whether the data has *already* been averaged across epochs

    Returns:
                           (float): Gaussian mutual information between s1 and s2.
    """
    n_epo = 1 if epoch_avg else s1.shape[0]
    result = 0

    miCalcClass = JPackage("infodynamics.measures.continuous.gaussian").MutualInfoCalculatorMultiVariateGaussian
    miCalc = miCalcClass()

    for epo_i in range(n_epo):

        X, Y = (s1, s2) if epoch_avg else (s1[epo_i, :], s2[epo_i, :])

        sig1 = setup_JArray(X)
        sig2 = setup_JArray(Y)
        
        miCalc.initialise()
        miCalc.setObservations(sig1, sig2)
        result += miCalc.computeAverageLocalOfObservations()

    return result / n_epo


def entropy_symb(s1: np.ndarray, s2: np.ndarray, epoch_avg: bool, l: int, m: int) -> np.ndarray:
    """Calculate the symbolic Shannon entropy of two given EEG signals

    Args:
        s1            (np.ndarray): EEG signal 1
        s2            (np.ndarray): EEG signal 2
        epoch_avg             (bool): Whether the EEG data has *already* been averaged across epochs.
        l            (int, optional): Lag step
        m            (int, optional): Embedding dimension

    Returns:
        np.ndarray: An array of Shannon entropic measures ([H(X), H(Y), H(X;Y)])
    """

    def symb_symbolise(X: np.ndarray, l: int, m: int):
        Y = np.empty((m, len(X) - (m - 1) * l))
        for i in range(m):
            Y[i] = X[i * l:i * l + Y.shape[1]]
        return Y.T
        
    def symb_incr_counts(key,d):
        d[key] = d.get(key, 0) + 1

    def symb_normalise(d):
        s=sum(d.values())        
        for key in d:
            d[key] /= s



    n_epo = 1 if epoch_avg else s1.shape[0]
    entropy_X, entropy_Y, entropy_XY = 0, 0, 0

    hashmult = np.power(m, np.arange(m))
    
    for epo_i in range(n_epo):

        X, Y = (s1, s2) if epoch_avg else (s1[epo_i, :], s2[epo_i, :])

        X = symb_symbolise(X, l, m).argsort(kind='quicksort')
        Y = symb_symbolise(Y, l, m).argsort(kind='quicksort')

        hashval_X = (np.multiply(X, hashmult)).sum(1) # multiply each symbol [1,0,3] by hashmult [1,3,9] => [1,0,27] and give a final array of the sum of each code ([.., .., 28, .. ])
        hashval_Y = (np.multiply(Y, hashmult)).sum(1)
        
        x_sym_to_perm = hashval_X
        y_sym_to_perm = hashval_Y
        
        p_xy = {}
        p_x = {}
        p_y = {}
        
        for i in range(len(x_sym_to_perm)-1):
            xy = str(x_sym_to_perm[i]) + "," + str(y_sym_to_perm[i])
            x = str(x_sym_to_perm[i])
            y = str(y_sym_to_perm[i])
            symb_incr_counts(xy,p_xy)
            symb_incr_counts(x,p_x)
            symb_incr_counts(y,p_y)
            
        symb_normalise(p_xy)
        symb_normalise(p_x)
        symb_normalise(p_y)
        
        p_xy = np.array(list(p_xy.values()))
        p_x = np.array(list(p_x.values()))
        p_y = np.array(list(p_y.values()))
        
        entropy_X += -np.sum(p_x * np.log2(p_x + np.finfo(float).eps)) 
        entropy_Y += -np.sum(p_y * np.log2(p_y + np.finfo(float).eps))
        entropy_XY += -np.sum(p_xy * np.log2(p_xy + np.finfo(float).eps))

    # If multiple epochs, calculates average entropy of a channel across epochs by summating entropy at each epoch and dividing by total no. of epochs. 
    # If incoming signals are already averaged across epochs (and therefore n_epo = 1), returns the entropy of that channel.
    avg_epo_entropies = np.array([entropy_X / n_epo, entropy_Y / n_epo, entropy_XY / n_epo])

    return avg_epo_entropies

def mi_symbolic(s1: np.ndarray, s2: np.ndarray, epoch_avg: bool, l: int = 1, m: int = 3) -> float:
    """Calculate symbolic mutual information between two time series

    Args:
        s1              (np.ndarray): EEG time series 1
        s2              (np.ndarray): EEG time series 2
        epoch_avg             (bool): Whether the data has *already* been averaged across epochs
        l            (int, optional): Lag step. Defaults to 1.
        m            (int, optional): Embedding dimension. Defaults to 3.

    Returns:
                             (float): Symbolic mutual information between s1 and s2.
    """
    entropies = entropy_symb(s1, s2, epoch_avg, l, m) # entropy_symb returns [n_ch, 3] for H(X), H(Y), and H(X,Y)
    return entropies[0] + entropies[1] - entropies[2] 



def compute_mi(eeg_1: np.ndarray, eeg_2: np.ndarray = None, epoch_avg: bool = True, mode: str = "kernel",  **kwargs) -> np.ndarray:
    """Main function to compute mutual information between all EEG channel combinations, whether intra- or inter-brain. 
       Incoming data can be epoched or already epoch-averaged. 
       Different estimators (mode) available:
        
         -  Kraskov-Stögbauer-Grassberger (KSG; 1 or 2)
         -  Kernel 
         -  Gaussian
         -  Symbolic

    Args:
        eeg_1                     (np.ndarray): Participant (1) EEG data
        eeg_2           (np.ndarray, optional): Participant 2 EEG data . Defaults to None.
        epoch_avg             (bool, optional): Whether the data has *already* been averaged across epochs. Defaults to True.
        mode                   (str, optional): Which estimator type to compute MI with. Defaults to "kernel".

    Returns:
        mi_matrix                 (np.ndarray): A mutual information matrix of all channel combinations. Note that the intra-brain MI will be symmetric.
    """


    inter_brain = eeg_2 is not None

    signal1 = check_eeg_data(eeg_1, epoch_avg)
    signal2 = check_eeg_data(eeg_2, epoch_avg) if inter_brain else signal1

    n_chan = signal1.shape[0 if epoch_avg else 1]
    

    mi_estimation_methods = {
        "kraskov": mi_ksg,
        "kernel": mi_kernel,
        "gaussian": mi_gaussian,
        "symbolic": mi_symbolic
    }

    if mode not in mi_estimation_methods:
        raise ValueError(f"Unsupported mode '{mode}'. Supported modes are: {list(mi_estimation_methods.keys())}.")
    
    mi_func = mi_estimation_methods[mode]

    mi_matrix = np.zeros((n_chan, n_chan))

    for i in tqdm(range(n_chan)):
        start_j = 0 if inter_brain else i
        for j in range(start_j, n_chan):
            if inter_brain or i != j:
                s1, s2 = (signal1[i, :], signal2[j, :]) if epoch_avg else (signal1[:, i, :], signal2[:, j, :]) # whether to keep epochs
                mi_matrix[i, j] = mi_func(s1, s2, epoch_avg, **kwargs)
                if not inter_brain:
                    mi_matrix[j, i] = mi_matrix[i, j] # or 0 if you want to avoid symmetry

    return mi_matrix



## TRANSFER ENTROPY

def te_ksg(s1: np.ndarray, s2: np.ndarray, epoch_avg: bool, optimise: bool = False, k: int = 1, k_tau: int = 1, l: int = 1, l_tau: int = 1, delay: int = 1, kraskov_param: int = 4) ->  float:
    """Calculates transfer entropy between 2 time series using Kraskov-Stögbauer-Grassberger (KSG) Estimator 

    Args:
        s1                (np.ndarray): EEG time series 1 (SOURCE)
        s2                (np.ndarray): EEG time series 2 (TARGET)
        epoch_avg               (bool): Whether the data has *already* been averaged across epochs
        k              (int, optional): TARGET history embedding length (i.e., length of past to consider). Defaults to 1.
        k_tau          (int, optional): TARGET history embedding delay (i.e., applied between elements of embedding vector). Defaults to 1.
        l              (int, optional): SOURCE history embedding length. Defaults to 1.
        l_tau          (int, optional): SOURCE history embedding delay. Defaults to 1.
        delay          (int, optional): Delay from SOURCE to TARGET. Defaults to 1.
        kraskov_param  (int, optional): Kraskov parameter for number of nearest searches. Defaults to 4.

    Returns:
                               (float): KSG TE estimation (s1->s2)
    """

    def te_kraskov_find_optimal_parameters(eeg_1: np.ndarray, eeg_2: np.ndarray = None, epoch_avg: bool = True) -> np.array:
        """Find the optimal k, k_tau, l, l_tau, and delay values for Kraskov Transfor Entropy estimation using the Ragwitz method

        Args:
            eeg1            (np.ndarray): EEG time series 1 (SOURCE).
            eeg2  (np.ndarray, optional): EEG time series 2 (TARGET). Defaults to None.

        Returns:
                            (np.array): List of optimal k, k_tau, l, l_tau, and delay values based on their respective statistical mode
        """
        
        # Do not need all epochs, so take first epoch if eeg_1 has not been epoch averaged. 
        sig1 = check_eeg_data(eeg_1, epoch_avg)[0] if not epoch_avg else check_eeg_data(eeg_1, epoch_avg) 
        n_chan = sig1.shape[0]
        inter_brain = eeg_2 is not None

        if inter_brain:
            sig2 = check_eeg_data(eeg_2, epoch_avg)[0] if not epoch_avg else check_eeg_data(eeg_2, epoch_avg) 
        else:
            sig2 = sig1

        print(sig1.shape)
        print(sig2.shape)

        K_values, KTau_values, L_values, LTau_values, delay_values = [], [], [], [], []
        teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
        teCalc = teCalcClass()
        teCalc.setProperty(teCalcClass.PROP_AUTO_EMBED_METHOD,
                        teCalcClass.AUTO_EMBED_METHOD_RAGWITZ)
        teCalc.setProperty(teCalcClass.PROP_K_SEARCH_MAX, "6")
        teCalc.setProperty(teCalcClass.PROP_TAU_SEARCH_MAX, "6")
        teCalc.initialise()

        for i in tqdm(range(n_chan)):
            for j in range(n_chan):
                if inter_brain or i != j:

                    X, Y = sig1[i, :], sig2[j, :]
                    s1 = setup_JArray(X)
                    s2 = setup_JArray(Y)

                    teCalc.setObservations(s1, s2)

                    K_values.append(int(str(teCalc.getProperty(teCalcClass.K_PROP_NAME))))
                    KTau_values.append(int(str(teCalc.getProperty(teCalcClass.K_TAU_PROP_NAME))))
                    L_values.append(int(str(teCalc.getProperty(teCalcClass.L_PROP_NAME))))
                    LTau_values.append(int(str(teCalc.getProperty(teCalcClass.L_TAU_PROP_NAME))))
                    delay_values.append(int(str(teCalc.getProperty(teCalcClass.DELAY_PROP_NAME))))


        def get_mode_value(modes_result):
            if np.isscalar(modes_result.mode):
                return modes_result.mode
            else:
                return modes_result.mode[0]

        return np.array((get_mode_value(mode(K_values)),
                        get_mode_value(mode(KTau_values)),
                        get_mode_value(mode(L_values)),
                        get_mode_value(mode(LTau_values)),
                        get_mode_value(mode(delay_values)))) 


    if optimise:
        k, k_tau, l, l_tau, delay = te_kraskov_find_optimal_parameters(s1, s2, epoch_avg)

    n_epo = 1 if epoch_avg else s1.shape[0]
    result = 0

    teCalcClass = JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
    teCalc = teCalcClass()
    teCalc.setProperty("k_HISTORY", str(k))
    teCalc.setProperty("k_TAU", str(k_tau))
    teCalc.setProperty("l_HISTORY", str(l))
    teCalc.setProperty("l_TAU", str(l_tau))
    teCalc.setProperty("DELAY", str(delay))
    teCalc.setProperty("k", str(kraskov_param))
    teCalc.initialise() 

    for epo_i in range(n_epo):

        X, Y = (s1, s2) if epoch_avg else (s1[epo_i, :], s2[epo_i, :])

        sig1 = setup_JArray(X)
        sig2 = setup_JArray(Y)
        
        teCalc.setObservations(sig1, sig2)

        result += teCalc.computeAverageLocalOfObservations()

    return result / n_epo

def te_kernel(s1: np.ndarray, s2: np.ndarray, epoch_avg: bool, k: int = 1, kernel_width: float = 0.5) -> float:
    """Calculates transfer entropy between 2 time series using kernel estimator

    Args:
        s1                   (np.ndarray): EEG time series 1 (SOURCE)
        s2                   (np.ndarray): EEG time series 2 (TARGET)
        epoch_avg                  (bool): Whether the data has *already* been averaged across epochs
        k                 (int, optional): TARGET history embedding length, see Schreiber (2000). Defaults to 1. 
        kernel_width    (float, optional): Kernel width of normalised units. Defaults to 0.5.

    Returns:
                                  (float): Kernel TE estimation (s1->s2) 
    """

    n_epo = 1 if epoch_avg else s1.shape[0]
    result = 0

    teCalcClass = JPackage("infodynamics.measures.continuous.kernel").TransferEntropyCalculatorKernel
    teCalc = teCalcClass()
    teCalc.setProperty("NORMALISE", "true") 
    teCalc.initialise(k, kernel_width) 

    for epo_i in range(n_epo):

        X, Y = (s1, s2) if epoch_avg else (s1[epo_i, :], s2[epo_i, :])

        sig1 = setup_JArray(X)
        sig2 = setup_JArray(Y)
    
        teCalc.setObservations(sig1, sig2)

        result += teCalc.computeAverageLocalOfObservations()
    
    return result / n_epo

def te_gaussian(s1: np.ndarray, s2: np.ndarray, epoch_avg: bool, k: int = 1, k_tau: int = 1, l: int = 1, l_tau: int = 1, delay: int = 1, bias_correction: bool = False) -> float:
    """Calculates transfer entropy between 2 time series using Gaussian estimator

    Args:
        s1                     (np.ndarray): EEG time series 1 (SOURCE)
        s2                     (np.ndarray): EEG time series 2 (TARGET)
        epoch_avg                    (bool): Whether the data has *already* been averaged across epochs
        k                   (int, optional): TARGET history embedding length (i.e., length of past to consider). Defaults to 1.
        k_tau               (int, optional): TARGET history embedding delay (i.e., applied between elements of embedding vector). Defaults to 1.
        l                   (int, optional): SOURCE history embedding length. Defaults to 1.
        l_tau               (int, optional): SOURCE history embedding delay. Defaults to 1.
        delay               (int, optional): Delay from SOURCE to TARGET. Defaults to 1.
        bias_correction    (bool, optional): _description_. Defaults to False.

    Returns:
                                    (float): Gaussian TE estimation (s1->s2)
    """
    n_epo = 1 if epoch_avg else s1.shape[0]
    result = 0

    teCalcClass = JPackage("infodynamics.measures.continuous.gaussian").TransferEntropyCalculatorGaussian
    teCalc = teCalcClass()
    teCalc.setProperty("k_HISTORY", str(k))
    teCalc.setProperty("k_TAU", str(k_tau))
    teCalc.setProperty("l_HISTORY", str(l))
    teCalc.setProperty("l_TAU", str(l_tau))
    teCalc.setProperty("DELAY", str(delay))
    teCalc.setProperty("BIAS_CORRECTION", str(bias_correction).lower())
    teCalc.initialise()

    for epo_i in range(n_epo):

        X, Y = (s1, s2) if epoch_avg else (s1[epo_i, :], s2[epo_i, :])

        sig1 = setup_JArray(X)
        sig2 = setup_JArray(Y)

        teCalc.setObservations(sig1, sig2)
        result += teCalc.computeAverageLocalOfObservations()

    return result / n_epo

def te_symbolic(s1: np.ndarray, s2: np.ndarray, epoch_avg: bool, k: int = 1) -> float:
    """Calculates Transfer Entropy between 2 Univariate Time Series using Symbolic Estimation (Staniek & Lehrnertz)

    Args:
        s1             (np.ndarray): EEG time series 1 (SOURCE)
        s2             (np.ndarray): EEG time series 2 (TARGET)
        epoch_avg            (bool): Whether the data has *already* been averaged across epochs
        k           (int, optional): TARGET history embedding length. Defaults to 1.

    Returns:
        result       (float): Symbolic TE estimation (s1->s2)
    """

    sig1 = setup_array(s1)
    sig2 = setup_array(s2)

    teCalcClass = JPackage("infodynamics.measures.continuous.symbolic").TransferEntropyCalculatorSymbolic
    teCalc = teCalcClass()
    teCalc.setProperty("k_HISTORY", str(k))
    teCalc.initialise(2) # base = 2
    teCalc.setObservations(sig1, sig2)
    result = teCalc.computeAverageLocalOfObservations()
    return result



def compute_te(eeg_1: np.ndarray, eeg_2: np.ndarray = None, epoch_avg: bool = True, mode: str = "kernel", **kwargs) -> np.ndarray:
    """Main function to compute transfer entropy between all EEG channel combinations, whether intra- or inter-brain. 
       Incoming data can be epoched or already epoch-averaged. 
       Different estimators (mode) available:
        
         -  Kraskov-Stögbauer-Grassberger (KSG)
         -  Kernel 
         -  Gaussian
         -  Symbolic
         
    Args:
        eeg_1                     (np.ndarray): Participant (1) EEG data 
        eeg_2           (np.ndarray, optional): Participant 2 EEG data. Defaults to None.
        epoch_avg             (bool, optional): Whether the data has *already* been averaged across epochs. Defaults to True.
        mode                   (str, optional): Which estimator type to compute MI with. Defaults to "kernel".

    Returns:
        te_matrix                 (np.ndarray): A transfer entropy matrix of all channel combinations
    """

    inter_brain = eeg_2 is not None

    signal1 = check_eeg_data(eeg_1, epoch_avg)
    signal2 = check_eeg_data(eeg_2, epoch_avg) if inter_brain else signal1

    n_chan = signal1.shape[0 if epoch_avg else 1]

    te_estimation_methods = {
        "kraskov": te_ksg,
        "kernel": te_kernel,
        "gaussian": te_gaussian,
        "symbolic": te_symbolic
    }

    if mode not in te_estimation_methods:
        raise ValueError(f"Unsupported mode '{mode}'. Supported modes are: {list(te_estimation_methods.keys())}.")
    
    te_func = te_estimation_methods[mode]

    te_matrix_xy = np.zeros((n_chan, n_chan))
    te_matrix_yx = np.zeros((n_chan, n_chan))

    for i in tqdm(range(n_chan)):
        for j in range(n_chan):
            if inter_brain or i != j: # avoid self-channel calculations for intra_brain condition
                s1, s2 = (signal1[i, :], signal2[j, :]) if epoch_avg else (signal1[:, i, :], signal2[:, j, :]) # whether to keep epochs
            
                te_matrix_xy[i, j] = te_func(s1, s2, epoch_avg, **kwargs)
                te_matrix_yx[i, j] = te_func(s2, s1, epoch_avg, **kwargs)
       

    return te_matrix_xy, te_matrix_yx




## VISUALISATION

def plot_it(it_matrix: np.ndarray, inter_brain: bool, channel_names: List[str]):
    """Plots heatmap of mutual information or transfer entropy values for either intra-brain or inter-brain design.

    Args:
        it_matrix (np.ndarray): Matrix with shape (n_chan1, n_chan2). Can be same channels (intra-brain) or two-person channel (inter-brain). Note that intrabrain MI will be a symmetric heatmap. 
        inter_brain (bool): Whether the analysis is inter-brain analysis or not (i.e., intra-brain)
        channel_names (List[str]): List of channel names of EEG signals. Either a single list for intra-brain or two lists for inter-brain. 
    """

    if channel_names.ndim == 1 and not inter_brain:
        channel_names = [channel_names, channel_names]


    plt.figure(figsize=(10, 8))
    plt.matshow(it_matrix, fignum=1, cmap='viridis')  
    plt.colorbar()

    if inter_brain: 
        plt.title('Inter-Brain', pad=20)
        plt.xlabel('Participant 2 Channels (Target)')
        plt.ylabel('Participant 1 Channels (Source)')
    
    else: 
        plt.title('Intra-Brain', pad=20)
        plt.xlabel('Target Channels')
        plt.ylabel('Source Channels')

    plt.xticks(range(it_matrix.shape[0]), channel_names[1], rotation=90) 
    plt.yticks(range(it_matrix.shape[0]), channel_names[0])
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=True)
    plt.show()

    highest = np.max(it_matrix)
    channel_pair_with_highest = np.unravel_index(np.argmax(it_matrix), it_matrix.shape)
    print(f"Strongest regions: (Source Channel {channel_names[0][channel_pair_with_highest[0]]} --> " +
                             f" Target Channel {channel_names[1][channel_pair_with_highest[1]]}) = {highest}")


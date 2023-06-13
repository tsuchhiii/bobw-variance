import matplotlib.pyplot as plt

def setup_plotting_params():
    '''
    plot setting
    '''
    # matplotlib.font_manager._rebuild()
    # for avoiding type 3 fonts and change to type 1?
    # https://www.yuukinishiyama.com/2020/08/19/matplotlib_pdf_latex/
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath']
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = 'Helvetica'

    # =====

    # plt.close()
    plt.rcParams['font.family'] = 'Times New Roman'  # font familyの設定
    plt.rcParams['mathtext.fontset'] = 'stix'  # math fontの設定
    # plt.rcParams["font.size"] = 11  # 全体のフォントサイズが変更されます。
    # plt.rcParams["font.size"] = 13  # 全体のフォントサイズが変更されます。  # for AISTATS2023 submission
    plt.rcParams["font.size"] = 16 # 全体のフォントサイズが変更されます。  # for AISTATS2023 submission
    # plt.rcParams['xtick.labelsize'] = 9  # 軸だけ変更されます。
    # plt.rcParams['ytick.labelsize'] = 24  # 軸だけ変更されます
    # plt.rcParams['figure.figsize'] = (5., 5.)  # figure size in inch, 横×縦
    plt.rcParams['figure.figsize'] = (3.5, 3.5)  # figure size in inch, 横×縦
    #  todo: remove for now
    # plt.rcParams['figure.dpi'] = 300  # over 300 is good for paper
    plt.rcParams['figure.dpi'] = 100  # for preprint


## Further Adaptive Best-of-Both-Worlds Algorithm for Combinatorial Semi-Bandits
Public code for a paper
["Further Adaptive Best-of-Both-Worlds Algorithm for Combinatorial Semi-Bandits"](https://proceedings.mlr.press/v206/tsuchiya23a.html) in AISTATS2023.
All algorithms included are the followings:
```
- CombUCB1  (Kveton+ 2015)
- Thompson sampling (for multiple selection) (Wang and Chen 2018)
- HYBRID (Zimmert+ 2018)
- LBINF (Ito 2021)
- LBINFV-LS (This Work)
- LBINFV-GD (This Work)
```
(Remark. The codes are not refactored very well.)

### Environment
The following provides three ways for constructing the environment.

(1). If you use `pyenv`, just run `setup.sh` (it uses venv and automatically install the required packages) 
To setup the environment, use `Python 3.7` or a more up-to-date version.
```
bash setup.sh
source venv/bin/activate
```

(2) If you are using `conda` environment:
``` 
conda create -n bandit python=3.8
conda activate bandit
pip install -r requirements.txt
mkdir -p results/pe-mab/fig
```

Then, run
```bash
pip install -r requirements.txt 
```

(3) Otherwise, install the following packages with `Python 3.n (n >= 7)` environment.
```
- numpy
- pandas
- matplotlib
- seaborn
- tqdm
```

For (2) and (3), for generating the required directory for replicating the results, need to run the following:

```
mkdir -p results/multiple/fig
```

One can setup what algorithms to use in `config/config_multiple.py`


### How to Run (by shell script)
```
$ bash run_multiple.py {number of base-arms}

$ bash run_multiple.py 5     # synthetic data
$ bash run_multiple_realdata.sh    # real data

# and then plot 
$ bash plot_multiple_aistats.sh

# don't forget to run
$ mkdir -p results/multiple/fig
```

### How to Run (run LBINF-V (proposed algorithm) and CombUCB1)
```
$ python algorithms/multiple_play/test_multiple.py -ho {number of horizon} -na {number of actions} -nt {number of top arms (= number of maximum chosen arm)} -a {algorithm name} -ad {arm distribution} -s {setting} 

$ python algorithms/multiple_play/test_multiple.py -ho 10000 -na 5 -nt 2 -a LogBarrierINF-V-Semi-LS -ad Ber -s 1 
$ python algorithms/multiple_play/test_multiple.py -ho 10000 -na 5 -nt 2 -a LogBarrierINF-V-Semi-LS -ad Ber -s 2 
$ python algorithms/multiple_play/test_multiple.py -ho 10000 -na 5 -nt 2 -a LogBarrierINF-V-Semi-LS -ad SCA-Ber -s 1  

$ python algorithms/multiple_play/test_multiple.py -ho 10000 -na 5 -nt 2 -a CombUCB1 -ad Ber -s 1 
$ python algorithms/multiple_play/test_multiple.py -ho 10000 -na 5 -nt 2 -a CombUCB1 -ad Ber -s 2 
$ python algorithms/multiple_play/test_multiple.py -ho 10000 -na 5 -nt 2 -a CombUCB1 -ad SCA-Ber -s 1  
```
You can plot the results by
```
$ python plot_multiple.py -na {# of arms} -m multiple -ad {arm distribution} -s {problem setup} -ho {# of horizon}

# fix config/config_multiple.py if required
$ python plot_multiple.py -na 5 -m multiple -ad Ber -s 1 -ho 10000
$ python plot_multiple.py -na 5 -m multiple -ad Ber -s 2 -ho 10000
$ python plot_multiple.py -na 5 -m multiple -ad SCA-Ber -s 1 -ho 10000

```

### References
- B. Kveton, Z. Wen, A. Ashkan, and C. Szepesvari. Tight Regret Bounds for Stochastic Combinatorial Semi-Bandits. In Proceedings of the Eighteenth International Conference on Artificial Intelligence and Statistics, volume 38, pages 535–543, 2015.
- J. Zimmert, H. Luo, and C.Y. Wei. Beating stochastic and adversarial semi-bandits optimally and simultaneously. In Proceedings of the 36th International Conference on Machine Learning, volume 97, pages 7683–7692, 2019.
  (The code to sample an action from given an output of FTRL is taken from [here](https://github.com/diku-dk/CombSemiBandits).)
- S. Ito. Hybrid regret bounds for combinatorial semi-bandits and adversarial linear bandits. In Advances in Neural Information Processing Systems, volume 34, pages 2654–2667, 2021a.
- S. Wang and W. Chen. Thompson sampling for combinatorial semi-bandits. In Proceedings of the 35th International Conference on Machine Learning, volume 80, pages 5114–5122, 2018.


### Citation
```
@InProceedings{TIH2023further,
  title = 	 {Further Adaptive Best-of-Both-Worlds Algorithm for Combinatorial Semi-Bandits},
  author =       {Tsuchiya, Taira and Ito, Shinji and Honda, Junya},
  booktitle = 	 {Proceedings of The 26th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {8117--8144},
  year = 	 {2023},
  editor = 	 {Ruiz, Francisco and Dy, Jennifer and van de Meent, Jan-Willem},
  volume = 	 {206},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {25--27 Apr},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v206/tsuchiya23a/tsuchiya23a.pdf},
  url = 	 {https://proceedings.mlr.press/v206/tsuchiya23a.html},
  abstract = 	 {We consider the combinatorial semi-bandit problem and present a new algorithm with a best-of-both-worlds regret guarantee; the regrets are bounded near-optimally in the stochastic and adversarial regimes. In the stochastic regime, we prove a variance-dependent regret bound depending on the tight suboptimality gap introduced by Kveton et al. (2015) with a good leading constant. In the adversarial regime, we show that the same algorithm simultaneously obtains various data-dependent regret bounds. Our algorithm is based on the follow-the-regularized-leader framework with a refined regularizer and adaptive learning rate. Finally, we numerically test the proposed algorithm and confirm its superior or competitive performance over existing algorithms, including Thompson sampling under most settings.}
}
```

### Contact
If you have any problems or questions when running codes, please send an email to the authors.



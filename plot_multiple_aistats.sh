# synthetic data, T = 10^4
python plot_multiple.py -na 5 -m multiple -ad Ber -s 1 -ho 10000
python plot_multiple.py -na 5 -m multiple -ad Ber -s 2 -ho 10000
python plot_multiple.py -na 5 -m multiple -ad SCA-Ber -s 1 -ho 10000

# semi-synthetic data
python plot_multiple.py -na 6 -m multiple -ad Ber -s 5 -ho 2000
python plot_multiple.py -na 8 -m multiple -ad Ber -s 7 -ho 2000
python plot_multiple.py -na 10 -m multiple -ad Ber -s 8 -ho 2000

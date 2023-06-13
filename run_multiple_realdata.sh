# if [ $# -ne 1 ]; then
#     echo specify n_arms
#     exit 1
# fi

algo_array=(CombUCB1 MPTS SemiLBINF LogBarrierINF-V-Semi-LS LogBarrierINF-V-Semi-GD-0.25)
for algo in "${algo_array[@]}"
    do
        echo python algorithms/multiple_play/test_multiple.py -na 5 -nt 3 -a $algo -ad Ber -s 4  
        python algorithms/multiple_play/test_multiple.py -ho 10000 -na 6 -nt 3 -a $algo -ad Ber -s 5 &
        python algorithms/multiple_play/test_multiple.py -ho 10000 -na 8 -nt 3 -a $algo -ad Ber -s 7 &
        python algorithms/multiple_play/test_multiple.py -ho 10000 -na 10 -nt 3 -a $algo -ad Ber -s 8
    done
exit 0


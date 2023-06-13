if [ $# -ne 1 ]; then
    echo specify n_arms
    exit 1
fi

algo_array=(CombUCB1 MPTS HYBRID SemiLBINF LogBarrierINF-V-Semi-LS LogBarrierINF-V-Semi-GD-0.25)

for algo in "${algo_array[@]}"
    do
        echo python algorithms/multiple_play/test_multiple.py -na $1 -s 1 -a $algo -ad Ber
        python algorithms/multiple_play/test_multiple.py -ho 10000 -na $1 -nt 2 -a $algo -ad Ber -s 1 &  
        python algorithms/multiple_play/test_multiple.py -ho 10000 -na $1 -nt 2 -a $algo -ad Ber -s 2 &
        python algorithms/multiple_play/test_multiple.py -ho 10000 -na $1 -nt 2 -a $algo -ad SCA-Ber -s 1  
    done


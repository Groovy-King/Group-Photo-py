for i in {1..9}
    for j in {1..3}
        do
        echo "Running volume $i, slice $j"
        python main.py --volume $i --slice $j
        done
    done
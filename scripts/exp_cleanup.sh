echo "input folder is $1"
rm  ~/kusanagi/experiments/logs/SparseChem/$1        -r  ## 2>/dev/null
rm  ~/kusanagi/experiments/checkpoints/SparseChem/$1 -r  ## 2>/dev/null
rm  ~/kusanagi/experiments/results/SparseChem/$1     -r  ## 2>/dev/null
rm  ~/kusanagi/new_experiments/SparseChem/$1         -r  ## 2>/dev/null
echo "clean up of run $1 is complete"

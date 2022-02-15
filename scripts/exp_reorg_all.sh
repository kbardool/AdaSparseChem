find ~/kusanagi/experiments/logs/SparseChem -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | while read dir; do
    echo "reorganize $dir into    new_experiments-->SparseChem-->$dir"
    # mkdir ~/kusanagi/new_exeriments/SparseChem/$1 -p
    #mkdir ~/kusanagi/new_experiments/SparseChem/$1/logs
    #mkdir ~/kusanagi/new_experiments/SparseChem/$1/results
    #mkdir ~/kusanagi/new_experiments/SparseChem/$1/chkpts
    cp ~/kusanagi/experiments/logs/SparseChem/$dir         ~/kusanagi/new_experiments/SparseChem/$dir  -r
    cp ~/kusanagi/experiments/checkpoints/SparseChem/$dir  ~/kusanagi/new_experiments/SparseChem  -r
    cp ~/kusanagi/experiments/results/SparseChem/$dir      ~/kusanagi/new_experiments/SparseChem  -r

    echo "movements of run $dir is complete"
done
 


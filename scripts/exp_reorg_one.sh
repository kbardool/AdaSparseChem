echo "reorganize $1 into    new_experiments-->SparseChem-->$1"
# mkdir ~/kusanagi/new_exeriments/SparseChem/$1 -p
#mkdir ~/kusanagi/new_experiments/SparseChem/$1/logs
#mkdir ~/kusanagi/new_experiments/SparseChem/$1/results
#mkdir ~/kusanagi/new_experiments/SparseChem/$1/chkpts
cp ~/kusanagi/experiments/logs/SparseChem/$1         ~/kusanagi/new_experiments/SparseChem/$1  -r
cp ~/kusanagi/experiments/checkpoints/SparseChem/$1  ~/kusanagi/new_experiments/SparseChem  -r
cp ~/kusanagi/experiments/results/SparseChem/$1      ~/kusanagi/new_experiments/SparseChem  -r

echo "movements of run $1 is complete"

 


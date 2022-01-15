find ~/kusanagi/experiments/logs/SparseChem -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | while read dir; do
  echo "reorganize $dir into new_experiments/SparseChem/$dir"
done
 
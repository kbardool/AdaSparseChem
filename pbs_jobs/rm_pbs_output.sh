 #!/bin/bash
 # -R: Read all files under each directory, recursively.  Follow  all symbolic links, unlike -r.
 # -E: Interpret PATTERNS as extended regular expressions (EREs).
 # -l: Suppress normal output; instead print the name of each input file from which output would 
 #     normally have been printed.
 grep -RFl "wandb: Find logs at: ."  ../pbs_output/ | awk '{print "rm " $1}' > del_pbs.sh
 grep -REl -e "^Job [[:digit:]]{8} finished" ../pbs_output/ | awk '{print "rm " $1}' >> del_pbs.sh
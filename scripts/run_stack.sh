~/sentinel/sentinel_stack.py
mkdir igrams && cd igrams
~/sentinel/sbas_list.py 500 500 && cat sbas_list
~/sentinel/ps_sbas_igrams.py sbas_list ../elevation.dem.rsc 1 1 6400 3400 8 8
for i in *.int ; do dismphfile $i 800 ; mv dismph.tif `echo $i | sed 's/int$/tif/'` ; done


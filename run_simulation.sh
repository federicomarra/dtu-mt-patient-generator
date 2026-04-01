#!/bin/bash
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J t1d-sim
### -- ask for number of cores (default: 1) -- 
#BSUB -n 32
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 3GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 7:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u furlanettoguido@gmail.com
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o /zhome/15/9/213553/dtu-mt-patient-generator/logs/output_%J.out
#BSUB -e /zhome/15/9/213553/dtu-mt-patient-generator/logs/output_%J.err

# here follow the commands you want to execute with input.in as the input file
source /zhome/15/9/213553/dtu-mt-patient-generator/.venv/bin/activate

cd /zhome/15/9/213553/dtu-mt-patient-generator

# use this for unbuffered output, so that you can check in real-time
# (with tail -f logs/output_.out logs/output_.err)
# what your program was printing "on the screen"
python3 -u library_generator.py

# use this for just piping everything into a file, 
# the program knows then, that it's outputting to a file
# and not to a screen, and also combine stdout&stderr
# python3 library_generator.py > logs/output_$LSB_JOBID.out 2>&1
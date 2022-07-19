
OLD FILES
/Users/graham/code/school/FALL2020/ECE697/finalproject/submission

===============================================================================================================================================
USING SLURM TO LAUNCH A .py SCRIPT


change the directory to scratch if you're not in scratch already.
$ cd scratch

make a directory to collect output and error reports
$ mkdir outputs

make necessary changes such as adding your <r2-user-name> and <your-email-id> to
tensorflow-batch-submit.bash file that was provided to you. Copy the modified
tensorflow-batch-submit.bash to the scratch folder on your r2 account.

submit the job by executing

$ sbatch tensorflow-batch-submit.bash

you'll receive email saying if the job was succesfull or not
if you received an email with subject:
SLURM Job_id=<jobid> Name=PYTHON Ended, Run time 00:00:00, FAILED, ExitCode 1
then your job failed.

if you received an email with subject:
SLURM Job_id=<jobid> Name=PYTHON Ended, Run time 00:00:26, COMPLETED, ExitCode 0
then your job ended properly.

====================================================================================================================================================
COPY NECESSARY FILES FROM R2 TO LOCAL MACHINE
scp <r2-user-name>@r2.boisestate.edu:/home/<r2-user-name>/scratch/outputs/filename.extension /home/<local-username>/Downloads

copy the cnn_model_history_<pid>.pkl file to local and open the pickle file by writing python code (code given in the last commented
out section of train_cnn_script.py) and plot the training and validation accuracies.



% BATCH SUBMIT
#!/bin/bash

#Account and Email Information
#SBATCH -A gannett
#SBATCH --mail-type=end
#SBATCH --mail-user=grahamannett@u.boisestate.edu

#SBATCH -J PYTHON          # job name
#SBATCH -o outputs/results.o%j # output and error file name (%j expands to jobID)
#SBATCH -e outputs/errors.e%j
#SBATCH -n 1               # Run one process
#SBATCH --cpus-per-task=28 # allow job to multithread across all cores
#SBATCH -p defq            # queue (partition) -- defq, ipowerq, eduq, gpuq.
#SBATCH -t 1-00:00:00      # run time (d-hh:mm:ss)
ulimit -v unlimited
ulimit -s unlimited
ulimit -u 1000

# module load cuda10.0/toolkit/10.0.130 # loading cuda libraries/drivers
module load python/intel/3.7 # loading python environment

python3 -m pip install --user sklearn
python3 -m pip install --user tensorflow==1.15
python3 -m pip install --user keras==2.2.4
python3 -m pip install --user opencv-python
python3 -m pip install --user pandas
python3 -m pip install --user progressbar2
python3 -m pip install --user h5py==2.10.0
python3 -m pip install --user seaborn
python3 -m pip install --user pillow

python3 src/main.py --dataset="imagenette2" --epochs 100
python3 train_cnn_script.py

#######################
### Import Packages ###
#######################

import numpy as np
import os
import sys
import psutil
import nibabel as nib
from time import time
import json
import dataflow as flow
import matplotlib.pyplot as plt
from contextlib import contextmanager
import warnings
from shutil import copyfile
warnings.filterwarnings("ignore")
import platform
if platform.system() != 'Windows':
    sys.path.insert(0, '/home/users/brezovec/.local/lib/python3.6/site-packages/lib/python/')
    import ants

#####################
### Main Function ###
#####################

def main(args):

	###############################################
	### Parse the dictionary of input arguments ###
	###############################################

    logfile = args['logfile']
    save_directory = args['save_directory']
    flip_X = args['flip_X']
    flip_Z = args['flip_Z']
    type_of_transform = args['type_of_transform'] # SyN or Affine
    save_warp_params = args['save_warp_params']

    ### Brain space you are warping into
    fixed_path = args['fixed_path']
    fixed_fly = args['fixed_fly']
    fixed_resolution = args['fixed_resolution']

    ### Brain that is being warped
    moving_path = args['moving_path']
    moving_fly = args['moving_fly']
    moving_resolution = args['moving_resolution']

    ### Downsample if you like
    low_res = args['low_res']
    very_low_res = args['very_low_res']

    ### Specific to ANTs SyN transform
    grad_step = args['grad_step']
    flow_sigma = args['flow_sigma']
    total_sigma = args['total_sigma']
    syn_sampling = args['syn_sampling']

    ### Mimic is an optional 2nd brain to warp based on the warp params of moving to fixed
    try:
        mimic_path = args['mimic_path']
        mimic_fly = args['mimic_fly']
        mimic_resolution = args['mimic_resolution']
    except:
        mimic_path = None
        mimic_fly = None
        mimic_resolution = None

    ### this pringlogging is custom to our system
    width = 120
    printlog = getattr(flow.Printlog(logfile=logfile), 'print_to_log')

    ###################
    ### Load Brains ###
    ###################

    ### Fixed
    fixed = np.asarray(nib.load(fixed_path).get_data().squeeze(), dtype='float32')
    fixed = ants.from_numpy(fixed)
    fixed.set_spacing(fixed_resolution)
    if low_res:
        fixed = ants.resample_image(fixed,(256,128,49),1,0)
    elif very_low_res:
        fixed = ants.resample_image(fixed,(128,64,49),1,0)

    ### Moving
    moving = np.asarray(nib.load(moving_path).get_data().squeeze(), dtype='float32')
    if flip_X:
        moving = moving[::-1,:,:]
    if flip_Z:
        moving = moving[:,:,::-1]
    moving = ants.from_numpy(moving)
    moving.set_spacing(moving_resolution)
    if low_res:
        moving = ants.resample_image(moving,(256,128,49),1,0)
    elif very_low_res:
        moving = ants.resample_image(moving,(128,64,49),1,0)

    ### Mimic
    if mimic_path is not None:
        mimic = np.asarray(nib.load(mimic_path).get_data().squeeze(), dtype='float32')
        if flip_X:
            mimic = mimic[::-1,:,:]
        if flip_Z:
            mimic = mimic[:,:,::-1]
        mimic = ants.from_numpy(mimic)
        mimic.set_spacing(mimic_resolution)
        printlog('Starting {} to {}, with mimic {}'.format(moving_fly, fixed_fly, mimic_fly))
    else:
        printlog('Starting {} to {}'.format(moving_fly, fixed_fly))

    #############
    ### Align ###
    #############

    t0=time()
    with stderr_redirected(): # to prevent itk gaussian error infinite printing
        moco = ants.registration(fixed,
                                 moving,
                                 type_of_transform=type_of_transform,
                                 grad_step=grad_step, 
                                 flow_sigma=flow_sigma,
                                 total_sigma=total_sigma,
                                 syn_sampling=syn_sampling)
        
    printlog('Fixed: {}, {} | Moving: {}, {} | {} | {}'.format(fixed_fly, fixed_path.split('/')[-1], moving_fly, moving_path.split('/')[-1], type_of_transform, sec_to_hms(time()-t0)))

    ################################
    ### Save warp params if True ###
    ################################

    if save_warp_params:
        fwdtransformlist = moco['fwdtransforms']
        fwdtransforms_save_dir = os.path.join(save_directory, '{}-to-{}_fwdtransforms'.format(moving_fly, fixed_fly))
        if low_res:
            fwdtransforms_save_dir += '_lowres'
        if not os.path.exists(fwdtransforms_save_dir):
            os.mkdir(fwdtransforms_save_dir)
        for source_path in fwdtransformlist:
            source_file = source_path.split('/')[-1]
            target_path = os.path.join(fwdtransforms_save_dir, source_file)
            copyfile(source_path, target_path)

    # Added this saving of inv transforms 2020 Dec 19
    if save_warp_params:
        fwdtransformlist = moco['invtransforms']
        fwdtransforms_save_dir = os.path.join(save_directory, '{}-to-{}_invtransforms'.format(moving_fly, fixed_fly))
        if low_res:
            fwdtransforms_save_dir += '_lowres'
        if not os.path.exists(fwdtransforms_save_dir):
            os.mkdir(fwdtransforms_save_dir)
        for source_path in fwdtransformlist:
            source_file = source_path.split('/')[-1]
            target_path = os.path.join(fwdtransforms_save_dir, source_file)
            copyfile(source_path, target_path)

    ##################################
    ### Apply warp params to mimic ###
    ##################################

    if mimic_path is not None:
        mimic_moco = ants.apply_transforms(fixed, mimic, moco['fwdtransforms'])

    ############
    ### Save ###
    ############

    if flip_X:
        save_file = os.path.join(save_directory, moving_fly + '_m' + '-to-' + fixed_fly)
    else:
        save_file = os.path.join(save_directory, moving_fly + '-to-' + fixed_fly)
    if low_res:
        save_file += '_lowres'
    save_file += '.nii'
    nib.Nifti1Image(moco['warpedmovout'].numpy(), np.eye(4)).to_filename(save_file)


def sec_to_hms(t):
        secs=F"{np.floor(t%60):02.0f}"
        mins=F"{np.floor((t/60)%60):02.0f}"
        hrs=F"{np.floor((t/3600)%60):02.0f}"
        return ':'.join([hrs, mins, secs])

@contextmanager
def stderr_redirected(to=os.devnull):

    fd = sys.stderr.fileno()

    def _redirect_stderr(to):
        sys.stderr.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stderr = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stderr:
        with open(to, 'w') as file:
            _redirect_stderr(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stderr(to=old_stderr) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

if __name__ == '__main__':
    main(json.loads(sys.argv[1]))
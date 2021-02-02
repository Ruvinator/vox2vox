from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import glob
import os
import nibabel as nb

class CTDataset(Dataset):
    def __init__(self, datapath, transforms_):
        self.datapath = datapath
        self.transforms = transforms_
        self.samples = ['.' + x for x in glob.glob(blabla + '/*')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Index representation for brain MRI images (.nii files are 4D):
        #     0: FLAIR: "Fluid Attenuated Inversion Recovery" (FLAIR)
        #     1: T1w: "T1-weighted"
        #     2: t1gd: "T1-weighted with gadolinium contrast enhancement" (T1-Gd)
        #     3: T2w: "T2-weighted"
        image = nb.load(self.samples[idx]).get_fdata()[:, :, :, 0]

        # Brain MRI labels are 3D as-is
        mask = nb.load(self.get_label_from_sample(self.samples[idx])).get_fdata()

        if self.transforms:
            image, mask = self.transforms(image), self.transforms(mask)

        return {"A": image, "B": mask}

    def get_label_from_sample(self, sample):
        s_dirname = os.path.dirname(sample)
        s_basename = os.path.basename(sample)
        return os.path.join(s_dirname, 'labels', s_basename)


# Command that works:
# python train.py --dataset_name BrainMRI --img_height 144 --img_width 240 --img_depth 240

# if __name__ == '__main__':
    # data_dir = './data/BrainMRI/train/'

    # # Changing filetype
    # for f in glob.glob(data_dir + '*'):
    #     f_new = f.split('.')[1]
    #     os.rename(f, f'.{f_new}.nii')

    # print(nb.load(glob.glob(data_dir)[0]).get_fdata())

    test_dir = os.path.join("./data/BrainMRI/train/", "*")
    samples = [x for x in glob.glob(test_dir) if x[-3:] == 'nii']
    print(samples)
    s_dirname = os.path.dirname(samples[0])
    s_basename = os.path.basename(samples[0])
    s_path = os.path.join(s_dirname, s_basename)
    s_path_label = os.path.join(s_dirname, 'labels', s_basename)

    image = nb.load(s_path).get_fdata()[:,:,:154,0]
    print(image.shape)

    label = nb.load(s_path_label).get_fdata()
    print(label.shape)

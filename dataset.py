from torch.utils.data import Dataset, DataLoader
import os, glob
import nibabel as nb

class CTDataset(Dataset):
    def __init__(self, datapath, transforms_):
        self.datapath = datapath
        self.transforms = transforms_
        self.samples = [x for x in glob.glob(datapath + '*') if x[-3:] == 'nii']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Must be divisible by 16
        dim_bound_xy = 192
        dim_bound_z = 144
        # Selected bound works with:
        # python train.py --dataset_name BrainMRI --img_height 144 --img_width 192 --img_depth 192
        # 240x240x144 results in CUDA out of memory error

        # Index representation for brain MRI images (.nii files are 4D):
        #     0: FLAIR: "Fluid Attenuated Inversion Recovery" (FLAIR)
        #     1: T1w: "T1-weighted"
        #     2: t1gd: "T1-weighted with gadolinium contrast enhancement" (T1-Gd)
        #     3: T2w: "T2-weighted"
        image = nb.load(self.samples[idx]).get_fdata()[:dim_bound_xy, :dim_bound_xy, :dim_bound_z, 0]
        image /= float(image.max()) # Normalize

        # Brain MRI labels are 3D as-is
        mask = nb.load(self.get_label_from_sample(self.samples[idx])).get_fdata()[:dim_bound_xy, :dim_bound_xy, :dim_bound_z]
        mask /= float(mask.max()) # Normalize

        if self.transforms:
            image, mask = self.transforms(image), self.transforms(mask)

        return {"A": image, "B": mask}

    def get_label_from_sample(self, sample):
        s_dirname = os.path.dirname(sample)
        s_basename = os.path.basename(sample)
        return os.path.join(s_dirname, 'labels', s_basename)

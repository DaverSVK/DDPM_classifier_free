
from pytorch_fid import fid_score

def calculateFID(generated_images_folder, dataset_images_folder, device, batch_size=2):
    return fid_score.calculate_fid_given_paths(
        [generated_images_folder, dataset_images_folder],
        batch_size=batch_size,
        device=device,
        dims=2048,
    )

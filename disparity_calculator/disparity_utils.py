from functions import initialize_environment, create_output_directory, load_configuration, initialize_model, preprocess_images, compute_disparity, save_disparity
from omegaconf import OmegaConf

def calc_disparity(img0, img1, out_dir: str, scale=1, hiera=False):
    
    initialize_environment()

    create_output_directory(out_dir)

    args = OmegaConf.create(load_configuration())
    
    args.left_file = img0
    args.right_file = img1
    
    args.scale = scale
    args.hiera = hiera

    img0, img1 = preprocess_images(args.left_file, args.right_file, args.scale)
    
    model = initialize_model(args)

    disp = compute_disparity(model, img0, img1, args)
    
    save_disparity(disp, f'{out_dir}/disp.npy')
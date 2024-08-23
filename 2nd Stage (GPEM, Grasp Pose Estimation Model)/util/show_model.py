import torch

def load_and_display_model_weights(file_path):
    # Load the .tar file
    checkpoint = torch.load(file_path, map_location=torch.device('cpu'))

    # Display the keys in the checkpoint
    print("Keys in the checkpoint:")
    for key in checkpoint.keys():
        print(key)
    
    # Assuming the checkpoint contains 'model_state_dict' and 'optimizer_state_dict'
    if 'model_state_dict' in checkpoint:
        print("\nModel State Dict:")
        model_state_dict = checkpoint['model_state_dict']
        for param_tensor in model_state_dict:
            print(f"{param_tensor}:\t{model_state_dict[param_tensor].size()}")


# Example usage
file_path = '/home/dsp/6DCM_Grasp/zuo/MSCG/logs/log_ssd_AFF_MSCG_concat_ATT/minkuresunet_epoch22.tar'
load_and_display_model_weights(file_path)

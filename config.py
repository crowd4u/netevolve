# Third Party Library
import torch

select_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# select_device = "mps"
# select_device = "cpu"

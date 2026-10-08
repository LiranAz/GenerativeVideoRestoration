import torch
import pyiqa
import tqdm, collections, argparse
from pathlib import Path
from datetime import datetime
from utils import util_image


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate(in_path, ref_path, ntest):
    metric_dict, metric_paired_dict = {}, {}

    ref_path_list = None
    if ref_path is not None:
        ref_path = Path(ref_path) if not isinstance(ref_path, Path) else ref_path
        ref_path_list = sorted([x for x in ref_path.glob("*.[jpJP][pnPN]*[gG]")])
        if ntest is not None: ref_path_list = ref_path_list[:ntest]
        
        metric_paired_dict["psnr"]=pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
        metric_paired_dict["lpips"]=pyiqa.create_metric('lpips').to(device)
    
    metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(device)
    metric_dict["musiq"] = pyiqa.create_metric('musiq').to(device)
    metric_dict["maniqa"] = pyiqa.create_metric('maniqa').to(device)
    
    in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
    assert in_path.is_dir()
    
    lr_path_list = sorted([x for x in in_path.glob("*.[jpJP][pnPN]*[gG]")])
    if ntest is not None: lr_path_list = lr_path_list[:ntest]
    
    print(f'Find {len(lr_path_list)} images in {in_path}')
    result = collections.OrderedDict()
    for i in tqdm.tqdm(range(len(lr_path_list))):
        _in_path = lr_path_list[i]
        _ref_path = ref_path_list[i] if ref_path_list is not None else None
        
        im_in = util_image.imread(_in_path, chn='rgb', dtype='float32')  # h x w x c
        im_in_tensor = util_image.img2tensor(im_in).cuda()              # 1 x c x h x w

        if _ref_path is not None:
            im_ref = util_image.imread(_ref_path, chn='rgb', dtype='float32')  # h x w x c
            im_ref_tensor = util_image.img2tensor(im_ref).cuda()    
            for key, metric in metric_paired_dict.items():
                result[key] = result.get(key, 0) + metric(im_in_tensor, im_ref_tensor).item()

        for key, metric in metric_dict.items():
            with torch.cuda.amp.autocast():
                result[key] = result.get(key, 0) + metric(im_in_tensor).item()
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('results/log.txt', 'a') as file:
        file.write(f"\n[{current_time}]: {in_path}\n")
        for key, res in result.items():
            print(f"{key}: {round(res/len(lr_path_list), 4):.4f}")
            file.write(f"{key}:{round(res/len(lr_path_list), 4):.4f}\n")
        
        if ref_path is not None:
            fid_metric = pyiqa.create_metric('fid')
            fid_score = fid_metric(in_path, ref_path)
            print(f"fid: {round(fid_score, 4):.4f}\n")
            file.write(f"fid: {round(fid_score, 4):.4f}\n")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',"--in_path", type=str, required=True)
    parser.add_argument("-r", "--ref_path", type=str, default=None)
    parser.add_argument("--ntest", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.in_path, args.ref_path, args.ntest)
    
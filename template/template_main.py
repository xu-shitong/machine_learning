import sys
from attrdict import AttrDict
from utils import *
import os
args_dict = get_config(sys.argv[1])
args = AttrDict(args_dict)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
import torch
from torch.utils.data import DataLoader
import traceback
from tqdm import tqdm

# Import packages
import sys,humanize,psutil,GPUtil

# Define function
def mem_report():
    print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))

    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

def train_func(args, epoch, model, dataloader, optimizer, device, train):
    if train:
        model.train()
    else:
        model.eval()

    acc_losses = [0] * ...

    titer = dataloader
    if train:
        titer = tqdm(dataloader, unit="iter")
    for i, data in enumerate(titer):

        ... # forward pass

        losses = ...
        l = ... # weighted loss for each loss term

        if train:
            titer.set_description(f"iter {i}")
            titer.set_postfix(loss=l.item(),
                              ...
                              )

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        acc_losses[1:] = [acc_l + new_l.item() for acc_l, new_l in zip(acc_losses[1:], losses)]
        acc_losses[0] += l.item()

    return [l / len(dataloader) for l in acc_losses]

def main_func(args):

    mem_report()

    torch.cuda.manual_seed(42)
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print("using device: ", dev)

    process_id = os.getpid()
    print("Start experiment", process_id)
    log = open(f"{process_id}.txt", "a")
    log.write("\n".join([str(key) + " " + str(args.get(key)) for key in args.keys()]) + "\n") # write hyperparameter in log file

    # ======== dataset ========

    train_dataset = ...
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=6)
    
    val_dataset = ...
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=6)

    # ======== model ========
    if args.model_name == "...":
        model = ...
        model.to(device)
    else:
        raise NotImplementedError(args.model_name)
    
    if args.load_model != "":
        model.load_state_dict(torch.load(args.load_model)["state_dict"], strict=True)

    # ======== optizer, scheduler =========
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0001,
            amsgrad=True)
    else:
        raise NotImplementedError(args.optimizer)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=args.lr_decay_epoch,
                                                   gamma=args.lr_decay_gamma)

    # ======== train =========
    # send_email(f"Started experiment {process_id}", "experiment args " + str(args))

    # train
    for epoch in range(args.epoch_num):
        train_losses = train_func(args, epoch, model, train_dataloader, optimizer, device=device, train=True)
        train_losses = [str(num) for num in train_losses]

        with torch.no_grad():
            val_losses = train_func(args, epoch, model, val_dataloader, optimizer=None, device=device, train=False)
            val_losses = [str(num) for num in val_losses]

        log.write(f"Epoch: {epoch}, train_losses: {', '.join(train_losses)}, " + 
                  f"val_losses: {', '.join(val_losses)}, lr:{lr_scheduler.get_last_lr()[0]}\n")
        print(f"Finish {epoch} / {args.epoch_num}, id {process_id}")

        lr_scheduler.step()

        if epoch % args.save_epoch == 0 and epoch != 0:
            torch.save({"state_dict": model.state_dict()}, f"{process_id}_{epoch}.pt")
    torch.save({"state_dict": model.state_dict()}, f"{process_id}.pt")
    log.close()

    # send_email(f"Finished experiment {process_id}", "experiment args " + str(args))

    mem_report()
    print("Finish experiment", process_id)


if __name__ == "__main__":
    process_id = os.getpid()
    try:
        print("running ", process_id)

        args_dict = get_config(sys.argv[1])
        args = AttrDict(args_dict)

        main_func(args)
    except Exception:
        print("training failed", traceback.format_exc())
        # send_email(f"Experiment {process_id} failed", traceback.format_exc())


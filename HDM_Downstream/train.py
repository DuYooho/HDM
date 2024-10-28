import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PraNet.lib.PraNet_Res2Net_cross import PraNet
from PraNet.utils.utils import clip_gradient, adjust_lr, AvgMeter
from PVT.lib.pvt_cross import PolypPVT
from FCBFormer.Models.models_cross_attn import FCBFormer
from FCBFormer.Metrics.losses import SoftDiceLoss
from FCBFormer.Metrics.performance_metrics import DiceScore
from data.polyp_all import PolypBase

from global_config import workspace, parser_config, EXP


def get_data(args):
    # 2D_Polyp
    train_dataset = PolypBase(name="public_polyp_train", size=args.trainsize, use_processed_features=True)
    val_dataset = PolypBase(name="public_polyp_validation", size=args.trainsize, use_processed_features=True)

    # SUN-SEG
    # train_dataset = PolypBase(name="SUN-SEG/TrainDataset", size=args.trainsize, use_processed_features=True)
    # val_dataset = PolypBase(name="SUN-SEG/TestUnseenDataset", size=args.trainsize, use_processed_features=True)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=6,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=6,
        shuffle=False,
    )

    return train_dataloader, val_dataloader


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def train_PraNet(args, train_loader, model, optimizer, epoch, val_loader):
    model.train()
    global best
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record2, loss_record3, loss_record4, loss_record5 = (
        AvgMeter(),
        AvgMeter(),
        AvgMeter(),
        AvgMeter(),
    )
    loop = tqdm(enumerate(train_loader, start=1), total=len(train_loader))
    for i, pack in loop:
        for rate in size_rates:
            optimizer.zero_grad()
            images, gts, features = pack["image"], pack["mask"], pack["features"]

            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            features = Variable(features).cuda()
            features = F.interpolate(
                features,
                size=(32, 32),
                mode="bilinear",
                align_corners=True,
            )

            # ---- rescale ----
            trainsize = int(round(args.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(
                    images,
                    size=(trainsize, trainsize),
                    mode="bilinear",
                    align_corners=True,
                )
                gts = F.interpolate(
                    gts,
                    size=(trainsize, trainsize),
                    mode="bilinear",
                    align_corners=True,
                )
            # ---- forward ----
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2 = model(images, features)
            # ---- loss function ----
            loss5 = structure_loss(lateral_map_5, gts)
            loss4 = structure_loss(lateral_map_4, gts)
            loss3 = structure_loss(lateral_map_3, gts)
            loss2 = structure_loss(lateral_map_2, gts)
            loss = loss2 + loss3 + loss4 + loss5  # TODO: try different weights for loss
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record2.update(loss2.data, args.batch_size)
                loss_record3.update(loss3.data, args.batch_size)
                loss_record4.update(loss4.data, args.batch_size)
                loss_record5.update(loss5.data, args.batch_size)

        loop.set_description(f"Train Epoch [{epoch}/{args.epoch}]")
        loop.set_postfix(lateral_5=loss_record5.show().item())
    print(f"Epoch {epoch}, Loss: {loss_record5.show().item()}")
    save_path = Path(workspace).joinpath("logs", "PraNet", EXP).as_posix()
    os.makedirs(save_path, exist_ok=True)

    mean_dice = val_PraNet(val_loader, model)
    print(f"Epoch: {epoch}, dice = {mean_dice}, last best dice = {best}")
    if mean_dice > best:
        best = mean_dice
        torch.save(model.state_dict(), os.path.join(save_path, "PraNet-best.pth"))
        torch.save({"best": epoch}, os.path.join(save_path, "PraNet-best-%d.pth" % epoch))
        print(f"Reached best dice at epoch: {epoch}; best dice = {best}")


def train_FCBFormer(args, optimizer, train_loader, model, epoch, val_loader, Dice_loss, BCE_loss, perf):
    model.train()
    global best

    loss_record = AvgMeter()
    loop = tqdm(enumerate(train_loader, start=1), total=len(train_loader))
    for i, example in loop:
        image, mask, features = example["image"], example["mask"], example["features"]

        image, mask, features = image.cuda(), mask.cuda(), features.cuda()

        features = F.interpolate(
            features,
            size=(32, 32),
            mode="bilinear",
            align_corners=True,
        )
        optimizer.zero_grad()
        output = model(image, features)

        loss = Dice_loss(output, mask) + BCE_loss(torch.sigmoid(output), mask)
        loss.backward()
        optimizer.step()

        loss_record.update(loss.data, args.batch_size)

        loop.set_description(f"Train Epoch [{epoch}/{args.epoch}]")
        loop.set_postfix(loss=loss_record.show().item())

    print(f"Epoch {epoch}, Loss: {loss_record.show().item()}")
    save_path = Path(workspace).joinpath("logs", "FCBFormer", EXP).as_posix()
    os.makedirs(save_path, exist_ok=True)

    test_measure_mean, test_measure_std = val_FCBFormer(val_loader, model, perf)
    print(f"Epoch: {epoch}, dice_mean = {test_measure_mean}, dice_std = {test_measure_std}, last best dice = {best}")
    if test_measure_mean > best:
        best = test_measure_mean
        torch.save(model.state_dict(), os.path.join(save_path, "FCBFormer-best.pth"))
        torch.save({"best": epoch}, os.path.join(save_path, "FCBFormer-best-%d.pth" % epoch))
        print(f"Reached best dice at epoch: {epoch}; best dice = {best}")

    return test_measure_mean


def train_PVT(opt, train_loader, model, optimizer, epoch, val_loader):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25]
    loss_P2_record = AvgMeter()
    loop = tqdm(enumerate(train_loader, start=1), total=len(train_loader))
    for i, pack in loop:
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts, features = pack["image"], pack["mask"], pack["features"]

            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            features = Variable(features).cuda()
            features = F.interpolate(
                features,
                size=(32, 32),
                mode="bilinear",
                align_corners=True,
            )

            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(
                    images,
                    size=(trainsize, trainsize),
                    mode="bilinear",
                    align_corners=True,
                )
                gts = F.interpolate(
                    gts,
                    size=(trainsize, trainsize),
                    mode="bilinear",
                    align_corners=True,
                )
            # ---- forward ----
            P1, P2 = model(images, features)
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss = loss_P1 + loss_P2
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_P2_record.update(loss_P2.data, opt.batch_size)

        # loop
        loop.set_description(f"Train Epoch [{epoch}/{opt.epoch}]")
        loop.set_postfix(lateral_5=loss_P2_record.show().item())

    # save model
    print(f"Epoch {epoch}, Loss: {loss_P2_record.show().item()}")
    save_path = Path(workspace).joinpath("logs", "PVT", EXP).as_posix()
    os.makedirs(save_path, exist_ok=True)

    mean_dice = val_PVT(val_loader, model)
    print(f"Epoch: {epoch}, dice = {mean_dice}, last best dice = {best}")
    if mean_dice > best:
        best = mean_dice
        torch.save(model.state_dict(), os.path.join(save_path, "PVT-best.pth"))
        torch.save({"best": epoch}, os.path.join(save_path, "PVT-best-%d.pth" % epoch))
        print(f"Reached best dice at epoch: {epoch}; best dice = {best}")


def val_PVT(val_loader, model):
    model.eval()
    data_len = len(val_loader)
    print(f"Val data length: {data_len}")
    DSC = 0.0
    with torch.no_grad():
        loop = tqdm(enumerate(val_loader, start=1), total=data_len)
        for i, example in loop:
            # ---- data prepare ----
            image, gt_shape, name, gt, feature = (
                example["image"],
                example["ori_shape"],
                example["image_name"],
                example["mask"],
                example["features"],
            )

            image = image.cuda()

            feature = feature.cuda()
            feature = F.interpolate(
                feature,
                size=(32, 32),
                mode="bilinear",
                align_corners=True,
            )

            # ---- forward ----
            res, res1 = model(image, feature)
            # eval Dice
            res = F.interpolate(res + res1, size=gt.shape[-2:], mode="bilinear", align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            input = res
            target = gt.numpy().squeeze()

            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = input_flat * target_flat
            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice = "{:.4f}".format(dice)
            dice = float(dice)
            DSC = DSC + dice

            # loop
            loop.set_description(f"Val")
            loop.set_postfix(dice=dice)

    model.train()
    return DSC / data_len


def val_PraNet(val_loader, model):
    model.eval()
    data_len = len(val_loader)
    print(f"Val data length: {data_len}")
    DSC = 0.0
    with torch.no_grad():
        loop = tqdm(enumerate(val_loader, start=1), total=data_len)
        for i, example in loop:
            # ---- data prepare ----
            image, gt_shape, name, gt, feature = (
                example["image"],
                example["ori_shape"],
                example["image_name"],
                example["mask"],
                example["features"],
            )

            image = image.cuda()

            feature = feature.cuda()
            feature = F.interpolate(
                feature,
                size=(32, 32),
                mode="bilinear",
                align_corners=True,
            )

            # ---- forward ----
            res5, res4, res3, res2 = model(image, feature)
            res = res2
            # eval Dice
            res = F.interpolate(res, size=gt.shape[-2:], mode="bilinear", align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            input = res
            target = gt.numpy().squeeze()

            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = input_flat * target_flat
            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice = "{:.4f}".format(dice)
            dice = float(dice)
            DSC = DSC + dice

            # loop
            loop.set_description(f"Val")
            loop.set_postfix(dice=dice)

    model.train()
    return DSC / data_len


def val_FCBFormer(val_loader, model, perf_measure):
    model.eval()
    data_len = len(val_loader)
    print(f"Val data length: {data_len}")
    perf_accumulator = []
    with torch.no_grad():
        loop = tqdm(enumerate(val_loader, start=1), total=data_len)
        for i, example in loop:
            # ---- data prepare ----
            image, gt_shape, name, gt, feature = (
                example["image"],
                example["ori_shape"],
                example["image_name"],
                example["mask"],
                example["features"],
            )

            image = image.cuda()
            gt = gt.cuda()

            feature = feature.cuda()
            feature = F.interpolate(
                feature,
                size=(32, 32),
                mode="bilinear",
                align_corners=True,
            )

            # ---- forward ----
            output = model(image, feature)
            perf_accumulator.append(perf_measure(output, gt).item())

            # loop
            loop.set_description(f"Val")
            loop.set_postfix(dice=np.mean(perf_accumulator))

    model.train()
    return np.mean(perf_accumulator), np.std(perf_accumulator)


if __name__ == "__main__":
    args = parser_config()
    train_dataloader, val_dataloader = get_data(args)

    if args.model_name == "PVT":
        base_lr = 6.25e-6  # 1e-4 / 16
        args.lr = args.batch_size * base_lr
        model = PolypPVT().cuda()

        params = model.parameters()
        optimizer = torch.optim.AdamW(params, args.lr, weight_decay=1e-4)

        total_step = len(train_dataloader)

        print("#" * 20, "Start Training", "#" * 20)

        best = 0.0

        for epoch in range(1, args.epoch + 1):
            adjust_lr(optimizer, args.lr, epoch, 0.1, 200)
            train_PVT(args, train_dataloader, model, optimizer, epoch, val_dataloader)

    elif args.model_name == "PraNet":
        base_lr = 6.25e-6  # 1e-4 / 16
        args.lr = args.batch_size * base_lr
        model = PraNet().cuda()
        params = model.parameters()
        optimizer = torch.optim.AdamW(params, args.lr)

        total_step = len(train_dataloader)

        print("#" * 20, "Start Training", "#" * 20)

        best = 0.0

        for epoch in range(1, args.epoch + 1):
            adjust_lr(optimizer, args.lr, epoch, 0.1, 50)
            train_PraNet(args, train_dataloader, model, optimizer, epoch, val_dataloader)

    elif args.model_name == "FCBFormer":
        base_lr = 6.25e-6  # 1e-4 / 16
        args.lr = args.batch_size * base_lr

        args.lrs = "true"
        base_lrs_min = 6.25e-8  # 1e-6 / 16
        args.lrs_min = args.batch_size * base_lrs_min

        Dice_loss = SoftDiceLoss()
        BCE_loss = torch.nn.BCELoss()

        perf = DiceScore()

        model = FCBFormer().cuda()

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, min_lr=args.lrs_min, verbose=True
        )

        total_step = len(train_dataloader)

        print("#" * 20, "Start Training", "#" * 20)

        best = 0.0

        for epoch in range(1, args.epoch + 1):
            test_measure_mean = train_FCBFormer(
                args, optimizer, train_dataloader, model, epoch, val_dataloader, Dice_loss, BCE_loss, perf
            )
            scheduler.step(test_measure_mean)

import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder
from .dataset import SpeechDataset
from .optimization import create_optimizer, create_scheduler
from .augmentations import create_augmentation_pipeline
from .losses.factory import create_loss


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    device = "cuda"

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    # Create loss function using modular factory (supports both old and new config style)
    if "loss" in args and isinstance(args["loss"], dict):
        # New modular style
        loss_fn = create_loss(args["loss"])
    else:
        # Old style (backward compatibility) - standard CTC loss
        loss_fn = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    
    # Create optimizer using modular factory (supports both old and new config style)
    if "optimizer" in args and isinstance(args["optimizer"], dict):
        # New modular style
        optimizer = create_optimizer(model, args["optimizer"])
    else:
        # Old style (backward compatibility)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args["lrStart"],
            betas=(0.9, 0.999),
            eps=0.1,
            weight_decay=args["l2_decay"],
        )
    
    # Create scheduler using modular factory
    if "scheduler" in args and isinstance(args["scheduler"], dict):
        # New modular style
        scheduler = create_scheduler(optimizer, args["scheduler"], total_steps=args["nBatch"])
    else:
        # Old style (backward compatibility)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args["lrEnd"] / args["lrStart"],
            total_iters=args["nBatch"],
        )
    
    # Create augmentation pipeline
    augmentation_pipeline = None
    if "augmentation" in args and isinstance(args["augmentation"], dict):
        # New modular style
        augmentation_pipeline = create_augmentation_pipeline(args["augmentation"])
    # Old style handled in training loop below

    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    for batch in range(args["nBatch"]):
        model.train()

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Apply augmentations
        if augmentation_pipeline is not None:
            # New modular style
            X = augmentation_pipeline(X)
        else:
            # Old style (backward compatibility)
            if args.get("whiteNoiseSD", 0) > 0:
                X += torch.randn(X.shape, device=device) * args["whiteNoiseSD"]

            if args.get("constantOffsetSD", 0) > 0:
                X += (
                    torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                    * args["constantOffsetSD"]
                )

        # Compute prediction error
        pred = model.forward(X, dayIdx)

        # Calculate input lengths for CTC
        input_lengths = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
        
        # Apply loss function
        # Note: CTC expects logits in (seq_len, batch, n_classes) format
        logits = torch.permute(pred.log_softmax(2), [1, 0, 2])
        loss = loss_fn(logits, y, input_lengths, y_len)
        
        # Handle reduction (some losses return per-sample, some return scalar)
        if loss.dim() > 0:
            loss = torch.sum(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping (if specified)
        if args.get("grad_clip", None) is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args["grad_clip"])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )

                    pred = model.forward(X, testDayIdx)
                    # Calculate input lengths for CTC
                    input_lengths = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
                    # Apply loss function
                    logits = torch.permute(pred.log_softmax(2), [1, 0, 2])
                    loss = loss_fn(logits, y, input_lengths, y_len)
                    # Handle reduction
                    if loss.dim() > 0:
                        loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()
import numpy as np
import torch
from bsq.util import AverageMeter


def hit(gt_item, pred_items):
    return int(gt_item in pred_items)


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index+2))
    return 0


def metrics(model, test_loader, top_k, criterion, extractor=None):
    HR, NDCG = [], []
    test_loss = AverageMeter()
    first_batch_flag = True
    with torch.no_grad():
        for (user, item), label in test_loader:
            user = user.cpu()
            item = item.cpu()
            label = label.float().cpu()
            if first_batch_flag and extractor:
                extractor.begin_extractor()
            predictions = model(user, item)
            if first_batch_flag and extractor:
                extractor.end_extractor()
                first_batch_flag = False
            loss = criterion(predictions, label.reshape(-1))
            _, indices = torch.topk(predictions, top_k)
            recommends = torch.take(
                    item, indices).cpu().numpy().tolist()

            gt_item = item[0].item()
            HR.append(hit(gt_item, recommends))
            NDCG.append(ndcg(gt_item, recommends))
            test_loss.update(loss.item(), predictions.shape[0])

    return test_loss.avg, np.mean(HR)* 100, np.mean(NDCG) * 100

import torch

def compute_wa_ua(output, target, num_classes):
    """
    Computes Weighted Accuracy (WA) and Unweighted Accuracy (UA).

    Args:
        output (torch.Tensor): Prediction matrix of shape (batch_size, num_classes).
        target (torch.LongTensor): Ground truth labels of shape (batch_size).
        num_classes (int): Number of classes.

    Returns:
        tuple: WA (Weighted Accuracy), UA (Unweighted Accuracy)
    """
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    
    _, pred = output.topk(1, 1, True, True)
    pred = pred.view(-1)

    
    correct_per_class = torch.zeros(num_classes, device=output.device)
    total_per_class = torch.zeros(num_classes, device=output.device)

    for c in range(num_classes):
        mask = target == c
        correct_per_class[c] = (pred[mask] == c).sum()
        total_per_class[c] = mask.sum()

   
    wa = correct_per_class.sum() / batch_size * 100.0  

    
    ua = (correct_per_class / (total_per_class + 1e-6)).mean() * 100.0 

    return wa.item(), ua.item()

def compute_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res

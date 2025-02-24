import torch
import torch.nn.functional as F

def train_model_mc(model, loader, optimizer, device, mc_passes=5, lambda_unc=0.05, epochs=5):
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits_list = []
            for _ in range(mc_passes):
                out = model(xb)
                logits_list.append(out.unsqueeze(0))
            stacked = torch.cat(logits_list, dim=0)
            mean_logits = stacked.mean(dim=0)
            var_logits = stacked.var(dim=0)
            loss = criterion(mean_logits, yb) + lambda_unc * var_logits.mean()
            loss.backward()
            optimizer.step()

def evaluate_model_mc(model, loader, device, mc_passes=5):
    import torch.nn as nn
    model.train()
    all_labels, all_preds, all_probs, all_vars = [], [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits_list = []
            for _ in range(mc_passes):
                out = model(xb)
                logits_list.append(out.unsqueeze(0))
            stacked = torch.cat(logits_list, dim=0)
            mean_logits = stacked.mean(dim=0)
            var_logits = stacked.var(dim=0).mean(dim=1)
            probs = F.softmax(mean_logits, dim=1)
            preds = probs.argmax(dim=1)
            all_labels.extend(yb.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:,1].cpu().numpy().tolist())
            all_vars.extend(var_logits.cpu().numpy().tolist())
    return all_labels, all_preds, all_probs, all_vars

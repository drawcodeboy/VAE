
def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler, device):
    model.train()
    total_loss = []
    
    for batch_idx, (x, _) in enumerate(dataloader, start=1):
        optimizer.zero_grad()
        
        x = x.to(device)
        
        x_prime, mu, std = model(x)
        
        loss = loss_fn(x_prime, x, mu, std)
        
        total_loss.append(loss.item())
        
        loss.backward()
        optimizer.step()
        
        print(f"\rTraining: {100*batch_idx/len(dataloader):.2f}%, ELBO: {-sum(total_loss)/len(total_loss):.6f}, LR: {scheduler.get_last_lr()[0]:.6f}", end="")
    print()
    
    return sum(total_loss)/len(total_loss)
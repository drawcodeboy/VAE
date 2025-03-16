
def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler, device):
    model.train()
    total_loss = []
    first_loss = []
    second_loss = []
    
    for batch_idx, (x, _) in enumerate(dataloader, start=1):
        optimizer.zero_grad()
        
        x = x.to(device)
        
        x_prime, mu, std = model(x)
        
        loss, first, second = loss_fn(x_prime, x, mu, std)
        
        total_loss.append(loss.item())
        first_loss.append(first.item())
        second_loss.append(second.item())
        
        loss.backward()
        optimizer.step()
        
        print(f"\rTraining: {100*batch_idx/len(dataloader):.2f}%, ELBO: {sum(total_loss)/len(total_loss):.6f}, First: {sum(first_loss)/len(first_loss):.4f}, Second: {sum(second_loss)/len(second_loss):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}", end="")
    print()
    # scheduler.step(sum(total_loss)/len(total_loss))
    
    return sum(total_loss)/len(total_loss)
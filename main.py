from nanochat.dataloader import tokenizing_distributed_data_loader_with_state

if __name__ == "__main__":
    print("Starting dataloader")
    batches = tokenizing_distributed_data_loader_with_state(B=1, T=1, split="train", device="cpu")
    print("Dataloader started")
    for inputs, targets, state_dict in batches:
        print(inputs)
        print(targets.shape)
        print(state_dict)
        break
    print("Dataloader finished")
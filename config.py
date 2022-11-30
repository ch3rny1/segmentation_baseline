EPOCH_NUM = 2
LOSS_FN = torch.nn.BCEWithLogitsLoss()
OPTIMIZER = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
MODEL = UNet(n_channels=1, n_classes=1)

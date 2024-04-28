import os
import sys
import json
from early_stopping import EarlyStopping
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from tqdm import tqdm
import timm

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomRotation(degrees=None),
                                     transforms.RandomAffine(degrees=None),
                                     transforms.GaussianBlur(kernel_size=None),
                                     transforms.ColorJitter(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.2394,0.2421,0.2381], [0.1849, 0.28, 0.2698])]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.2394,0.2421,0.2381], [0.1849, 0.28, 0.2698])])}
    train_dataset = datasets.ImageFolder(root="trainset_root_path",
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    nasal_class_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in nasal_class_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 128
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root="valset_root_path",
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    net = timm.create_model('cait_s24_224', pretrained=True, num_classes=7)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=0.001, betas=(0.9, 0.999))

    epochs = 150
    train_steps = len(train_loader)
    early_stopping = EarlyStopping(verbose=True)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, running_loss / train_steps, val_accurate))

        # 调用早停逻辑
        early_stopping(val_accurate, net)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('Finished Training')


if __name__ == '__main__':
    main()
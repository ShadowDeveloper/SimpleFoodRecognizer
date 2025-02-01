import random
import time
full_log = ""
full_log += f"BEGIN LOG FOR train.py\nCURRENT TIME:\n{time.strftime('%d %B, %Y  %H:%M:%S', time.gmtime())} UTC\n{time.strftime('%d %B, %Y  %H:%M:%S', time.localtime())} Local Time (CT)\n\n"


def log(*msgs):
    global full_log
    for msg in msgs:
        print(msg)
        full_log += msg + "\n"

def save_log(PATH="model.log"):
    with open(PATH, 'w') as f:
        f.write(full_log)

log("Loading libraries and utilities...")
start = time.time_ns()

import torch
import torchvision
from datasets import load_dataset

log(f"Libraries and utilities loaded in {round((time.time_ns() - start) / 1000000000, 3)} s.")

log("Setting config...")
start = time.time_ns()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)
epochs = 10
lr = 0.01
batch_size = 128
step_size = 1
gamma = 0.9
random.seed(69) # this ensures that the images and labels are shuffled in the same way. also, nice.


log(f"Config set in {round((time.time_ns() - start) / 1000000, 3)} ms.")

log("Loading data...")
start = time.time_ns()

data_train = load_dataset("ethz/food101", split="train[:5000]") # only load the first 5000 images for now. once the model is working and I rent a GPU, I can load the full dataset bc apparently 12 GB of VRAM can't handle more than this 😭
data_test = load_dataset("ethz/food101", split="validation[:5000]") # for some reason the test split is called validation???

log(f"Data loaded in {round((time.time_ns() - start) / 1000000000, 3)} s.")

log("Preprocessing data...")

def preprocess(images):
    imgs = []
    l = len(images)
    comp = torchvision.transforms.Compose([ # center crop to 256x256 and convert it to a tensor
        torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor()
    ])
    for i in range(l):
        img = images[i]
        img = comp(img)
        imgs.append(img.to(device))
        if i % (l/10) == 0:
            print(f"Processed {i}/{l}") # printed, not logged bc it's only for debugging
    return imgs

def shuffle(a):
    return sorted(a, key=lambda x: random.random())

pre_shuffle_state = random.getstate() # see line 36

train_imgs = shuffle(preprocess(data_train["image"]))
random.setstate(pre_shuffle_state)
train_labels = shuffle(data_train["label"])
random.setstate(pre_shuffle_state)
test_imgs = shuffle(preprocess(data_test["image"]))
random.setstate(pre_shuffle_state)
test_labels = shuffle(data_test["label"])

log(f"Data preprocessed in {round((time.time_ns() - start) / 1000000000, 3)} s.")

log("Creating model...")

from model import CNN

model = CNN(batch_size).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

log("Model created.")

log("Training model...")
start = time.time_ns()

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for i in range(0, len(train_imgs), batch_size):
        inputs = torch.stack(train_imgs[i:i + batch_size])
        labels = torch.tensor(train_labels[i:i + batch_size]).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if epoch % epochs/2 == 0:
        #                                                              v ------------------------------------------------------- note that memory usage is in gibibytes (1 GiB = 1024^3 bytes) -------------------------------------------------------------- v
        log(f"Epoch {epoch + 1} | total loss = {round(total_loss, 4)} | {round((torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1073741824, 3)}/{round(torch.cuda.get_device_properties(device).total_memory / 1073741824, 3)} GiB of VRAM used")
    else:
        log(f"Epoch {epoch + 1} | total loss = {round(total_loss, 4)}")
log(f"Model trained in {round((time.time_ns() - start) / 1000000000, 3)} s.")

log("Testing model...")
start = time.time_ns()

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for i in range(0, len(test_imgs), batch_size):
        inputs = torch.stack(test_imgs[i:i + batch_size])
        labels = torch.tensor(test_labels[i:i + batch_size]).to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += sum([1 if predicted[x] == labels[x] else 0 for x in range(len(predicted))])

log(f"Model tested in {round((time.time_ns() - start) / 1000000000, 3)} s.")

log(f"Accuracy: {round(100 * correct / total, 4)}%")

log("Saving model...")

torch.save(model.state_dict(), "model.pth")

log("Model saved.")

full_log += f"END OF LOG FOR train.py\nCURRENT TIME:\n{time.strftime('%d %B, %Y  %H:%M:%S', time.gmtime())} UTC\n{time.strftime('%d %B, %Y  %H:%M:%S', time.localtime())} Local Time (CT)\n\n"

save_log()

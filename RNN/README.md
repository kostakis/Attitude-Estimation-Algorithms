# Pytorch C++ & RNN

# Installation

Download the zip from pytorch https://pytorch.org/get-started/locally/  
Use the C++/Java and the CPU version since are task are light so no gpu needed for now
Unzip the downloaded zip file into a directory of your choice and update the Makefile
Pytorch version verified to work: 1.12.1

`wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.12.1%2Bcpu.zip`

# Link folders

ln -s ../Results/ .
ln -s ../Data .
ln -s ../include .

900 -> Regularazation

1000 -> The original

# Train to google cloud

## SHH to the VM instance


## Update CPP files on the remote VM instance
gcloud compute scp RNN/*.cpp RNN/*.h instance-20260325-174124:~/project/RNN/ --zone "europe-west1-b"

## Should not change that much after initial copy
gcloud compute scp RNN/Makefile instance-20260325-174124:~/project/RNN/ --zone "europe-west1-b"
gcloud compute scp RNN/Dockerfile instance-20260325-174124:~/project/RNN/ --zone "europe-west1-b"
gcloud compute scp --recurse include/ instance-20260325-174124:~/project/ --zone "europe-west1-b"
gcloud compute scp --recurse Data/ instance-20260325-174124:~/project/ --zone "europe-west1-b"



# Docker

## Run docker and mount disk to save results

### Create folder to mount docker after
```bash
# Inside VM
mkdir -p ~/results
```

### Build image
```bash
docker build -f RNN/Dockerfile -t lstm-app .
```

### Run detached and mount
```bash
docker run -d --name lstm-train -v ~/results:/app/output lstm-app bash -c \
    "./lstm.out --epochs 500 2>&1 | tee /app/output/train.log && cp lstm_model_epoch_*.pt /app/output/"
```


### Evaluate model
```bash
docker run -v ~/results:/app/output lstm-app bash -c \
"./lstmEvalNoAttn.out /app/output/lstm_model_epoch_500.pt --save-all && cp Results/lstm_predictions.csv /app/output/"
```

### Copy locally

```bash
gcloud compute scp instance-20260325-174124:~/results/* ./RNN/ --zone="europe-west1-b"
```
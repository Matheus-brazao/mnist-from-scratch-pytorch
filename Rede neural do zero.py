import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import time
from torchvision import datasets, transforms
from torch import nn, optim 
transforms = transforms.ToTensor() #definindo a conversão de imagem para tensor

trainset = datasets.MNIST('./MNIST_data', download=True, train=True, transform=transforms) #CARREGA A PARTE DE TREINO DO DATASET
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True) #CRIA UM BUFFER PARA PEGAR OS DADOS POR PARTES

valset = datasets.MNIST('./MNIST_data', download=True, train=False, transform=transforms) #CARREGA A PARTE DE VALIDAÇÃO DO DATASET
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True) #CRIA UM BUFFER PARA PEGAR OS DADOS POR PARTES
datater = iter(trainloader)
imagens, etiquetas = datater.next()
plt.imshow(imagens[0].numpy().squeeze(), cmap='gray_r')
print(imagens[0].shape)   # para verificar as dimensões do tensor de cada imagem
print(etiquetas[0].shape) # para verificar as dimensões do tensor de cada etiqueta
class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.linear1 = nn.Linear(28*28, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return F.log_softmax(x, dim=1)
    def validacao(modelo, valloader, device):
        modelo.eval()  # desliga dropout/batch‑norm etc.
        conta_corretas, conta_todas = 0, 0

        with torch.no_grad():  # desativa o autograd para acelerar
            for imagens, etiquetas in valloader:
                # **flatten** de [batch,1,28,28] → [batch,784]
                imagens = imagens.view(imagens.size(0), -1).to(device)
                etiquetas = etiquetas.to(device)

                # saída em log‑probabilidades
                logps = modelo(imagens)
                # converte para probabilidades “normais”
                ps = torch.exp(logps)

                # pega o índice da maior probabilidade
                _, classe_pred = ps.max(dim=1)

                # conta acertos
                conta_corretas += (classe_pred == etiquetas).sum().item()
                conta_todas    += etiquetas.size(0)

        print(f"Total de imagens testadas = {conta_todas}")
        print(f"Precisão do modelo = {conta_corretas*100/conta_todas:.2f}%")
Modelo = Modelo()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #verifica se tem cuda na GPU, se tiver NVDIA ele usa, se não tiver ele usa a CPU
Modelo.to(device) #coloca o modelo na GPU ou CPU



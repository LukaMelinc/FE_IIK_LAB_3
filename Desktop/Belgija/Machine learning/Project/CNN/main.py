import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

# hyperparams
num_epochs = 10
lr = 0.01
num_classes = 12
num_channels = 5
BATCH_SIZE = 64

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model, testDataLoader, device):
    model.eval().to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testDataLoader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = (correct / total) * 100
    return acc

def data_loader(batch_size):
    train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train = 
    test = 

    train_size = int(0.8 * len(train))
    val_size = len(train) - train_size
    train_dataset, val_dataset = random_split(train, [train_size, val_size]) 

    trainDataLoader = DataLoader(train, batch_size=batch_size, shuffle=True)
    testDataLoader = DataLoader(test, batch_size=batch_size)
    valDataLoader = DataLoader(val_dataset, batch_size=batch_size)

    return trainDataLoader, valDataLoader, testDataLoader

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),  # Adjust input dimensions accordingly
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        """
        
        Struktura ResNet mreže.

        1 konvolucijska plast (k=7, s=2, p=3).
        4 plasti, sestavljene iz residualnih blokov, definiranih v zgornjem classu. Št. blokov v plasti se definira ob klicu classa.
        Nakoncu avgpool (se ne rabi) in fc plast 512->10 (št. razredov v CIFAR10).

        Vhodi:
            Tip bloka: Residualni blok
            Arrray plasti/blokov: array, ki določi, koliko blokov sestavlja posamezno plast
            Št. razredov: 10
        
        """
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3), # ven pride 16 x 16 x 64
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)   # ven pride 8 x 8 x 64
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)    # ohrani se dimenzija 8 x 8 x 64, se pa izvaja konvolucija
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)   # pride ven 4 x 4 x 128
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)   # pride ven 2 x 2 x 256
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)   # pride ven 1 x 1 x 512
        #self.avgpool = nn.AvgPool2d(7, stride=1)    #ne rabi bit kernel 7, ker je itak slika že 1x1
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        
        Sestavlja sekvence (residualnih) blokov v plasti mreže.
        Na podlagi velikosti maske, koraka (stride) in dodajanja (padding) plast spreminja ali ohranja dimenzijo obdelane slike.
        Tudi v primeru, da se dimenzije ne zmanjšajo in se samo izvaja konvolucija, se mreža uči.
        Blokom ni potrebno, da zmanjšujejo dimenzijo ali spremeniti število filtrov, takrat imajo stride/korak = 1 in so brez downsampla.

        Vhodi:
            block: Tip bloka(residual block)
            planes: št. filtrov
            blocks: število blokov v posamezni plasti
            stride: velikost koraka (v prvi konvolucijski plasti)
        
        """
        downsample = None
        if stride != 1 or self.inplanes != planes:  # downsample se nastavi, če se zmanjšujejo dimenzije slike/št. vhodnih kanalov se ne ujema s št. izhodnih
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = [] # prazen array za držanje residualnih blokov
        layers.append(block(self.inplanes, planes, stride, downsample))  # Residualni blok se ustvari in doda k layers
        self.inplanes = planes  # št. izhodnih filtrov se shrani v spremenljivko št. vhodnih filtrov

        for i in range(1, blocks):  # ustvarimo toliko layerjev, kolikor je elementov v arrayu, s katerim kličemo model
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)     

        # * je unpacking operator, odpakira elemente v listu layers, da so lahko podani kot individualni argumenti v nn.Sequential konstruktor
        # V pythonu se * uporablja pred listom/tuplom pri klicu funkcije, ko odpakira iterable in poda elemente kot ločene argumente

    def forward(self, x):   # skip connection
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        """

        Zgradba Residual bloka.

        Prva konvolucijska plast zmanjšuje dimenzijo glede na stride,
        druga plast izvaja konvolucijo, a ohrani dimenzijo slike.

        Vhodi:
            št. vhodnih kanalov
            št. izhodnih kanalov
            stride - korak
            downsample - skip connection

        Prva konvolucijska plast dejansko lahko spremeni dimenzijo obdelane slike, če je pri klicu funkcije stride > 1 (zmanjša dimenzijo slike).
        Če je stride = 1, se pri teh parametrih dimenzija slike ohrani in se izvaja samo konvolucija.
        Druga plast ima default vrednost stride = 1, torej ne spreminja velikosti slike.
        
        """
        self.conv1 = nn.Sequential(     
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),  
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU()
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels)
                    )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):   # skip connection
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample: # če je 1, gre delat skp connection
            residual = self.downsample(x)
        out += residual # doda se skip vrednost
        out = self.relu(out)
        return out

model = CNN()
criterion = nn.CrossEntropyLoss()
train_loader, val_loader, test_loader = data_loader(BATCH_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
#writer = SummaryWriter(f"runs/Cet_18_7/02/LR_{lr}")

# Log the model graph
sample_input = torch.rand((1, num_channels, 32, 32)).to(device)
#writer.add_graph(model, sample_input)

total_step = len(train_loader)

for epoch in range(num_epochs):
    model.train()
    epoch_losses = []
    epoch_accuracies = []

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / labels.size(0)
        
        epoch_losses.append(loss.item())
        epoch_accuracies.append(accuracy)

        # Log loss and accuracy
        #writer.add_scalar('Train/Loss', loss.item(), epoch * total_step + i)
        #writer.add_scalar('Train/Accuracy', accuracy, epoch * total_step + i)

        if (i + 1) % 100 == 0:
            print(f'LR: {lr}, Epoch: [{epoch+1}/{num_epochs}], Step: [{i+1}/{total_step}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

    # Validation accuracy
    val_accuracy = test(model, val_loader, device)
    test_accuracy = test(model, test_loader, device)
    #writer.add_scalar('Validation/Accuracy', val_accuracy, epoch)
    #writer.add_scalar('Test/Accuracy', test_accuracy, epoch)

    print(f'Epoch [{epoch+1}/{num_epochs}] Validation Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')




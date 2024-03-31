import csv
import nltk
import copy
import scipy
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import wandb

# nltk.download('punkt')
# nltk.download('stopwords')

# defining the Classifier class
class RNNTrainer:
    # defining the RNN classifier with torch
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()

            self.hidden_size = hidden_size
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            self.fc1 = nn.Linear(hidden_size, hidden_size//2)
            self.fc2 = nn.Linear(hidden_size//2, output_size)
            self.relu = nn.ReLU()

        def forward(self, x, ids):
            out, _ = self.rnn(x)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    # defining the NewsDataset class
    class NewsDataset(Dataset):
        def __init__(self, embeddings, pad_ids, labels):
            self.embeddings = embeddings
            self.pad_ids = pad_ids
            self.labels = labels

        def __len__(self):
            return len(self.embeddings)

        def __getitem__(self, idx):
            return self.embeddings[idx], self.pad_ids[idx], self.labels[idx]

    # defining the constructor
    def __init__(
        self,
        train_data=None,
        test_data=None,
        batch_size=1,
        embedding_dim=100,
        embedding_type='svd',
        context_size=2,
        hidden_size=128,
        epochs=10,
        save_model=False,
        load_model=False,
        model_path=None,
        logging=False,
        run_name=None):

        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.embedding_type = embedding_type
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.save_model = save_model
        self.load_model = load_model
        self.model_path = model_path
        self.logging = logging
        self.name = run_name

        if self.logging:
            wandb.init(project='news-classification', entity='rockingharsha71', name=self.name, reinit=True)
            wandb.config.update({
                'embedding_type': self.embedding_type,
                'embedding_dim': self.embedding_dim,
                'context_size': self.context_size,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'hidden_size': self.hidden_size
            })

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []

        self.prepare_data()
        self.create_model()
        for epoch in range(self.epochs):
            print('Epoch: ', epoch+1)
            self.train()
            self.test()
            wandb.log({
                'train_loss': self.train_losses[-1],
                'train_accuracy': self.train_accuracies[-1],
                'test_loss': self.test_losses[-1],
                'test_accuracy': self.test_accuracies[-1]
            })
        wandb.finish()
        print('Training Complete')
        # self.plot_train_loss_acc()

    # generating svd embeddings
    def gen_svd_embeddings(self):
        # making the co-occurance matrix for the words
        co_occurance_matrix = scipy.sparse.lil_matrix((self.vocab_size, self.vocab_size))

        # make the co-occurance matrix using the window size of 2
        window_size = self.context_size
        for description in tqdm.tqdm(self.train_data_tokenised):
            for i in range(len(description)):
                for j in range(i+1, min(i+window_size, len(description))):
                    a,b = self.word_index[description[i]], self.word_index[description[j]]
                    co_occurance_matrix[a,b] += 1
                    co_occurance_matrix[b,a] += 1

        co_occurance_matrix = co_occurance_matrix.tocsr()

        # svd on the co-occurance matrix (limited computation)
        u, s, vt = scipy.sparse.linalg.svds(co_occurance_matrix.astype(np.float64), k=self.embedding_dim)
        del co_occurance_matrix

        # make sure the most important dimensions are first
        s = s[::-1]
        u = u[:, ::-1]

        self.embeddings = u @ np.diag(np.sqrt(s))

    # generating word2vec embeddings
    def gen_word2vec_embeddings(self):
        pass

    # function to generate embeddings and initialise the dataset and dataloader
    def prepare_data(self):
        # tokenising the train data to get the individual words
        train_data_description = [row[1] for row in self.train_data]
        train_data_tokenised = [nltk.word_tokenize(description) for description in train_data_description]
        test_data_description = [row[1] for row in self.test_data]
        test_data_tokenised = [nltk.word_tokenize(description) for description in test_data_description]
        # tokenise the words based on '\\' as well
        train_data_tokenised = [[word.split('\\') for word in description] for description in train_data_tokenised]
        test_data_tokenised = [[word.split('\\') for word in description] for description in test_data_tokenised]

        # print all the elements in the list within the tokenised list
        train_data_temp = copy.deepcopy(train_data_tokenised)
        test_data_temp = copy.deepcopy(test_data_tokenised)
        train_data_tokenised = []
        test_data_tokenised = []
        for description in train_data_temp:
            train_data_tokenised.append([])
            for words in description:
                for word in words:
                    if word != '':
                        train_data_tokenised[-1].append(word)
        for description in test_data_temp:
            test_data_tokenised.append([])
            for words in description:
                for word in words:
                    if word != '':
                        test_data_tokenised[-1].append(word)

        # convert all the words to lower case
        self.train_data_tokenised = [[word.lower() for word in description] for description in train_data_tokenised]
        self.test_data_tokenised = [[word.lower() for word in description] for description in test_data_tokenised]

        # making the co-occurance matrix for the words
        vocab = list(set([word for description in self.train_data_tokenised for word in description]))
        # include the <oov> token to the vocab
        vocab.append('<oov>')
        self.vocab = sorted(vocab)
        self.vocab_size = len(vocab)

        # make a dictionary for the words and their index
        self.word_index = {word: i for i, word in enumerate(self.vocab)}

        if self.embedding_type == 'svd':
            self.gen_svd_embeddings()
        elif self.embedding_type == 'word2vec':
            self.gen_word2vec_embeddings()
        else:
            print('Invalid embedding type')
            return

        # replace all the words not in the vocab with <oov>
        for i in range(len(self.test_data_tokenised)):
            for j in range(len(self.test_data_tokenised[i])):
                if self.test_data_tokenised[i][j] not in self.word_index:
                    self.test_data_tokenised[i][j] = '<oov>'

        # convert the tokenised data to embeddings
        self.train_data_x = []
        for description in self.train_data_tokenised:
            embeddings = [self.embeddings[self.word_index[word]] for word in description]
            self.train_data_x.append(embeddings)
        self.test_data_x = []
        for description in self.test_data_tokenised:
            embeddings = [self.embeddings[self.word_index[word]] for word in description]
            self.test_data_x.append(embeddings)

        # find the lengths of all the sequences
        self.train_data_x_lens = [len(description) for description in self.train_data_tokenised]
        self.test_data_x_lens = [len(description) for description in self.test_data_tokenised]
        # find the 95th percentile of the lengths
        self.train_data_x_len = np.percentile(self.train_data_x_lens, 90)
        self.test_data_x_len = np.percentile(self.test_data_x_lens, 90)
        # pad the sequence
        self.train_data_x = nn.utils.rnn.pad_sequence([torch.tensor(np.array(embeddings), dtype=torch.float32) for embeddings in self.train_data_x], batch_first=True, padding_value=0)
        self.test_data_x = nn.utils.rnn.pad_sequence([torch.tensor(np.array(embeddings), dtype=torch.float32) for embeddings in self.test_data_x], batch_first=True, padding_value=0)
        # trim the sequences to the 90th percentile length
        self.train_data_x = self.train_data_x[:, :int(self.train_data_x_len)]
        self.test_data_x = self.test_data_x[:, :int(self.test_data_x_len)]
        # reduce lens to the truncated length
        self.train_data_x_lens = [min(self.train_data_x_lens[i], int(self.train_data_x_len)) for i in range(len(self.train_data_x_lens))]
        self.test_data_x_lens = [min(self.test_data_x_lens[i], int(self.test_data_x_len)) for i in range(len(self.test_data_x_lens))]

        train_labels = [row[0] for row in self.train_data]

        # one hot encoding the labels
        labels = list(set(train_labels))
        label_index = {label: i for i, label in enumerate(labels)}
        labels_onehot_train = np.zeros((len(self.train_data), len(labels)))
        labels_onehot_test = np.zeros((len(self.test_data), len(labels)))
        for i in range(len(self.train_data)):
            labels_onehot_train[i, label_index[self.train_data[i][0]]] = 1
        for i in range(len(self.test_data)):
            labels_onehot_test[i, label_index[self.test_data[i][0]]] = 1

        self.train_data_y = labels_onehot_train
        self.test_data_y = labels_onehot_test

        self.train_dataset = self.NewsDataset(self.train_data_x, self.train_data_x_lens, self.train_data_y)
        self.test_dataset = self.NewsDataset(self.test_data_x, self.test_data_x_lens, self.test_data_y)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def create_model(self):
        self.model = self.RNN(input_size=self.embedding_dim, hidden_size=self.hidden_size, output_size=self.train_data_y.shape[1]).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(tqdm.tqdm(self.train_loader)):
            inputs, ids, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs, ids)
            # sum the outputs
            outputs = torch.sum(outputs, dim=1)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            # correct += (torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).sum().item()
            # self.train_losses.append(loss.item())
            # self.train_accuracies.append(correct/total)
            acc = accuracy_score(torch.argmax(labels, dim=1).cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy())
            pre = precision_score(torch.argmax(labels, dim=1).cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy(), average='weighted', zero_division=0)
            rec = recall_score(torch.argmax(labels, dim=1).cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy(), average='weighted', zero_division=0)
            f1 = f1_score(torch.argmax(labels, dim=1).cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy(), average='weighted', zero_division=0)
            self.train_losses.append(loss.item())
            self.train_accuracies.append(acc)

        print('Training Loss: ', self.train_losses[-1])
        print('Training Accuracy: ', self.train_accuracies[-1])

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, ids, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs, ids)
                outputs = torch.sum(outputs, dim=1)
                # total += labels.size(0)
                # correct += (torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).sum().item()
                # self.test_losses.append(self.criterion(outputs, labels).item())
                # self.test_accuracies.append(correct/total)
                acc = accuracy_score(torch.argmax(labels, dim=1).cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy())
                pre = precision_score(torch.argmax(labels, dim=1).cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy(), average='weighted', zero_division=0)
                rec = recall_score(torch.argmax(labels, dim=1).cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy(), average='weighted', zero_division=0)
                f1 = f1_score(torch.argmax(labels, dim=1).cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy(), average='weighted', zero_division=0)
                self.test_losses.append(self.criterion(outputs, labels).item())
                self.test_accuracies.append(acc)

        print('Test Loss: ', self.test_losses[-1])
        print('Test Accuracy: ', self.test_accuracies[-1])

    def plot_train_loss_acc(self):
        plt.figure()
        plt.plot(self.train_losses, label='Train Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Train Loss')
        plt.legend()
        plt.show()

    def save_model(self):
        pass

    def load_model(self):
        pass

# main function
if __name__ == "__main__":
    # load the data
    train_data_path = './Data/train.csv'
    test_data_path = './Data/test.csv'

    train_data = None
    test_data = None
    with open(train_data_path, 'r') as file:
        reader = csv.reader(file)
        train_data = list(reader)
        train_data = train_data[1:] # remove header
    with open(test_data_path, 'r') as file:
        reader = csv.reader(file)
        test_data = list(reader)
        test_data = test_data[1:] # remove header

    # params = {
    #     'train_data':train_data,
    #     'test_data':test_data,
    #     'batch_size':512,
    #     'embedding_dim':200,
    #     'embedding_type':'svd',
    #     'context_size':6,
    #     'hidden_size':256,
    #     'epochs':15,
    #     'logging':True,
    #     'run_name':'svd_200_6_256'
    # }

    # rnn_trainer = RNNTrainer(**params)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    for context_size in [2, 4, 6, 8]:
        context_train_losses = []
        context_train_accuracies = []
        context_test_losses = []
        context_test_accuracies = []
        for i in range(5):
            params = {
                'train_data':train_data,
                'test_data':test_data,
                'batch_size':512,
                'embedding_dim':200,
                'embedding_type':'svd',
                'context_size':context_size,
                'hidden_size':256,
                'epochs':15,
                'logging':True,
                'run_name':f'svd_200_{context_size}_256_{i}'
            }

            rnn_trainer = RNNTrainer(**params)
            context_train_losses.append(rnn_trainer.train_losses)
            context_train_accuracies.append(rnn_trainer.train_accuracies)
            context_test_losses.append(rnn_trainer.test_losses)
            context_test_accuracies.append(rnn_trainer.test_accuracies)
            del rnn_trainer

        context_train_losses = np.array(context_train_losses)
        context_train_accuracies = np.array(context_train_accuracies)
        context_test_losses = np.array(context_test_losses)
        context_test_accuracies = np.array(context_test_accuracies)

        print(context_train_losses.shape)
        print(context_train_accuracies.shape)
        print(context_test_losses.shape)
        print(context_test_accuracies.shape)

        train_losses.append(context_train_losses)
        train_accuracies.append(context_train_accuracies)
        test_losses.append(context_test_losses)
        test_accuracies.append(context_test_accuracies)

    train_losses = np.array(train_losses)
    train_accuracies = np.array(train_accuracies)
    test_losses = np.array(test_losses)
    test_accuracies = np.array(test_accuracies)

    # save the losses and accuracies
    np.save('./Data/train_loss.npy', train_losses)
    np.save('./Data/train_acc.npy', train_accuracies)
    np.save('./Data/test_loss.npy', test_losses)
    np.save('./Data/test_acc.npy', test_accuracies)
    print('Data saved')

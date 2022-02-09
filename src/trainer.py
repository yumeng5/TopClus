from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from nltk.corpus import stopwords
import string
from transformers import BertTokenizer
from model import TopClusModel
import os
from tqdm import tqdm
import argparse
from sklearn.cluster import KMeans
from utils import TopClusUtils
import numpy as np


class TopClusTrainer(object):

    def __init__(self, args):
        self.args = args
        pretrained_lm = 'bert-base-uncased'
        self.n_clusters = args.n_clusters
        self.model = TopClusModel.from_pretrained(pretrained_lm,
                                                  output_attentions=False,
                                                  output_hidden_states=False,
                                                  input_dim=args.input_dim,
                                                  hidden_dims=eval(args.hidden_dims),
                                                  n_clusters=args.n_clusters,
                                                  kappa=args.kappa)
        self.utils = TopClusUtils()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = eval(args.hidden_dims)[-1]
        tokenizer = BertTokenizer.from_pretrained(pretrained_lm, do_lower_case=True)
        self.vocab = tokenizer.get_vocab()
        self.inv_vocab = {k:v for v, k in self.vocab.items()}
        self.filter_vocab()
        self.data_dir = os.path.join("datasets", args.dataset)
        self.utils.create_dataset(self.data_dir, "texts.txt", "text.pt")
        data = self.load_dataset(self.data_dir, "text.pt")
        input_ids = data["input_ids"]
        attention_masks = data["attention_masks"]
        valid_pos = data["valid_pos"]
        self.data = TensorDataset(input_ids, attention_masks, valid_pos)
        self.batch_size = args.batch_size
        self.res_dir = f"results_{args.dataset}"
        os.makedirs(self.res_dir, exist_ok=True)
        self.log_files = {}

    # invalid words to be filtered out from results
    def filter_vocab(self):
        stop_words = set(stopwords.words('english'))
        self.filter_idx = []
        for i in self.inv_vocab:
            token = self.inv_vocab[i]
            if token in stop_words or token.startswith('##') or len(token) <=2 \
               or token in string.punctuation or token.startswith('['):
                self.filter_idx.append(i)

    def load_dataset(self, dataset_dir, loader_name):
        loader_file = os.path.join(dataset_dir, loader_name)
        assert os.path.exists(loader_file)
        print(f"Loading encoded texts from {loader_file}")
        data = torch.load(loader_file)
        return data

    # pretrain autoencoder with reconstruction loss
    def pretrain(self, pretrain_epoch=20):
        pretrained_path = os.path.join(self.data_dir, "pretrained.pt")
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained model from {pretrained_path}")
            trainer.model.ae.load_state_dict(torch.load(pretrained_path))
        else:
            print(f"Pretraining autoencoder")
            sampler = RandomSampler(self.data)
            dataset_loader = DataLoader(self.data, sampler=sampler, batch_size=self.batch_size)
            model = self.model.to(self.device)
            model.eval()
            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr)
            for epoch in range(pretrain_epoch):
                total_loss = 0
                for batch_idx, batch in enumerate(tqdm(dataset_loader, desc=f"Epoch {epoch+1}/{pretrain_epoch}")):
                    optimizer.zero_grad()
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    max_len = attention_mask.sum(-1).max().item()
                    input_ids, attention_mask = tuple(t[:, :max_len] for t in (input_ids, attention_mask))
                    input_embs, output_embs = model(input_ids, attention_mask, pretrain=True)
                    loss = F.mse_loss(output_embs, input_embs)
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                print(f"epoch {epoch}: loss = {total_loss / (batch_idx+1):.4f}")
            torch.save(model.ae.state_dict(), pretrained_path)
            print(f"Pretrained model saved to {pretrained_path}")

    # initialize topic embeddings via K-Means clustering in the spherical latent space
    def cluster_init(self):
        latent_emb_path = os.path.join(self.data_dir, "init_latent_emb.pt")
        model = self.model.to(self.device)
        if os.path.exists(latent_emb_path) and os.path.exists(latent_emb_path):
            print(f"Loading initial latent embeddings from {latent_emb_path}")
            latent_embs, freq = torch.load(latent_emb_path)
        else:
            sampler = SequentialSampler(self.data)
            dataset_loader = DataLoader(self.data, sampler=sampler, batch_size=self.batch_size)
            model.eval()
            latent_embs = torch.zeros((len(self.vocab), self.latent_dim)).to(self.device)
            freq = torch.zeros(len(self.vocab), dtype=int).to(self.device)
            with torch.no_grad():
                for batch in tqdm(dataset_loader, desc="Obtaining initial latent embeddings"):
                    input_ids = batch[0].to(self.device)
                    attention_mask = batch[1].to(self.device)
                    valid_pos = batch[2].to(self.device)
                    max_len = attention_mask.sum(-1).max().item()
                    input_ids, attention_mask, valid_pos = tuple(t[:, :max_len] for t in (input_ids, attention_mask, valid_pos))
                    latent_emb = model.init_emb(input_ids, attention_mask, valid_pos)
                    valid_ids = input_ids[valid_pos != 0]
                    latent_embs.index_add_(0, valid_ids, latent_emb)
                    freq.index_add_(0, valid_ids, torch.ones_like(valid_ids))
            latent_embs = latent_embs[freq > 0].cpu()
            freq = freq[freq > 0].cpu()
            latent_embs = latent_embs / freq.unsqueeze(-1)
            print(f"Saving initial embeddings to {latent_emb_path}")
            torch.save((latent_embs, freq), latent_emb_path)

        print(f"Running K-Means for initialization")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.args.seed)
        kmeans.fit(latent_embs.numpy(), sample_weight=freq.numpy())
        model.topic_emb.data = torch.tensor(kmeans.cluster_centers_).to(self.device)

    # obtain topic discovery results and latent document embeddings for clustering
    def inference(self, topk=10, suffix=""):
        sampler = SequentialSampler(self.data)
        dataset_loader = DataLoader(self.data, sampler=sampler, batch_size=self.batch_size)
        model = self.model.to(self.device)
        model.eval()
        latent_doc_embs = []
        word_topic_sim = -1 * torch.ones((len(self.vocab), self.n_clusters))
        word_topic_sim_dict = defaultdict(list)
        with torch.no_grad():
            for batch in tqdm(dataset_loader, desc="Inference"):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                max_len = attention_mask.sum(-1).max().item()
                input_ids, attention_mask = tuple(t[:, :max_len] for t in (input_ids, attention_mask))
                latent_doc_emb, word_ids, sim = model.inference(input_ids, attention_mask)
                latent_doc_embs.append(latent_doc_emb.detach().cpu())
                for word_id, s in zip(word_ids, sim):
                    word_topic_sim_dict[word_id.item()].append(s.cpu().unsqueeze(0))
        for i in range(len(word_topic_sim)):
            if len(word_topic_sim_dict[i]) > 5:
                word_topic_sim[i] = torch.cat(word_topic_sim_dict[i], dim=0).mean(dim=0)
        word_topic_sim[self.filter_idx, :] = -1

        # better organized topic display
        topic_sim_mat = torch.matmul(model.topic_emb, model.topic_emb.t())
        cur_idx = torch.randint(len(topic_sim_mat), (1,))
        topic_file = open(os.path.join(self.res_dir, f"topics{suffix}.txt"), "w")
        for i in range(len(topic_sim_mat)):
            sort_idx = topic_sim_mat[cur_idx].argmax().cpu().numpy()
            _, top_idx = torch.topk(word_topic_sim[:, sort_idx], topk)
            result_string = []
            for idx in top_idx:
                result_string.append(f"{self.inv_vocab[idx.item()]}")
            topic_file.write(f"Topic {i}: {','.join(result_string)}\n")
            topic_sim_mat[:, sort_idx] = -1
            cur_idx = sort_idx

        latent_doc_embs = torch.cat(latent_doc_embs, dim=0)
        doc_emb_path = os.path.join(self.res_dir, "latent_doc_emb.pt")
        print(f"Saving document embeddings to {doc_emb_path}")
        torch.save(latent_doc_embs, doc_emb_path)

        # print(f"Evaluating document clustering with latent document embeddings")
        # for aspect in ['topic', 'location']:
        # # label_path = os.path.join(self.data_dir, f"labels.txt")
        #     print(f"aspect: {aspect}")
        #     label_path = os.path.join(self.data_dir, f"label_{aspect}.txt")
        #     self.utils.cluster_eval(label_path, doc_emb_path, self.args.seed)
        return 

    # compute target distribution for distinctive topic clustering
    def target_distribution(self, preds):
        targets = preds**2 / preds.sum(dim=0)
        targets = (targets.t() / targets.sum(dim=1)).t()
        return targets

    # train model with three objectives
    def clustering(self, epochs=20):
        self.pretrain(pretrain_epoch=self.args.pretrain_epoch)
        self.cluster_init()
        sampler = RandomSampler(self.data)
        dataset_loader = DataLoader(self.data, sampler=sampler, batch_size=self.batch_size)
        model = self.model.to(self.device)
        model.eval()
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.lr)
        for epoch in range(epochs):
            total_rec_loss = 0
            total_rec_doc_loss = 0
            total_clus_loss = 0
            for batch_idx, batch in enumerate(tqdm(dataset_loader, desc=f"Clustering epoch {epoch+1}/{epochs}")):
                optimizer.zero_grad()
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                valid_pos = batch[2].to(self.device)
                max_len = attention_mask.sum(-1).max().item()
                input_ids, attention_mask, valid_pos = tuple(t[:, :max_len] for t in (input_ids, attention_mask, valid_pos))
                doc_emb, input_embs, output_embs, rec_doc_emb, p_word = model(input_ids, attention_mask, valid_pos)
                rec_loss = F.mse_loss(output_embs, input_embs)
                rec_doc_loss = F.mse_loss(rec_doc_emb, doc_emb)
                targets = self.target_distribution(p_word).detach()
                clus_loss = F.kl_div(p_word.log(), targets, reduction='batchmean')
                loss = rec_loss + rec_doc_loss + self.args.cluster_weight * clus_loss
                total_rec_loss += rec_loss.item()
                total_clus_loss += clus_loss.item()
                total_rec_doc_loss += rec_doc_loss.item()
                loss.backward()
                optimizer.step()
            if (epoch+1) % 10 == 0 and self.args.do_inference:
                self.inference(topk=self.args.k, suffix=f"_{epoch}")
            print(f"epoch {epoch+1}: rec_loss = {total_rec_loss / (batch_idx+1):.4f}; rec_doc_loss = {total_rec_doc_loss / (batch_idx+1):.4f}; cluster_loss = {total_clus_loss / (batch_idx+1):.4f}")

        model_path = os.path.join(self.data_dir, "model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"model saved to {model_path}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='yelp')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_clusters', default=100, type=int, help='number of topics')
    parser.add_argument('--k', default=10, type=int, help='number of top words to display per topic')
    parser.add_argument('--input_dim', default=768, type=int, help='embedding dimention of pretrained language model')
    parser.add_argument('--pretrain_epoch', default=20, type=int, help='number of epochs for pretraining autoencoder')
    parser.add_argument('--kappa', default=10, type=float, help='concentration parameter kappa')
    parser.add_argument('--hidden_dims', default='[500, 500, 1000, 100]', type=str)
    parser.add_argument('--do_cluster', action='store_true')
    parser.add_argument('--do_inference', action='store_true')
    parser.add_argument('--cluster_weight', default=0.1, type=float, help='weight of clustering loss')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs for clustering')

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer = TopClusTrainer(args)
    
    if args.do_cluster:
        trainer.clustering(epochs=args.epochs)
    if args.do_inference:
        model_path = os.path.join("datasets", args.dataset, "model.pt")
        try:
            trainer.model.load_state_dict(torch.load(model_path))
        except:
            print("No model found! Run clustering first!")
            exit(-1)
        trainer.inference(topk=args.k, suffix=f"_final")

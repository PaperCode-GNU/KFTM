import torch
import torch.nn as nn
import logging
import warnings
import os
import random
import torch.optim as optim
import numpy as np
import argparse
# import config

from tqdm import tqdm
from sklearn import metrics
from data_loader import load_data, load_law_text, LkdfDataset, load_charge_text
from model import LKDF
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertTokenizer

class Trainer(object):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
    
    def computer_metrics(self, y_true, y_pred):
        accuracy = metrics.accuracy_score(y_true, y_pred)
        macro_recall = metrics.recall_score(y_true, y_pred, average='macro')
        macro_precision = metrics.precision_score(y_true, y_pred, average='macro')
        macro_f1 = metrics.f1_score(y_true, y_pred, average='macro')

        return accuracy, macro_recall, macro_precision, macro_f1
    
    def computer_loss(self, output_law, output_accu, output_term, law_labels, accu_labels, term_labels):
        law_loss = nn.CrossEntropyLoss()(output_law, law_labels)
        #mid_law_loss = nn.CrossEntropyLoss()(mid_law, law_labels)
        accu_loss = nn.CrossEntropyLoss()(output_accu, accu_labels)
        term_loss = nn.CrossEntropyLoss()(output_term, term_labels)

        return law_loss, accu_loss, term_loss#, mid_law_loss
    
    def computer_prediction(self, output_law, output_accu, output_term):
        law_pred = nn.functional.softmax(output_law, dim = 1)
        accu_pred = nn.functional.softmax(output_accu, dim = 1)
        term_pred = nn.functional.softmax(output_term, dim = 1)

        res_law = torch.argmax(law_pred, dim = 1)
        res_accu = torch.argmax(accu_pred, dim = 1)
        res_term = torch.argmax(term_pred, dim = 1)

        return res_law, res_accu, res_term
    
    def train_epoch(self, train_loader, all_law_text, all_charge_text, model, optimizer, scheduler, epoch):
        total_loss = 0
        model.train()

        law_input_ids, law_attention_mask = all_law_text
        charge_input_ids, charge_attention_mask = all_charge_text
        for batch_idx, batch_samples in enumerate(train_loader):
            batch_fact_text, batch_fact_attention, batch_law_labels, batch_accu_labels, batch_term_labels = batch_samples
            output_law, output_accu, output_term = model(batch_fact_text, batch_fact_attention, law_input_ids, law_attention_mask, charge_input_ids, charge_attention_mask, batch_law_labels, batch_accu_labels, "train")
            law_loss, accu_loss, term_loss = self.computer_loss(output_law, output_accu, output_term, batch_law_labels, batch_accu_labels, batch_term_labels)
            loss_total = (law_loss + accu_loss + term_loss)/3 # + mid_law_loss
            total_loss += loss_total.item()
            optimizer.zero_grad()
            loss_total.backward()
            nn.utils.clip_grad_norm_(parameters = model.parameters(), max_norm = self.config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if batch_idx % 250 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:.4f}'.format(epoch, batch_idx, loss_total.item()))
        logging.info("-----------Go to evaluation-----------")
    
    def evaluate(self, valid_loader, model, all_law_text, all_charge_text):
        model.eval()

        predic_law, predic_accu, predic_term = [], [], []
        y_law, y_accu, y_term = [], [], []

        with torch.no_grad():
            law_input_ids, law_attention_mask = all_law_text
            charge_input_ids, charge_attention_mask = all_charge_text
            for batch_idx, batch_samples in enumerate(tqdm(valid_loader)):
                batch_fact_text, batch_fact_attention, batch_law_labels, batch_accu_labels, batch_term_labels = batch_samples
                output_law, output_accu, output_term = model(batch_fact_text, batch_fact_attention, law_input_ids, law_attention_mask, charge_input_ids, charge_attention_mask, batch_law_labels, batch_accu_labels, "eval")

                law_predic, accu_predic, term_predic = self.computer_prediction(output_law, output_accu, output_term)

                y_law.extend(batch_law_labels.cpu().numpy())
                y_accu.extend(batch_accu_labels.cpu().numpy())
                y_term.extend(batch_term_labels.cpu().numpy())

                law_prediction = law_predic.cpu().numpy()
                accu_prediction = accu_predic.cpu().numpy()
                term_prediction = term_predic.cpu().numpy()

                predic_law.extend(law_prediction)
                predic_accu.extend(accu_prediction)
                predic_term.extend(term_prediction)

        metrics = []
        accuracy_law, macro_recall_law, macro_precision_law, macro_f1_law = self.computer_metrics(y_law, predic_law)
        accuracy_accu, macro_recall_accu, macro_precision_accu, macro_f1_accu = self.computer_metrics(y_accu, predic_accu)
        accuracy_term, macro_recall_term, macro_precision_term, macro_f1_term = self.computer_metrics(y_term, predic_term)

        metrics.append((accuracy_law, macro_recall_law, macro_precision_law, macro_f1_law))
        metrics.append((accuracy_accu, macro_recall_accu, macro_precision_accu, macro_f1_accu))
        metrics.append((accuracy_term, macro_recall_term, macro_precision_term, macro_f1_term))

        return metrics
    
    def train(self, train_loader, valid_loader, all_law_text, all_charge_text, model, optimizer, scheduler):
        best_valid_accuracy = 0.0
        patience = 0
        task = ["law", "accu", "term"]

        for epoch in range(1, self.config.num_epochs + 1):
            self.train_epoch(train_loader, all_law_text, all_charge_text, model, optimizer, scheduler, epoch)
            valid_metrics = self.evaluate(valid_loader, model, all_law_text, all_charge_text)
            accuracy = []
            macro_recall = []
            macro_precision = []
            macro_f1 = []

            for i in range(len(task)):
                accuracy.append(valid_metrics[i][0])
                macro_recall.append(valid_metrics[i][1])
                macro_precision.append(valid_metrics[i][2])
                macro_f1.append(valid_metrics[i][3])
            logging.info("---------Model valid result----------")
            for j in range(len(task)):
                logging.info("Task: {}, Accuracy: {:.4f}, Macro Recall: {:.4f}, Macro Precision: {:.4f}, Macro F1: {:.4f}".format(task[j], accuracy[j], macro_recall[j], macro_precision[j], macro_f1[j]))
            
            # if accuracy[0] > best_valid_accuracy:
            #     best_valid_accuracy = accuracy[0]
            #     torch.save(model.state_dict(), self.config.model_save_path)
            #     logging.info("-------Best Model saved at epoch {}-------".format(epoch))
            #     patience = 0
            # else:
            #     patience += 1

            torch.save(model.state_dict(), os.path.join(self.config.model_save_path, 'model_epoch_{}.pth'.format(epoch)))
            logging.info("-------Model saved at epoch {}-------".format(epoch))

            if (patience > self.config.patience and epoch > self.config.min_epochs) or epoch == self.config.num_epochs:
                logging.info("-------Stop train at epoch {}-------".format(epoch))
                break

class Run(object):
    def __init__(self, config) -> None:
        super(Run, self).__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-lert-base", do_lower_case=True)
        self.model = LKDF(config)
        self.trainer = Trainer(config)

    def set_seed(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def set_logger(self, log_path):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)

            # Logging to console
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(stream_handler)
    
    def load_law_token(self):
        law_token = []
        all_law_text = load_law_text(self.config.law_text_path)
        for law_text in all_law_text:
            law_text_tokens = self.tokenizer(law_text)
            law_text_input_ids = law_text_tokens["input_ids"]
            law_text_attention_mask = law_text_tokens["attention_mask"]
            law_token.append((law_text_input_ids, law_text_attention_mask))
        
        batch_len = len(law_token)
        # max_len = min(max([len(item[0]) for item in law_token]), 512)
        max_len = 256

        pad_law_input_ids = 0 * np.ones((batch_len, max_len))
        pad_law_attention_mask = 0 * np.ones((batch_len, max_len))

        for i in range(batch_len):
            cur_len = len(law_token[i][0])
            if cur_len <= max_len:
                pad_law_input_ids[i][:cur_len] = law_token[i][0]
                pad_law_attention_mask[i][:cur_len] = law_token[i][1]
            else:
                pad_law_input_ids[i] = law_token[i][0][:max_len]
                pad_law_attention_mask[i] = law_token[i][1][:max_len]
        
        law_tokens = torch.tensor(pad_law_input_ids, dtype=torch.long).to("cuda")
        law_attention_mask = torch.tensor(pad_law_attention_mask, dtype=torch.long).to("cuda")
        return [law_tokens, law_attention_mask]
    
    def load_charge_token(self):
        charge_token = []
        all_charge_text = load_charge_text(self.config.charge_text_path)
        for charge_text in all_charge_text:
            charge_text_tokens = self.tokenizer(charge_text)
            charge_text_input_ids = charge_text_tokens["input_ids"]
            charge_text_attention_mask = charge_text_tokens["attention_mask"]
            charge_token.append((charge_text_input_ids, charge_text_attention_mask))
        
        batch_len = len(charge_token)
        # max_len = min(max([len(item[0]) for item in charge_token]), 512)
        max_len = 256

        pad_charge_input_ids = 0 * np.ones((batch_len, max_len))
        pad_charge_attention_mask = 0 * np.ones((batch_len, max_len))

        for i in range(batch_len):
            cur_len = len(charge_token[i][0])
            if cur_len <= max_len:
                pad_charge_input_ids[i][:cur_len] = charge_token[i][0]
                pad_charge_attention_mask[i][:cur_len] = charge_token[i][1]
            else:
                pad_charge_input_ids[i] = charge_token[i][0][:max_len]
                pad_charge_attention_mask[i] = charge_token[i][1][:max_len]
        
        charge_tokens = torch.tensor(pad_charge_input_ids, dtype=torch.long).to("cuda")
        charge_attention_mask = torch.tensor(pad_charge_attention_mask, dtype=torch.long).to("cuda")
        return [charge_tokens, charge_attention_mask]

    def test(self):
        self.set_logger(self.config.log_path)
        task = ["law", "accu", "term"]

        fact_text, law_labels_test, accy_labels_test, term_labels_test = load_data(self.config.test_data_path)
        logging.info("-------Test data loaded-------")
        test_dataset = LkdfDataset(fact_text, law_labels_test, accy_labels_test, term_labels_test)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0, collate_fn=test_dataset.collate_fn, drop_last=True)
        logging.info("-------Test dataloader loaded-------")
        all_law_text = self.load_law_token()
        # all_charge_text = self.load_charge_token()

        self.model.to("cuda")
        for num_epoch in range(1, self.config.num_epochs + 1):
            self.model.load_state_dict(torch.load(os.path.join(self.config.model_save_path, 'model_epoch_{}.pth'.format(num_epoch))))
            self.model.to("cuda")

            logging.info("-------Epoch_{} Test metrics-------".format(num_epoch))
            test_metrics = self.trainer.evaluate(test_loader, self.model, all_law_text)
            accuracy = []
            macro_recall = []
            macro_precision = []
            macro_f1 = []

            for i in range(len(task)):
                accuracy.append(test_metrics[i][0])
                macro_recall.append(test_metrics[i][1])
                macro_precision.append(test_metrics[i][2])
                macro_f1.append(test_metrics[i][3])
            for j in range(len(task)):
                logging.info("Epoch: {}, Task: {}, Accuracy: {:.4f}, Macro Recall: {:.4f}, Macro Precision: {:.4f}, Macro F1: {:.4f}".format(num_epoch, task[j], accuracy[j], macro_recall[j], macro_precision[j], macro_f1[j]))
        
        # self.model.load_state_dict(torch.load(os.path.join(self.config.model_save_path, 'model_epoch_{}.pth'.format(11))))
        # self.model.to("cuda")

        # logging.info("-------Epoch_{} Test metrics-------".format(11))
        # test_metrics = self.trainer.evaluate(test_loader, self.model, all_law_text)
        # accuracy = []
        # macro_recall = []
        # macro_precision = []
        # macro_f1 = []

        # for i in range(len(task)):
        #     accuracy.append(test_metrics[i][0])
        #     macro_recall.append(test_metrics[i][1])
        #     macro_precision.append(test_metrics[i][2])
        #     macro_f1.append(test_metrics[i][3])
        # for j in range(len(task)):
        #     logging.info("Epoch: {}, Task: {}, Accuracy: {:.4f}, Macro Recall: {:.4f}, Macro Precision: {:.4f}, Macro F1: {:.4f}".format(11, task[j], accuracy[j], macro_recall[j], macro_precision[j], macro_f1[j]))
        # logging.info("-------Test finished-------")

    def runing(self):
        self.set_logger(self.config.log_path)
        logging.info("-------Train log in the: {}".format(self.config.log_path))
        logging.info("-------Train in the device: {}".format("cuda"))

        fact_train, law_labels_train, accu_labels_train, term_labels_train = load_data(self.config.train_data_path)
        logging.info("-------Train data loaded-------")
        train_dataset = LkdfDataset(fact_train, law_labels_train, accu_labels_train, term_labels_train)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=0, collate_fn=train_dataset.collate_fn, drop_last=True)
        logging.info("-------Train dataloader loaded-------")

        fact_valid, law_labels_valid, accu_labels_valid, term_labels_valid = load_data(self.config.valid_data_path)
        logging.info("-------Valid data loaded-------")
        valid_dataset = LkdfDataset(fact_valid, law_labels_valid, accu_labels_valid, term_labels_valid)
        valid_loader = DataLoader(valid_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=0, collate_fn=valid_dataset.collate_fn, drop_last=True)
        logging.info("-------Valid dataloader loaded-------")

        train_size = len(train_dataset)
        valid_size = len(valid_dataset)
        logging.info("-------Train size: {}, Valid size: {}-------".format(train_size // self.config.batch_size, valid_size // self.config.batch_size))

        all_law_text = self.load_law_token()
        all_charge_text = self.load_charge_token()
        logging.info("-------Law and Charge text loaded-------")

        self.model.to("cuda")
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        train_steps_per_epoch = train_size // self.config.batch_size
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps = (self.config.num_epochs // 10) * train_steps_per_epoch,
                                                    num_training_steps = self.config.num_epochs * train_steps_per_epoch)
        logging.info("-------Start Train-------")
        self.trainer.train(train_loader, valid_loader, all_law_text, all_charge_text, self.model, optimizer, scheduler)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_save_path", default="./output_big/")
    parser.add_argument("--train_data_path", default="../data/new_data_big/Rtrain.json")
    parser.add_argument("--valid_data_path", default="../data/new_data_big/Rvalid.json")
    parser.add_argument("--test_data_path", default="../data/new_data_big/Rtest.json")
    parser.add_argument("--law_text_path", default="../data/new_data_big/law.txt")
    parser.add_argument("--charge_text_path", default="./data/new_data_big/charge_interpret.txt")
    parser.add_argument("--log_path", default="./log.log")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--num_epochs", default=16, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--min_epochs", default=5, type=int)
    parser.add_argument("--num_law", default=118, type=int)
    parser.add_argument("--num_accu", default=130, type=int)
    # parser.add_argument("--num_law", default=103, type=int)
    # parser.add_argument("--num_accu", default=119, type=int)
    parser.add_argument("--num_term", default=11, type=int)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--max_grad_norm", default=5, type=int)

    config = parser.parse_args()
    # model = LKDF(config)
    # model.to("cuda")
    # model.load_state_dict(torch.load(os.path.join(config.model_save_path, 'model_epoch_{}.pth'.format(8))))
    # total_params = sum(p.numel() for p in model.parameters())
    # print("Total parameters:", total_params)

    run = Run(config)
    run.set_seed(42)
    # run.runing()
    run.test()
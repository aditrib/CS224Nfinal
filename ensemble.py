import argparse
import torch
import numpy as np
from torch import nn
from multitask_bert import MultitaskBERT, MultitaskBERTDualityOfMan
from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)
from evaluation import model_eval_sst, model_eval_para, model_eval_sts, model_eval_multitask, model_eval_test_multitask
from torch.utils.data import DataLoader

class EnsembledModel(nn.Module):
    def __init__(self, models):
        super(EnsembledModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, input_ids, attention_mask):
        raise NotImplementedError("EnsembledModel does not support the forward method directly. Use predict_sentiment, predict_paraphrase, or predict_similarity methods.")

    def predict_sentiment(self, input_ids, attention_mask):
        with torch.no_grad():
            logits = torch.stack([model.predict_sentiment(input_ids, attention_mask) for model in self.models], dim=0)
            return torch.round(torch.mean(logits, dim=0)).int()

    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        with torch.no_grad():
            logits = torch.stack([model.predict_paraphrase(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2) for model in self.models], dim=0)
            return torch.round(torch.mean(logits, dim=0)).int()

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        with torch.no_grad():
            logits = torch.stack([model.predict_similarity(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2).unsqueeze(-1) for model in self.models], dim=0)
            return torch.mean(logits, dim=0).squeeze()


def load_models(filepaths, device):
    models = []
    for filepath in filepaths:
        print(f"====== Loading Model: {filepath} ======")
        saved = torch.load(filepath, map_location=device)
        config = saved['model_config']
        if "duality" in args.filepath:
            model = MultitaskBERTDualityOfMan(config)
        else:
            model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        model.eval()
        models.append(model)
    return models

def test_multitask(args, model, benchmark=False):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')

        print(f"Device Set: {device}\n")

        
        model = model.to(device)

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test', extra_clean = args.extra_clean)

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev', extra_clean = args.extra_clean)

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)
        
        print(f"\n============= Evaluating Dev Scores (SST, Para, STS) =============\n")

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)
        
        if not benchmark:
            print(f"\n============= Evaluating Test Scores (SST, Para, STS) =============\n")

            test_sst_y_pred, \
                test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                    model_eval_test_multitask(sst_test_dataloader,
                                            para_test_dataloader,
                                            sts_test_dataloader, model, device)

            with open(args.sst_dev_out, "w+") as f:
                print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
                f.write(f"id \t Predicted_Sentiment \n")
                for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                    f.write(f"{p} , {s} \n")
            print(f"Wrote SST dev results to: {args.sst_dev_out}")


            with open(args.sst_test_out, "w+") as f:
                f.write(f"id \t Predicted_Sentiment \n")
                for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                    f.write(f"{p} , {s} \n")
            print(f"Wrote SST test results to : {args.sst_test_out}")
            
            with open(args.para_dev_out, "w+") as f:
                print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
                f.write(f"id \t Predicted_Is_Paraphrase \n")
                for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                    f.write(f"{p} , {s} \n")
            print(f"Wrote Paraphrase dev results to: {args.para_dev_out}")
            
            with open(args.para_test_out, "w+") as f:
                f.write(f"id \t Predicted_Is_Paraphrase \n")
                for p, s in zip(test_para_sent_ids, test_para_y_pred):
                    f.write(f"{p} , {s} \n")
            print(f"Wrote Paraphrase test results to: {args.para_test_out}")        
            
            with open(args.sts_dev_out, "w+") as f:
                print(f"dev sts corr :: {dev_sts_corr :.3f}")
                f.write(f"id \t Predicted_Similiary \n")
                for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                    f.write(f"{p} , {s} \n")
            print(f"Wrote STS dev results to: {args.sts_dev_out}")

            with open(args.sts_test_out, "w+") as f:
                f.write(f"id \t Predicted_Similiary \n")
                for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                    f.write(f"{p} , {s} \n")
            print(f"Wrote STS test results to: {args.sts_test_out}")
            
        else:
            return dev_sentiment_accuracy, dev_paraphrase_accuracy, dev_sts_corr

def main(args):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    print(f"Device Set: {device}\n")

    model_list = load_models(args.filepaths, device)
    ensemble_model = EnsembledModel(model_list).to(device)

    test_multitask(args, model=ensemble_model)

if __name__ == "__main__":
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--extra_clean", action='store_true', default=False, help="Perform extra nltk etc. data cleaning.")
        parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
        parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")
        parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
        parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")
        parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
        parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")
        parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
        parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")
        parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
        parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")
        parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
        parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument('--filepaths', nargs='+', required=True, help='List of .pt files')
        parser.add_argument("--use_gpu", action='store_true')

        args = parser.parse_args()
        return args

    args = get_args()
    main(args)




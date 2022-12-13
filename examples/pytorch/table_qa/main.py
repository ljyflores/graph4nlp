import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from graph4nlp.pytorch.data.dataset import Table2TextDataset
from graph4nlp.pytorch.models.graph2seq import Graph2Seq
from graph4nlp.pytorch.models.graph2seq_loss import Graph2SeqLoss
from graph4nlp.pytorch.modules.utils.copy_utils import prepare_ext_vocab

from args import get_args
from build_model import get_model
from evaluation import ExpressionAccuracy
from utils import get_log, wordid2str

def tokenize_jobs(str_input):
    return str_input.strip().split()

class TableQA:
    def __init__(self, opt):
        super(TableQA, self).__init__()
        self.opt = opt
        self.use_copy = self.opt["decoder_args"]["rnn_decoder_share"]["use_copy"]
        self.use_coverage = self.opt["decoder_args"]["rnn_decoder_share"]["use_coverage"]
        self._build_device(self.opt)
        self._build_logger(self.opt["log_file"])
        self._build_dataloader()
        self._build_model()
        self._build_optimizer()
        self._build_evaluation()
        self._build_loss_function()

    def _build_device(self, opt):
        seed = opt["seed"]
        np.random.seed(seed)
        if opt["use_gpu"] != 0 and torch.cuda.is_available():
            print("[ Using CUDA ]")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            from torch.backends import cudnn

            cudnn.benchmark = True
            # device = torch.device("cuda" if opt["gpu"] < 0 else "cuda:%d" % opt["gpu"])
            device = torch.device("cuda:0")
        else:
            print("[ Using CPU ]")
            device = torch.device("cpu")
        self.device = device

    def _build_logger(self, log_file):
        import os

        log_folder = os.path.split(log_file)[0]
        if not os.path.exists(log_file):
            os.makedirs(log_folder)
        self.logger = get_log(log_file)

    def _build_dataloader(self):
        dataset = Table2TextDataset(
            root_dir                 = 'bothWikiSQL' if self.opt["dataset"]=='WikiSQL' else 'bothWTQ',
            topology_subdir          = 'TableGraph',
            share_vocab              = True,
            tokenizer                = tokenize_jobs,
            pretrained_word_emb_name = "6B",
            pretrained_word_emb_url  = None,
            pretrained_word_emb_cache_dir = None,
            merge_strategy = 'tailhead',
            edge_strategy  = 'heterogeneous',
            seed = None,
            word_emb_size = 300,
            # dynamic_init_graph_name=self.opt["graph_construction_args"][
            #     "graph_construction_private"
            # ].get("dynamic_init_graph_type", None),
            thread_number = 1,
            port = 9000,
        )

        # self.train_table_dataloader = DataLoader(
        #     dataset.train_table,
        #     batch_size=4,
        #     shuffle=True,
        #     num_workers=self.opt["num_works"],
        #     collate_fn=dataset.collate_fn,
        # )
        # self.train_sql_dataloader = DataLoader(
        #     dataset.train_sql,
        #     batch_size=4,
        #     shuffle=True,
        #     num_workers=self.opt["num_works"],
        #     collate_fn=dataset.collate_fn,
        # )
        self.test_dataloader = DataLoader(
            dataset.test,
            batch_size=1,
            shuffle=False,
            num_workers=self.opt["num_works"],
            collate_fn=dataset.collate_fn,
        )
        self.dev1_dataloader = DataLoader(
            dataset.dev1,
            batch_size=1,
            shuffle=False,
            num_workers=self.opt["num_works"],
            collate_fn=dataset.collate_fn,
        )
        self.dev2_dataloader = DataLoader(
            dataset.dev2,
            batch_size=1,
            shuffle=False,
            num_workers=self.opt["num_works"],
            collate_fn=dataset.collate_fn,
        )
        self.dev3_dataloader = DataLoader(
            dataset.dev3,
            batch_size=1,
            shuffle=False,
            num_workers=self.opt["num_works"],
            collate_fn=dataset.collate_fn,
        )
        self.perturb_test_orig_dataloader = DataLoader(
            dataset.perturb_test_orig,
            batch_size=1,
            shuffle=False,
            num_workers=self.opt["num_works"],
            collate_fn=dataset.collate_fn,
        )
        self.perturb_test_changed_dataloader = DataLoader(
            dataset.perturb_test_changed,
            batch_size=1,
            shuffle=False,
            num_workers=self.opt["num_works"],
            collate_fn=dataset.collate_fn,
        )
        self.perturb_dev_orig_dataloader = DataLoader(
            dataset.perturb_dev_orig,
            batch_size=1,
            shuffle=False,
            num_workers=self.opt["num_works"],
            collate_fn=dataset.collate_fn,
        )
        self.perturb_dev_changed_dataloader = DataLoader(
            dataset.perturb_dev_changed,
            batch_size=1,
            shuffle=False,
            num_workers=self.opt["num_works"],
            collate_fn=dataset.collate_fn,
        )        
        self.vocab = dataset.vocab_model

    def _build_model(self):
        if self.opt["mode"]=='train':
            self.model = get_model(self.opt, vocab_model=self.vocab).to(self.device)
        else:
            print("loading model")
            self.model = Graph2Seq.load_checkpoint(self.opt["checkpoint_save_path"],
                                                   "best_w2v_wikisql.pt" if self.opt["dataset"]=="WikiSQL"\
                                                   else "best_w2v_wtq.pt").to(
                self.device
            )
            print("Loaded pretrained model!")
        
    def _build_optimizer(self):
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(parameters, lr=self.opt["learning_rate"])

    def _build_evaluation(self):
        self.metrics = [ExpressionAccuracy()]

    def _build_loss_function(self):
        self.loss = Graph2SeqLoss(
            ignore_index=self.vocab.out_word_vocab.PAD,
            use_coverage=self.use_coverage,
            coverage_weight=0.3,
        )

    def train(self, split):
        max_score = -1
        self._best_epoch = -1
        for epoch in range(10):
            self.model.train()
            self.train_epoch(epoch, split)
            self._adjust_lr(epoch)
            if epoch >= 0:
                eval_split = "train_sql" if split=="train_sql" else "test"
                score = self.evaluate(split=eval_split)
                if score >= max_score:
                    self.logger.info("Best model saved, epoch {}".format(epoch))
                    self.model.save_checkpoint(self.opt["checkpoint_save_path"], 
                                               "best_w2v_wikisql.pt" if self.opt["dataset"]=="WikiSQL"\
                                                   else "best_w2v_wtq.pt")
                    self._best_epoch = epoch
                max_score = max(max_score, score)
            if epoch >= 30 and self._stop_condition(epoch):
                break
        return max_score

    def _stop_condition(self, epoch, patience=20):
        return epoch > patience + self._best_epoch

    def _adjust_lr(self, epoch):
        def set_lr(optimizer, decay_factor):
            for group in optimizer.param_groups:
                group["lr"] = group["lr"] * decay_factor

        epoch_diff = epoch - self.opt["lr_start_decay_epoch"]
        if epoch_diff >= 0 and epoch_diff % self.opt["lr_decay_per_epoch"] == 0:
            if self.opt["learning_rate"] > self.opt["min_lr"]:
                set_lr(self.optimizer, self.opt["lr_decay_rate"])
                self.opt["learning_rate"] = self.opt["learning_rate"] * self.opt["lr_decay_rate"]
                self.logger.info("Learning rate adjusted: {:.5f}".format(self.opt["learning_rate"]))

    def train_epoch(self, epoch, split):
        assert split in ["train_table", "train_sql"]
        self.logger.info("Start training in split {}, Epoch: {}".format(split, epoch))
        loss_collect = []
        dataloader = self.train_table_dataloader if split=="train_table" else self.train_sql_dataloader
        step_all_train = len(dataloader)
        for step, data in enumerate(dataloader):
            try:
                graph, tgt, gt_str = data["graph_data"], data["tgt_seq"], data["output_str"]
                graph = graph.to(self.device)
                tgt = tgt.to(self.device)

                oov_dict = None
                if self.use_copy:
                    oov_dict, tgt = prepare_ext_vocab(
                        graph, self.vocab, gt_str=gt_str, device=self.device
                    )

                prob, enc_attn_weights, coverage_vectors = self.model(graph, tgt, oov_dict=oov_dict)
                loss = self.loss(
                    logits=prob,
                    label=tgt,
                    enc_attn_weights=enc_attn_weights,
                    coverage_vectors=coverage_vectors,
                )

                loss_collect.append(loss.item())
                if step % self.opt["loss_display_step"] == 0 and step != 0:
                    self.logger.info(
                        "Epoch {}: [{} / {}] loss: {:.3f}".format(
                            epoch, step, step_all_train, np.mean(loss_collect)
                        )
                    )
                    loss_collect = []

                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
            except Exception as e:
                print(f"Error on set {split}, epoch {epoch}, step {step}: {e}")


    def evaluate(self, split="test"):
        split_dict = {# "train_sql": self.train_sql_dataloader,
                      "test": self.test_dataloader,
                      "dev1": self.dev1_dataloader,
                      "dev2": self.dev2_dataloader,
                      "dev3": self.dev3_dataloader,
                      "perturb_test_orig":    self.perturb_test_orig_dataloader,
                      "perturb_test_changed": self.perturb_test_changed_dataloader,
                      "perturb_dev_orig":     self.perturb_dev_orig_dataloader,
                      "perturb_dev_changed":  self.perturb_dev_changed_dataloader}
        self.model.eval()
        pred_collect = []
        gt_collect = []
        assert split in ["train_sql","test","dev1","dev2","dev3",
                         "perturb_test_orig","perturb_test_changed",
                         "perturb_dev_orig","perturb_dev_changed"]
        dataloader = split_dict[split]
        for idx, data in enumerate(dataloader):
            try:
                graph, gt_str = data["graph_data"], data["output_str"]
                graph = graph.to(self.device)
                if self.use_copy:
                    oov_dict = prepare_ext_vocab(
                        batch_graph=graph, vocab=self.vocab, device=self.device
                    )
                    ref_dict = oov_dict
                else:
                    oov_dict = None
                    ref_dict = self.vocab.out_word_vocab

                prob, _, _ = self.model(graph, oov_dict=oov_dict)
                pred = prob.argmax(dim=-1)

                pred_str = wordid2str(pred.detach().cpu(), ref_dict)
                pred_collect.extend(pred_str)
                gt_collect.extend(gt_str)
            except Exception as e:
                print(f"Skipping evaluation batch {idx}, Reason: {e}")

        if split in ["test","dev1","dev2","dev3",
                     "perturb_test_orig","perturb_test_changed",
                     "perturb_dev_orig","perturb_dev_changed"]:
            save_dir = "wikisql" if self.opt["dataset"]=='WikiSQL' else 'wtq'
            with open(f'output/w2v_{save_dir}_{split}_gt.txt', 'w') as fp:
                for item in gt_collect:
                    # write each item on a new line
                    fp.write("%s\n" % item)
                print('Done')
            with open(f'output/w2v_{save_dir}_{split}_pred.txt', 'w') as fp:
                for item in pred_collect:
                    # write each item on a new line
                    fp.write("%s\n" % item)
                print('Done')
 
        score = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        self.logger.info("Evaluation accuracy in `{}` split: {:.3f}".format(split, score))
        return score

    @torch.no_grad()
    def translate(self):
        self.model.eval()

        pred_collect = []
        gt_collect = []
        dataloader = self.test_dataloader
        for idx, data in enumerate(dataloader):
            try:
                graph, gt_str = data["graph_data"], data["output_str"]
                graph = graph.to(self.device)
                if self.use_copy:
                    oov_dict = prepare_ext_vocab(
                        batch_graph=graph, vocab=self.vocab, device=self.device
                    )
                    ref_dict = oov_dict
                else:
                    oov_dict = None
                    ref_dict = self.vocab.out_word_vocab

                pred = self.model.translate(batch_graph=graph, oov_dict=oov_dict, beam_size=4, topk=1)

                pred_ids = pred[:, 0, :]  # we just use the top-1

                pred_str = wordid2str(pred_ids.detach().cpu(), ref_dict)

                pred_collect.extend(pred_str)
                gt_collect.extend(gt_str)
            
            except Exception as e:
                print(f"Skipping evaluation batch {idx}, Reason: {e}")

        score = self.metrics[0].calculate_scores(ground_truth=gt_collect, predict=pred_collect)
        self.logger.info("Evaluation accuracy in `{}` split: {:.3f}".format("test", score))
        return score


if __name__ == "__main__":
    opt = get_args()
    print(opt['mode'])
    if opt["mode"] == 'train':
        runner = TableQA(opt)
        max_score = runner.train("train_sql")
        runner.logger.info("Train SQL finish, best val score: {:.3f}".format(max_score))
        max_score = runner.train("train_table")
        runner.logger.info("Train Table finish, best val score: {:.3f}".format(max_score))
        runner.evaluate("test")
        runner.evaluate("dev1")
        runner.evaluate("dev2")
        runner.evaluate("dev3")
    elif opt["mode"] == 'evaluate':
        runner = TableQA(opt)
        runner.evaluate("test")
        runner.evaluate("dev1")
        runner.evaluate("dev2")
        runner.evaluate("dev3")
    elif opt["mode"] == "perturb":
        runner = TableQA(opt)
        runner.evaluate("perturb_test_orig")
        runner.evaluate("perturb_test_changed")
        runner.evaluate("perturb_dev_orig")
        runner.evaluate("perturb_dev_changed")
    else:
        print("mode should be in ['train','evaluate','perturb']")
        assert False


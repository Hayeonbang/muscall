import os
import numpy as np
from sklearn import metrics
import json 
import torch
from torch.utils.data import DataLoader
from torchaudio.datasets.gtzan import gtzan_genres
from transformers.models.clip.tokenization_clip import CLIPTokenizer
from transformers import BertTokenizer

from muscall.tasks.retrieval import Retrieval
from muscall.datasets.tagging import MTTDataset, TestDataset
from muscall.datasets.gtzan import GTZAN
from sklearn.metrics import roc_auc_score, average_precision_score

TAGNAMES = ['Beautiful', 'Emotional', 'Romantic', 'Background-music', 'Relaxing/Calm',
        'Soft', 'Bright', 'Happy', 'Upbeat/Energetic', 'Cute', 'Playful', 'Dreamy',
        'Mysterious', 'Sad', 'Dark', 'Tense', 'Scary', 'Epic', 'Intense/Grand',
        'Passionate', 'Powerful', 'Difficult', 'Easy', 'Speedy', 'Laid-back', 'Jazz',
        'New-age', 'Pop-Piano Cover', 'Classical', 'Swing', 'Funk', 'Latin', 'Blues', 'Ragtime', 'Ballad', 'Pop-rock']

tags = [tag.lower() for tag in TAGNAMES]
def prepare_labels(labels, prompt=None):
    #tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    max_length = 77
    text_prompts = torch.zeros((len(labels), max_length), dtype=torch.long).cuda()
    attention_masks = torch.zeros((len(labels), max_length), dtype=torch.long).cuda()

    for i, label in enumerate(labels):
        if prompt is None:
            text_to_tokenize = label
        else:
            text_to_tokenize = "A {} track".format(label)

        encoded = tokenizer(
            text_to_tokenize, max_length=max_length, truncation=True, return_tensors="pt"
        )
        input_ids = encoded['input_ids'][0]
        input_ids = torch.cat([input_ids, torch.zeros(max_length - input_ids.size(0), dtype=torch.long)], dim=0)

        attention_mask = encoded['attention_mask'][0]
        attention_mask = torch.cat([attention_mask, torch.zeros(max_length - attention_mask.size(0), dtype=torch.long)], dim=0)

        text_prompts[i] = input_ids
        attention_masks[i] = attention_mask

    return text_prompts, attention_masks



def get_metrics(predictions, ground_truth, tags):
    # 그라디언트 추적을 중단하고 NumPy 배열로 변환
    predictions = predictions.detach().cpu().numpy()
    ground_truth = ground_truth.detach().cpu().numpy()

    results = {}
    tag_results = {}
    
    # 레이블별로 ROC-AUC 계산
    roc_auc = roc_auc_score(ground_truth, predictions, average=None)
    pr_auc = average_precision_score(ground_truth, predictions, average=None)
    for i, tag in enumerate(tags):
        tag_results[tag] = {
            'ROC-AUC': roc_auc[i],
            'PR-AUC': pr_auc[i]
        }
    results['tagwise'] = tag_results
    results["ROC-AUC-macro"] = np.mean(roc_auc)
    results["PR-AUC-macro"] = np.mean(pr_auc)
    
    return results


@torch.no_grad()

def encode_labels(label_str, all_tags):
    labels = label_str.lower().split(", ")
    label_vector = [1 if tag in labels else 0 for tag in all_tags]
    return torch.tensor(label_vector, dtype=torch.float)


def compute_muscall_similarity_score(model, data_loader, text_prompts, attention_masks, device):
    dataset_size = data_loader.dataset.__len__()

    all_audio_features = torch.zeros(dataset_size, 512).to("cuda")
    ground_truth = torch.zeros(dataset_size, len(tags)).to("cuda")  # 태그 수에 맞게 ground_truth 텐서 초기화

#TODO: text_mask 추가
    all_text_features = model.encode_text(text_prompts, attention_masks)
    all_text_features = all_text_features / all_text_features.norm(dim=-1, keepdim=True)

    for i, batch in enumerate(data_loader):
        batch = tuple(t.to(device=device, non_blocking=True) if torch.is_tensor(t) else t for t in batch)
        input_audio, labels_tuple = batch
        labels_str = labels_tuple[0]  # 튜플에서 문자열 추출
        labels = encode_labels(labels_str, tags)  # 이진 벡터로 인코딩

        input_audio = input_audio.to(device=device)

        audio_features = model.encode_audio(input_audio)
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)

        num_samples_in_batch = input_audio.size(0)

        all_audio_features[i * num_samples_in_batch : (i + 1) * num_samples_in_batch] = audio_features
        ground_truth[i * num_samples_in_batch : (i + 1) * num_samples_in_batch] = labels

    logits_per_audio = all_audio_features @ all_text_features.t()

    return logits_per_audio, ground_truth

def random_predictions(ground_truth, seed=None):
    np.random.seed(seed)
    random_preds = np.random.rand(*ground_truth.shape)
    return torch.tensor(random_preds, dtype=torch.float)

def evaluate_random_predictions(ground_truth, tags):
    random_preds = random_predictions(ground_truth)
    metrics_random = get_metrics(random_preds, ground_truth, tags)
    return metrics_random

def get_top5_tags(score_matrix, tags):
    top5_indices = torch.topk(score_matrix, 5, dim=1).indices  # 상위 5개 인덱스 추출
    top5_tags = [[tags[idx] for idx in indices] for indices in top5_indices.cpu().numpy()]
    return top5_tags

class Zeroshot(Retrieval):
    def __init__(self, pretrain_config, json_path, npy_dir):
        self.json_path = json_path
        self.npy_dir = npy_dir
        super().__init__(pretrain_config)

    def load_dataset(self):
        self.test_dataset = TestDataset(self.json_path, self.npy_dir)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=1)

    def evaluate(self):
        text_prompts, attention_masks = prepare_labels(tags)

        score_matrix, ground_truth = compute_muscall_similarity_score(
            self.model, self.test_loader, text_prompts, attention_masks, self.device
        )
        
        top5_tags = get_top5_tags(score_matrix, tags)
        for i, (top_tags, actual_labels) in enumerate(zip(top5_tags, ground_truth)):
            actual_tags = [tags[j] for j in range(len(tags)) if actual_labels[j] == 1]
            print(f"Sample {i}: \n Actual Tags: {actual_tags}, \n Top-5 Predicted Tags: {top_tags}")
    
        
        # 실제 모델의 성능 평가
        model_metrics = get_metrics(score_matrix.cpu(), ground_truth.cpu(), tags)

        # 무작위 예측의 성능 평가
        random_metrics = evaluate_random_predictions(ground_truth.cpu(), tags)
        results = {
            "Model Metrics": model_metrics,
            "Random Prediction Metrics": random_metrics
        }

        with open('./tagging_results-.json', 'w') as file:
            json.dump(results, file, indent=4)

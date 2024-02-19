import os
import json
import torch
from torch.utils.data import DataLoader, Subset

from muscall.models.muscall import MusCALL
from muscall.datasets.audiocaption import AudioCaptionDataset
from sklearn.metrics import roc_auc_score, average_precision_score

TAGNAMES = ['Beautiful', 'Emotional', 'Romantic', 'Background-music', 'Relaxing/Calm',
        'Soft', 'Bright', 'Happy', 'Upbeat/Energetic', 'Cute', 'Playful', 'Dreamy',
        'Mysterious', 'Sad', 'Dark', 'Tense', 'Scary', 'Epic', 'Intense/Grand',
        'Passionate', 'Powerful', 'Difficult', 'Easy', 'Speedy', 'Laid-back', 'Jazz',
        'New-age', 'Pop-Piano Cover', 'Classical', 'Swing', 'Funk', 'Latin', 'Blues', 'Ragtime', 'Ballad', 'Pop-rock']

tag_list = [tag.lower() for tag in TAGNAMES]

@torch.no_grad()
def get_muscall_features(model, data_loader, device):
    dataset_size = data_loader.dataset.__len__()

    all_audio_features = torch.zeros(dataset_size, 512).to(device)
    all_text_features = torch.zeros(dataset_size, 512).to(device)

    samples_in_previous_batch = 0
    for i, batch in enumerate(data_loader):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        _, input_audio, text_input_ids, _, text_attention_mask, _ = batch

        audio_features = model.encode_audio(input_audio)
        text_features = model.encode_text(text_input_ids, text_attention_mask)

        audio_features = audio_features / \
            audio_features.norm(dim=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        samples_in_current_batch = input_audio.size(0)
        start_index = i * samples_in_previous_batch
        end_index = start_index + samples_in_current_batch
        samples_in_previous_batch = samples_in_current_batch

        all_audio_features[start_index:end_index] = audio_features
        all_text_features[start_index:end_index] = text_features

    return all_audio_features, all_text_features


def compute_sim_score(audio_features, text_features):
    logits_per_audio = audio_features @ text_features.t()
    logits_per_text = logits_per_audio.t()

    return logits_per_text


def get_ranking(score_matrix, device):
    num_queries = score_matrix.size(0)
    num_items = score_matrix.size(1)

    scores_sorted, retrieved_indices = torch.sort(
        score_matrix, dim=1, descending=True)
    gt_indices = torch.zeros((num_queries, num_items, 1))

    for i in range(num_queries):
        gt_indices[i] = torch.full((num_queries, 1), i)

    gt_indices = gt_indices.squeeze(-1).to(device)

    return retrieved_indices, gt_indices


def compute_metrics(retrieved_indices, gt_indices):
    num_items = gt_indices.size(1)

    bool_matrix = retrieved_indices == gt_indices

    r1 = 100 * bool_matrix[:, 0].sum() / num_items
    r5 = 100 * bool_matrix[:, :5].sum() / num_items
    r10 = 100 * bool_matrix[:, :10].sum() / num_items

    median_rank = (torch.where(bool_matrix == True)[1] + 1).median()

    retrieval_metrics = {
        "R@1": r1,
        "R@5": r5,
        "R@10": r10,
        "Median Rank": median_rank,
    }

    return retrieval_metrics

def run_retrieval(model, data_loader, device):
    """Wrapper function to run all steps for text-audio/audio-text retrieval"""
    audio_features, text_features = get_muscall_features(
        model, data_loader, device)
    score_matrix = compute_sim_score(audio_features, text_features)
    retrieved_indices, gt_indices = get_ranking(score_matrix, device)
    retrieval_metrics = compute_metrics(retrieved_indices, gt_indices)

    return retrieval_metrics


class Retrieval:
    def __init__(self, muscall_config, test_set_size=0):
        super().__init__()
        self.muscall_config = muscall_config
        self.device = torch.device(self.muscall_config.training.device)
        self.path_to_model = os.path.join(
            self.muscall_config.env.experiments_dir,
            self.muscall_config.env.experiment_id,
            "best_model.pth.tar",
        )
        print("path to model", self.path_to_model)

        self.test_set_size = test_set_size

        self.load_dataset()
        self.build_model()

    def load_dataset(self):
        dataset = AudioCaptionDataset(self.muscall_config.dataset_config, dataset_type="test")
        indices = torch.randperm(len(dataset))[: self.test_set_size]
        random_dataset = Subset(dataset, indices)
        self.batch_size = 1
        self.data_loader = DataLoader(
            dataset=random_dataset,
            batch_size=self.batch_size,
            drop_last=False,
        )

    def build_model(self):
        self.model = MusCALL(self.muscall_config.model_config)
        self.checkpoint = torch.load(self.path_to_model)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()


    def print_top_results(self, retrieved_indices):
        # JSON 파일에서 테스트 데이터셋 로드
        with open('./data/datasets/audiocaption/dataset_test.json', 'r') as file:
            test_dataset = json.load(file)

        for query_idx in range(min(len(test_dataset), self.test_set_size)):
            # 기준 텍스트 쿼리 데이터 가져오기
            query_caption = test_dataset[query_idx]['caption']
            #print(f"\n기준 텍스트 쿼리: {query_caption}")

            # 상위 5개 검색 결과
            top_5_indices = retrieved_indices[query_idx][:5].tolist()
            #print("Top 5 샘플의 텍스트 쿼리:")
            for rank, idx in enumerate(top_5_indices, start=1):
                top_caption = test_dataset[idx]['caption']
                #print(f"  {rank}: {top_caption}")
                
    def evaluate(self):
        audio_features, text_features = get_muscall_features(
            self.model, self.data_loader, self.device
        )
        score_matrix = compute_sim_score(text_features, audio_features)

        retrieved_indices, gt_indices = get_ranking(score_matrix, self.device)
        retrieval_metrics = compute_metrics(retrieved_indices, gt_indices)
        print('Retrieval_metrics: ', retrieval_metrics)

        self.print_top_results(retrieved_indices)
        return retrieval_metrics                

 
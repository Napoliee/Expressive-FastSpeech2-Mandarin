import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

from text.ipa_processor import text_to_sequence_ipa
from utils.tools import pad_1D, pad_2D
from tqdm import tqdm


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, model_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.max_seq_len = model_config["max_seq_len"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text, self.aux_data = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
            json_raw = json.load(f)
            self.emotion_map = json_raw["emotion_dict"]
            self.arousal_map = json_raw["arousal_dict"]
            self.valence_map = json_raw["valence_dict"]
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        aux_data = self.aux_data[idx].split("|")
        emotion = self.emotion_map[aux_data[-1]]
        arousal = self.arousal_map[aux_data[-1]]
        valence = self.valence_map[aux_data[-1]]
        raw_text = self.raw_text[idx]
        
        # 使用IPA文本处理器
        phone = np.array(text_to_sequence_ipa(self.text[idx]), dtype=np.int64)
        
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path).astype(np.float32)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path).astype(np.float32)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path).astype(np.float32)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path).astype(np.float32)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "emotion": emotion,
            "arousal": arousal,
            "valence": valence,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            aux_data = []
            for line in tqdm(f.readlines()):
                line_split = line.strip("\n").split("|")
                n, s, t, r = line_split[:4]
                mel_path = os.path.join(
                    self.preprocessed_path,
                    "mel",
                    "{}-mel-{}.npy".format(s, n),
                )
                mel = np.load(mel_path).astype(np.float32)
                if mel.shape[0] > self.max_seq_len:
                    continue
                a = "|".join(line_split[4:])
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                aux_data.append(a)
            return name, speaker, text, raw_text, aux_data

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        emotions = [data[idx]["emotion"] for idx in idxs]
        arousals = [data[idx]["arousal"] for idx in idxs]
        valences = [data[idx]["valence"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts], dtype=np.int64)
        mel_lens = np.array([mel.shape[0] for mel in mels], dtype=np.int64)

        speakers = np.array(speakers, dtype=np.int64)
        emotions = np.array(emotions, dtype=np.int64)
        arousals = np.array(arousals, dtype=np.float32)
        valences = np.array(valences, dtype=np.float32)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids,
            raw_texts,
            speakers,
            emotions,
            arousals,
            valences,
            texts,
            text_lens,
            max(text_lens),
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config, model_config):
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.max_seq_len = model_config["max_seq_len"]

        self.basename, self.speaker, self.text, self.raw_text, self.aux_data = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)
        with open(os.path.join(self.preprocessed_path, "emotions.json")) as f:
            json_raw = json.load(f)
            self.emotion_map = json_raw["emotion_dict"]
            self.arousal_map = json_raw["arousal_dict"]
            self.valence_map = json_raw["valence_dict"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        aux_data = self.aux_data[idx].split("|")
        emotion = self.emotion_map[aux_data[-1]]
        arousal = self.arousal_map[aux_data[-1]]
        valence = self.valence_map[aux_data[-1]]
        raw_text = self.raw_text[idx]
        
        # 使用IPA文本处理器
        phone = np.array(text_to_sequence_ipa(self.text[idx]), dtype=np.int64)

        return (basename, speaker_id, emotion, arousal, valence, phone, raw_text)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            aux_data = []
            for line in tqdm(f.readlines()):
                line_split = line.strip("\n").split("|")
                n, s, t, r = line_split[:4]
                mel_path = os.path.join(
                    self.preprocessed_path,
                    "mel",
                    "{}-mel-{}.npy".format(s, n),
                )
                mel = np.load(mel_path).astype(np.float32)
                if mel.shape[0] > self.max_seq_len:
                    continue
                a = "|".join(line_split[4:])
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                aux_data.append(a)
            return name, speaker, text, raw_text, aux_data

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data], dtype=np.int64)
        emotions = np.array([d[2] for d in data], dtype=np.int64)
        arousals = np.array([d[3] for d in data], dtype=np.float32)
        valences = np.array([d[4] for d in data], dtype=np.float32)
        texts = [d[5] for d in data]
        raw_texts = [d[6] for d in data]
        text_lens = np.array([text.shape[0] for text in texts], dtype=np.int64)

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, emotions, arousals, valences, texts, text_lens, max(text_lens) 
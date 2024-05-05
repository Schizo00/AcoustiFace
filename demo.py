import tensorflow as tf
import pickle
from weighted_mse import WeightedLoss
from speechbrain.pretrained import EncoderClassifier
import os
from tqdm import tqdm
import open3d as o3d
from dataset_tf import VoxDataset as dataset
import librosa
import torch
import torchaudio
import numpy as np


class Pipeline():

    def __init__(self, model_path, loss_config_path):

        with open(loss_config_path, 'rb') as file:
            self.weighted_loss_config = pickle.load(file)

        self.weighted_loss = WeightedLoss(tf.keras.losses.MeanSquaredError(), self.weighted_loss_config['attention_ids'], self.weighted_loss_config['weight'])
        print(self.weighted_loss_config)
        self.model_path = model_path
        self.model = tf.keras.models.load_model(self.model_path, custom_objects={"WeightedLoss" : self.weighted_loss})
        self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cpu"})
        self.faces = dataset().get_faces()
        self.dataset = dataset()


    def scale_signals(self,target_len, signal):
        if signal.shape[1] < target_len:
            missing = target_len - signal.shape[1]
            last_dim = (0, missing)
            signal = torch.nn.functional.pad(signal, last_dim)
            print("inside scale")
            return signal

        elif signal.shape[1] > target_len:
            signal = signal[:, :target_len]
            print("inside scale")
            return signal
        
        else:
            print("inside scale")
            return signal
        

    def get_embedding(self, audio):
        print(type(audio))
        signal, fs = torchaudio.load(audio)
        if signal.shape[0] == 1:
            print("converting")
            signal = signal.repeat(2, 1)
        # print("scale")
        print("before scaling: ",signal.shape)
        signal = self.scale_signals(683433, signal)
        print("after scaling: ", signal.shape)
        # print("encode")
        embedding = self.classifier.encode_batch(signal)
        print("shape:, ", embedding.shape)
        # print("to numpy")
        # embedding = signal
        embedding = embedding.cpu().numpy()

        return embedding
    

    def __call__(self, audio):
        embedding = self.get_embedding(audio).flatten()
        
        # expanded = tf.expand_dims(embedding, axis=0)
        # print(expanded.shape)


        inputs = [embedding]
        print(inputs)

        expanded = np.array(inputs)  / 113.46356

        print("new shape: ", expanded.shape)



        points = self.model.predict(expanded)
        points_reshaped = self.dataset.to_mesh_points(points)
        vertices = o3d.utility.Vector3dVector(points_reshaped)
        triangles = o3d.utility.Vector3iVector(self.faces)
        mesh = o3d.geometry.TriangleMesh(vertices, triangles)
        path = "temp.glb"
        o3d.io.write_triangle_mesh(f"temp.glb", mesh, 
                                    write_ascii=False
                                )
        # return embedding
        print(path)
        return path
        # points = self.model.predict(audio)


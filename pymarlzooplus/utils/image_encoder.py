import inspect

import numpy as np
import torch
from torch import nn


class ImageEncoder(nn.Module):
    """
    Observation wrapper for converting images to vectors (using a pretrained image encoder) or
    for preparing images to be fed to a CNN.
    """

    def __init__(
            self,
            called_from,  # Valid options: "env", "parallel_runner"
            centralized_image_encoding,
            trainable_cnn,
            image_encoder,
            image_encoder_batch_size,
            image_encoder_use_cuda
    ):

        super(ImageEncoder, self).__init__()

        self.called_from = called_from
        self.centralized_image_encoding = centralized_image_encoding
        self.trainable_cnn = trainable_cnn
        self.print_info = None

        assert not (self.centralized_image_encoding is True and self.trainable_cnn is True), \
            "'centralized_image_encoding' and 'trainable_cnn' cannot be both True!"

        ## Define image encoder. In this case, a pretrained model is used frozen, i.e., without further training.
        self.image_encoder = None
        self.observation_space = None
        if (
                self.centralized_image_encoding is False or
                (self.centralized_image_encoding is True and self.called_from == "parallel_runner")
        ):

            if self.trainable_cnn is False:

                # Define the device to be used
                self.device = "cpu"
                if image_encoder_use_cuda is True and torch.cuda.is_available() is True:
                    self.device = "cuda"

                # Define the batch size of the image encoder
                self.image_encoder_batch_size = image_encoder_batch_size

                # Encoder
                self.image_encoder_predict = None
                if image_encoder == "ResNet18":

                    # Imports
                    import albumentations as A
                    from albumentations.pytorch import ToTensorV2
                    from torch import nn
                    from torchvision.models import resnet18

                    # Define ResNet18
                    self.print_info = "Loading pretrained ResNet18 model..."
                    self.image_encoder = resnet18(weights='IMAGENET1K_V1')
                    self.image_encoder.fc = nn.Identity()
                    self.image_encoder = self.image_encoder.to(self.device)
                    self.image_encoder.eval()

                    # Image transformations
                    img_size = 224
                    self.transform = A.Compose([
                        A.LongestMaxSize(max_size=img_size, interpolation=1),  # Resize the longest side to 224
                        self.make_pad_if_needed(img_size=img_size, border_mode=0, value=(0, 0, 0)),  # Pad to make the image square
                        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2()
                    ])

                    # Get the number of features by feeding the model with a dummy input
                    dummy_input = np.ones((1, img_size, img_size, 3), dtype=np.uint8)*255
                    dummy_output = self.resnet18_predict(dummy_input)
                    n_features = dummy_output.shape[1]

                    # Define the function to get predictions
                    self.image_encoder_predict = self.resnet18_predict

                elif image_encoder == "SlimSAM":

                    # Imports
                    from transformers import SamModel, SamProcessor

                    # Define SAM
                    self.print_info = "Loading pretrained SlimSAM model..."
                    self.image_encoder = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50").to(self.device)
                    self.image_encoder.eval()

                    # Image transformations
                    self.processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")

                    # Get the number of features by feeding the model with a dummy input
                    img_size = 224
                    dummy_input = np.ones((1, img_size, img_size, 3), dtype=np.uint8) * 255
                    dummy_output = self.sam_predict(dummy_input)
                    n_features = dummy_output.shape[1]

                    # Define the function to get predictions
                    self.image_encoder_predict = self.sam_predict

                elif image_encoder == "CLIP":

                    # Imports
                    from transformers import AutoProcessor, CLIPVisionModel

                    # Define CLIP-image-encoder
                    self.print_info = "Loading pretrained CLIP-image-encoder model..."
                    self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                    self.image_encoder.eval()

                    # Image transformations
                    self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

                    # Get the number of features by feeding the model with a dummy input
                    img_size = 224
                    dummy_input = np.ones((1, img_size, img_size, 3), dtype=np.uint8) * 255
                    dummy_output = self.clip_predict(dummy_input)
                    n_features = dummy_output.shape[1]

                    # Define the function to get predictions
                    self.image_encoder_predict = self.clip_predict

                else:
                    raise NotImplementedError(f"Invalid image encoder: {image_encoder}")

                # Define the observation space
                self.observation_space = (n_features,)

            elif self.trainable_cnn is True:
                # In this case, we use NatureCNN, adopted from openAI:
                # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/torch_layers.py

                # Image transformations.
                # Images will be downscaled to 224 max-size, reducing the complexity,
                # and will be padded (if needed) to become square.
                # Images will be normalized simply by dividing by 255 (as in the original Nature paper,
                # but without converting to gray-scale).
                import albumentations as A
                from albumentations.pytorch import ToTensorV2
                img_size = 224
                self.transform = A.Compose([
                    A.LongestMaxSize(max_size=img_size, interpolation=1),  # Resize the longest side to 224
                    self.make_pad_if_needed(img_size=img_size, border_mode=0, value=(0, 0, 0)),  # Pad to make the image square
                    ToTensorV2()
                ])

                # Define the observation space
                self.observation_space = (3, img_size, img_size)
            else:
                raise NotImplementedError()

    @staticmethod
    def make_pad_if_needed(img_size, border_mode, value):

        import albumentations as A

        params = inspect.signature(A.PadIfNeeded).parameters
        kwargs = dict(
            min_height=img_size,
            min_width=img_size,
            border_mode=border_mode,
        )
        if 'fill' in params:  # Albumentations >= 2.0
            kwargs['fill'] = value
        else:  # Albumentations < 2.0
            kwargs['value'] = value

        return A.PadIfNeeded(**kwargs)

    def resnet18_predict(self, observation):
        """
        observation: np.array of shape [batch_size, height, width, 3]
        """

        assert isinstance(observation, np.ndarray), \
            f"'observation' is not a numpy array! 'type(observation)': {type(observation)} "
        assert observation.ndim == 4 and observation.shape[3] == 3, \
            f"'observation' has not the right dimensions! 'observation.shape': {observation.shape}"

        observation = [self.transform(image=obs)["image"][None] for obs in observation]
        observation = torch.concatenate(observation, dim=0)
        observation = observation.to(self.device)
        with torch.no_grad():
            observation = self.image_encoder(observation)
            observation = observation.detach().cpu().numpy()

        return observation

    def sam_predict(self, observation):
        """
        observation: np.array of shape [batch_size, height, width, 3]
        """

        assert isinstance(observation, np.ndarray), \
            f"'observation' is not a numpy array! 'type(observation)': {type(observation)} "
        assert observation.ndim == 4 and observation.shape[3] == 3, \
            f"'observation' has not the right dimensions! 'observation.shape': {observation.shape}"

        observation = self.processor(observation, return_tensors="pt")['pixel_values'].to(self.device)
        with torch.no_grad():
            observation = self.image_encoder.get_image_embeddings(pixel_values=observation)
            bs = observation.shape[0]
            observation = observation.view((bs, -1)).detach().cpu().numpy()

        return observation

    def clip_predict(self, observation):
        """
        observation: np.array of shape [batch_size, height, width, 3]
        """

        assert isinstance(observation, np.ndarray), \
            f"'observation' is not a numpy array! 'type(observation)': {type(observation)} "
        assert observation.ndim == 4 and observation.shape[3] == 3, \
            f"'observation' has not the right dimensions! 'observation.shape': {observation.shape}"

        observation = self.processor(images=observation, return_tensors="pt").to(self.device)
        with torch.no_grad():
            observation = self.image_encoder(**observation).pooler_output
            observation = observation.detach().cpu().numpy()

        return observation

    def observation(self, observations):

        if isinstance(observations, tuple):
            # When 'observations' is tuple it means that it has been called from gym reset()
            # and it carries pettingzoo observations and info
            observations = observations[0]

        observations_ = []
        if (
                self.trainable_cnn is False and
                (
                        self.centralized_image_encoding is False or
                        (
                                self.centralized_image_encoding is True and self.called_from == "parallel_runner"
                        )
                )
        ):
            # Get image representations
            observations_tmp = []
            observations_tmp_counter = 0
            observations_tmp_counter_total = 0
            for observation_ in observations.values():
                observations_tmp.append(observation_[None])
                observations_tmp_counter += 1
                observations_tmp_counter_total += 1
                if observations_tmp_counter == self.image_encoder_batch_size or \
                   observations_tmp_counter_total == len(observations.values()):
                    # Predict in batches. When GPU is used, this is faster than inference over single images.
                    observations_tmp = np.concatenate(observations_tmp, axis=0)
                    observations_tmp = self.image_encoder_predict(observations_tmp)
                    observations_.extend([obs for obs in observations_tmp])
                    # Reset tmp
                    observations_tmp = []
                    observations_tmp_counter = 0

        elif self.trainable_cnn is True and self.centralized_image_encoding is False:
            # Preprocess images for a CNN network.
            observations_ = [
                self.transform(image=observation_)["image"].detach().cpu().numpy()
                for observation_ in observations.values()
            ]

        elif self.trainable_cnn is False and self.centralized_image_encoding is True and self.called_from == "env":
            observations_ = [observations]

        else:
            raise NotImplementedError()

        return tuple(observations_,)

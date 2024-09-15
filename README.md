# Nano GPT
This code is about experiments with Nano-GPT developed by Andrej Karpathy.

## Execution
To successfuly train the model and use it for inference here are the steps:
1. Load dataset (OpenWebText from HuggingFace) using `data/prepare_dataset.py` script (takes approximately 30 minutes depending on the internet connection and processing power).

   **Note**: HuggingFace loads the dataset to the cache folder, but it does not clear it automatically, so need to clear it manually, which is done in the script.
   
2. Run `training.py` script. Configuration of both: the model and training, is written in `configs/config.py`. There you can define size of the model, sequence length, hyperparameters and so on. The script saves checkpoints (better not do it always, only in "production" mode) and also plots and weights after training.

   **For reference:** training with my config and using MPS device (Apple Silicon M1) would take approsimately 3-4 hours. It ***will not*** use all data from the dataset. The default configuration has embeddings dimension of 128, sequence length (block size) of 64, 4 heads and 4 layers.

3. Run `generate.py`, which shall load the model weights and generate the output. You can change the input in the script.

   **Note:** you need to completely train the model in order to generate the output. Or you can use weights that I have got during training (by default they are stored in `trained_model/gpt2_trained_weights.pth` and also this path is used by default.

## Results:

1) Here is the loss plot over steps after short pre-training:

![image](https://github.com/user-attachments/assets/9dc1221c-8db9-4f3c-bd0c-d3459e614f64)

There is still room for imrpovements. Hyperparameters can be tuned to obtain better training results.
Since I cannot iterate super fast due to processing power limitations, I would probably stop here, but let's see.

Here is the loss plot of one of the previous trainings with 8000 iterations:

![image](https://github.com/user-attachments/assets/a2ec4497-8a77-43a7-936e-cc2e31945457)

2) Output results (`generate.py`) after 40000 iterations with default config:

<img width="858" alt="image" src="https://github.com/user-attachments/assets/9b8c6739-afae-4ecc-9972-d5463a501b88">

As you can see, the output does not make any sense, which is expected. There are several reasons for that:
- the model is undertrained, the whole dataset was not used because the training would take a lot of time on my hardware
- the model is too small, because again the training would take a lot of time
- the training process is not optimized (hyperparameters tuning), because again training takes a lot of time (lack of opprotunity to iterate quickly)

## Future work (TODO):
- improve hyperparameters tuning
- add modes so user can pause, resume or reset training and so on
- add flexibilty using scripts, add args
- improve dataloader
- try out training on Google Cloud if possible
- ...



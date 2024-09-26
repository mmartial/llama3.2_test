# Testing Llama 3.2 11B Vision

Pre-requisite: Llama 3.2 is a gated model, you will need to apply for access before you can use this test code.
Go to https://huggingface.co/meta-llama/Llama-3.2-11B-Vision and follow the instructions to get access.

GPU note: when run, the model used about 22GB of VRAM, you will need an GPU with at least 24GB of memory to run this.

We will be using [CTPO](https://github.com/Infotrend-Inc/CTPO) as the base container for availability of most dependencies and easy re-deployable solution.
CTPO is a 20GB container. Obtain it by performing a 
```bash
docker pull infotrend/ctpo-cuda_tensorflow_pytorch_opencv:12.3.2_2.16.1_2.2.2_4.9.0-20240421
```

Note that we will still create caches separate from the container for easy re-deployment as needed:
1. a `HF_HOME` directory for download from HuggingFace to be separete from the container 
2. a `venv` within the container for quick re-run as needed.


Before continuing, you must get:
1. access to the model (see pre-requisite section above)
2. create a `.env` file containing a `HF_HOME=` variable with the content of your HF token (do not share this token)

## GPU test

To run the container; we are mounting the current directory (where this `README.md` is as `/iti`) where the `bash` will start --so that the `HF_HOME` and `venv` are external to the container-- and the container will run as the calling user's `uid` and `gid` (with the caveat that you will have a `I have no name` user)
```bash
docker run --rm -it --runtime nvidia --gpus all --user $(id -u):$(id -g) -v `pwd`:/iti infotrend/ctpo-cuda_tensorflow_pytorch_opencv:12.3.2_2.16.1_2.2.2_4.9.0-20240421
```

From within the bash terminal within the running CTPO container:
```bash
# Set a HF_HOME directory for the model to be downloaded to -- about 20GB for the 11B vision model
mkdir -p HF_HOME; export HF_HOME=`pwd`/HF_HOME

# create and activate a python virtualenv and install the required python packages -- about 500MB for the additional packages
python3 -m venv --system-site-packages venv; source venv/bin/activate; mkdir -p venv/cache; pip3 install -r requirements.txt --cache-dir venv/cache

# Run the test
python3 ./haiku_test.py
```

For re-runs:
```bash
export HF_HOME=`pwd`/HF_HOME
source venv/bin/activate
python3 ./haiku_test.py
```

## a note on CPU test

Although it would be possible to run the code on CPU (the models will work on either CPU or GPU matrix) what takes seconds on GPU will take tens of minutes on CPU (and over 34GB of memory for the `python3` executable -- after 20 I `docker kill`ed it)

To test on CPU (or if you have no GPU on the sytem), you can replace the `docker run` command line to use the TPO version:
```bash
docker run --rm -it --user $(id -u):$(id -g) -v `pwd`:/iti infotrend/ctpo-tensorflow_pytorch_opencv:2.16.1_2.2.2_4.9.0-20240421
```

Use the same run or re-run steps as described in the GPU section.

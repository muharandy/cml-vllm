# Deploying LLM Model Serving using vLLM and FastAPI on CML

In this tutorial we will be deploying LLM model serving on CML. Before we start, make sure you have the following:
* Access to CML (either on CDP Public Cloud or PVC DS)
* CML GPU Worker nodes
    * Minimum 16GB VRAM (24GB or bigger is better to run bigger models)
    * Minimum compute capability 7.0 (8.0 or higher is better to run model with bfloat16)
* CML 2023.05 NVIDIA GPI runtimes with Python 3.10

## Install setuptools

Do this first and then restart the kernel or restart your session just to make sure. We need to use a lower version of `setuptools` because CML default 2023.05 CUDA runtime has some issues during the installation for some of the python packages required for LLMs.


```python
!pip install --q setuptools==59.8.0
```

## Install vLLM

VLLM allows faster inference using PagedAttention. More details can be found in https://vllm.ai

We will be using the REST API server example given in the vllm package which uses FastAPI and adapted from Fastchat OpenAI REST API server implementation. The difference between the implementation of REST API server in VLLM example with the one in Fastchat is the VLLM example is simplified to use only one serving component. The Fastchat implementation requires a controller, at least one REST server, and multiple workers. 

Depending on the requirement, the Fastchat implementation can be used or adapted into a more scalable solution. However for the purpose of this tutorial we will be using VLLM example implementation as it will only require us to deploy a single Application in CML.


```python
!pip install --q vllm
```

## Download the model

Refer to the requirements stated above, several things to note here are:
* VRAM size
* Compute Capability

For this example we will be using `lmsys/vicuna-7b-v1.3` since it is supported by vLLM, it can fit in a GPU with 16GB VRAM, and it can run on GPU with 7.0 compute capability. The only catch of running this model (or at least in my environment) is I cannot run it with the tokenizer that comes along with the model. I need to use `hf-internal-testing/llama-tokenizer` for it to work. For more details, refer to:
* https://github.com/vllm-project/vllm/pull/284

Optionally, we can download a different model and adjust the rest of the steps accordingly. Since we are using vLLM, just make sure that we are using one of the supported models.


```python
# Initialize LFS and clone the model repo
!git lfs install
!git lfs clone https://huggingface.co/lmsys/vicuna-7b-v1.3

# Move to models directory
!mv vicuna-7b-v1.3/ models/
```

### [Optional] Download Tokenizer
Your milage may vary, but in my case I need use a different tokenizer


```python
## download hf-internal-testing/llama-tokenizer
!git lfs clone https://huggingface.co/hf-internal-testing/llama-tokenizer
!mv llama-tokenizer/ models/
```

## Prepare server script

Create a directory called `server` and then create a file called `api_serve.py` with the following content


```shell
!python -m vllm.entrypoints.openai.api_server \
    --port $CDSW_APP_PORT \
    --host 127.0.0.1 \
    --model ./models/vicuna-7b-v1.3 \
    --tokenizer ./models/llama-tokenizer
```

## Create Application

Once the script is ready, we will create an application deployment in CML
* Go to `Applications`
* Click `New Application`
* Fill in the `Name` and `Subdomain` as per your choosings
* Select the script that you just created `server/api_serve.py`
* Make sure to pick the NVIDIA runtime with Python 3.10
* Select the resource profile as per the model requirement (suggested minimum: 4 vcores, 16GB, and 1GPU)
* Once everything is filled up, hit `Create Application`

We will wait for the application to be deployed and running, then click the application to open up a new tab in our browser.

_Hint: At this moment CML will check for liveliness by constantly probing the REST server. This will create a swarm of HTTP 404 error due to non-existing path. We might need to disable logging once we make sure that the everything is running well_

When we opened the application in a new tab in our browser, we will get something like this:


```json
{
  "detail": "Not Found"
}
```

That's actually a good sign. If we change the url by adding `v1/models` we will get something like this (depending on the model that we loaded)


```json
{
  "object": "list",
  "data": [
    {
      "id": "./models/vicuna-7b-v1.3",
      "object": "model",
      "created": 1690096055,
      "owned_by": "vllm",
      "root": "./models/vicuna-7b-v1.3",
      "parent": null,
      "permission": [
        {
          "id": "modelperm-906c6225ade24cceb83e496c21a4e061",
          "object": "model_permission",
          "created": 1690096055,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```

We can also try accessing the REST API using `curl` following OpenAI REST API specs by changing the `v1/models` to `/v1/completions/`


```python
!curl <YOUR_API_SERVER_URL>/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "./models/mpt-7b-instruct",
        "prompt": "Tell me about Apache Iceberg",
        "max_tokens": 300,
        "temperature": 0
    }'
```

### Configure REST Server URL as CML Environment Variable

If things are working correctly, copy our faux OpenAI base URL which is API Server URL deployed in CML up until `v1` omitting the `completions` or the `models`

* Then go to `Project Settings` -> `Advanced`
* Create a new variable as `LLM_API_SERVER_BASE` 
* Paste your base URL and hit `Submit`

_Hint: We can also set this in the `Site Administration` level to make it globally accessible. We can also make the LLM path as environment variable. In our example we can make `./models/vicuna-7b-v1.3` as `LLM_LOADED`_

## Accessing The REST API from Sessions

Create a CPU session (no need GPU) then install the `openai` client package by running `pip install openai`. We are not using OpenAI service here. We are only using the client library and use our recently deployed REST API server instead

Head on to examples directory for to see how we can make use of our recently deployed REST API. This list will keep growing as I build more examples:
* [`Simple.ipynb`](examples/Simple.ipynb) For a simple example of completions API
* [`Simple-langchain.ipynb`](examples/Simple-langchain.ipynb) For as simple langchain integration example
* [`Simple-streaming.ipynb`](examples/Simple-streaming.ipynb) For a simple example but utilizing streaming token output instead of batch
* `Simple-RAG.ipynb` [WIP] For an example of RAG text generation from a supplied document context
* `Pandasai.ipynb` [WIP] For an example on how to use PandasAI (requires Starcoder model to be deployed)

In order to setup the connection to our local LLM Server we need to specify the base URL and an empty OpenAI token. In my case, I have `LLM_API_SERVER_BASE` and `LLM_LOADED` Environment variables set.

```python
import openai
import os

openai.api_key = "EMPTY"
openai.api_base = os.environ["LLM_API_SERVER_BASE"]
model = os.environ["LLM_LOADED"]
```
Once all these are set, we can then make a call to our REST API

```python
response = openai.Completion.create(
                        model=model,
                        prompt="Sunda Kelapa is",
                        max_tokens=500,
                        temperature=0.7,
                        stop="</s>")
print("Response result:", response)
```

We will then get the response in JSON format as what we will normally get using OpenAI API

```json
Response result: {
  "id": "cmpl-048810d973b4426483ea9840f6b9ca3c",
  "object": "text_completion",
  "created": 1690099835,
  "model": "./models/vicuna-7b-v1.3",
  "choices": [
    {
      "index": 0,
      "text": "a harbour district located in Central Jakarta, Indonesia. It is named after the Sundanese people who lived in the area before the arrival of the Dutch colonizers. The Sundanese people used to trade with the local Javanese and Arab traders. The harbour was an important trading center for spices, textiles, and other goods. Today, Sunda Kelapa is a popular tourist attraction due to its history and cultural significance.",
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "total_tokens": 103,
    "completion_tokens": 97
  }
}
```
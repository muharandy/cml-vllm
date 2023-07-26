from vllm import LLM, SamplingParams
import textwrap

model ="./models/vicuna-7b-v1.3"
llm = LLM(model=model,tokenizer="./models/llama-tokenizer")

INTRO_BLURB = ""
QUESTION_KEY = "User:"
RESPONSE_KEY = "Assistant:"

PROMPT_FOR_GENERATION_FORMAT = """{intro}
{question_key} {question}
{response_key}
""".format(
    intro=INTRO_BLURB,
    question_key=QUESTION_KEY,
    question="{question}",
    response_key=RESPONSE_KEY,
)

def get_prompt(human_question):
    # prompt_template=f"{human_prompt}"
    prompt_template = PROMPT_FOR_GENERATION_FORMAT.format(question=human_question)
    #print(prompt_template)
    return prompt_template

def remove_blurbs(string, substring):
    return string.replace(substring, "")
  
def parse_text(output):
    generated_text = output[0].outputs[0].text
    wrapped_text = textwrap.fill(generated_text, width=100)
    cleansed_text = remove_blurbs(wrapped_text,"<|endoftext|>")
    return cleansed_text

def generate(args):
    text = args["prompt"]
    prompt = get_prompt(text)
    sampling_params = SamplingParams(
                                    max_tokens=512,
                                    temperature=0.7,
                                    stop="</s>"
                                     )
    outputs = llm.generate([prompt], sampling_params)
    cleansed_text = parse_text(outputs)
    return cleansed_text
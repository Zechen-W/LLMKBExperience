# LLM-Knowledge-Boundary

See our paper: [Investigating the Factual Knowledge Boundary of Large Language Models with Retrieval Augmentation.](https://arxiv.org/abs/2307.11019)

## 🚀 Quick Start

1. Preprocess data and install dependencies.
    ```bash
    bash preparation.sh
    python data_preparation.py -d [nq/tq/hq]
    ```

2. Get supporting documents generated by ChatGPT (take Natural Questions dataset as an example).
    ```bash
    OPENAI_API_KEY=[your api key] \
    python run_llm.py \
        --source=data/source/nq.json \
        --usechat \
        --type=generate \
        --ra=none \
        --outfile=data/source/nq-chat.json
    ```

## 🔍 Conduct Experiments

1. Question answering.
    ```bash
    OPENAI_API_KEY=[your api key] \
    python run_llm.py \
        --source=data/source/nq-chat.json \
        --usechat \
        --type=qa \
        --ra=none \
        --outfile=data/qa/nq-none-qa.json
    ```
2. Priori judgement.
    ```bash
    OPENAI_API_KEY=[your api key] \
    python run_llm.py \
        --source=data/source/nq-chat.json \
        --usechat \
        --type=prior \
        --ra=dense \
        --outfile=data/prior/nq-dense-prior.json
    ```
3. Posteriori judgement.
    ```bash
    OPENAI_API_KEY=[your api key] \
    python run_llm.py \
        --source=data/qa/nq-none-qa.json \
        --usechat \
        --type=post \
        --ra=sparse \
        --outfile=data/post/nq-sparse-post.json
    ```

## 🌟 Acknowledgement

Please cite the following paper if you find our code helpful.

```bibtex
@article{ren2023investigating,
  title={Investigating the Factual Knowledge Boundary of Large Language Models with Retrieval Augmentation},
  author={Ren, Ruiyang and Wang, Yuhao and Qu, Yingqi and Zhao, Wayne Xin and Liu, Jing and Tian, Hao and Wu, Hua and Wen, Ji-Rong and Wang, Haifeng},
  journal={arXiv preprint arXiv:2307.11019},
  year={2023}
}
```
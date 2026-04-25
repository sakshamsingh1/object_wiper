<h1 align="center">
  Object-WIPER <img src="assets/logo.png" alt="logo" height="32"> :  Training-Free Object and Associated Effect Removal in Videos
</h1>

<div align="center">
  <div class="is-size-5 publication-authors">
    <span class="author-block">
      <a href="https://sakshamsingh1.github.io/">Saksham Singh Kushwaha</a><sup>1,*</sup>,&nbsp;&nbsp;</span>
    <span class="author-block">
      <a href="https://sayannag.github.io/">Sayan Nag</a><sup>2</sup>,&nbsp;&nbsp;</span>
    <span class="author-block">
      <a href="https://www.yapengtian.com/">Yapeng Tian</a><sup>1</sup>,&nbsp;&nbsp;</span> 
    <span class="author-block">
      <a href="https://kuldeepkulkarni.github.io/">Kuldeep Kulkarni</a><sup>2</sup></span>
  </div>

  <div class="is-size-5 publication-authors">
    <span class="author-block"><sup>1</sup>The University of Texas at Dallas,&nbsp;&nbsp;</span>
    <span class="author-block"><sup>2</sup>Adobe Research</span>
  </div>

  <div class="is-size-6 publication-authors">
    <span class="author-block">(* Work done during an internship at Adobe)</span>
  </div>
</div>

<p align="center">
  <a href="https://arxiv.org/pdf/2601.06391"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0iI2ZmZiI+PHBhdGggZD0iTTQgMWg1bDQgNHYxMGExIDEgMCAwIDEtMSAxSDRhMSAxIDAgMCAxLTEtMVYyYTEgMSAwIDAgMSAxLTFabTQgMXYzaDMiLz48cGF0aCBkPSJNNSA3aDZ2MUg1Wm0wIDJoNnYxSDVabTAgMmg0djFINVoiLz48L3N2Zz4=&logoColor=white"></a>
  <a href="https://sakshamsingh1.github.io/object_wiper_webpage/"><img alt="Demo Page" src="https://img.shields.io/badge/Website-Demo%20Page-yellow"></a>
  <a href="https://huggingface.co/datasets/sakshamsingh1/WIPER-bench"><img alt="Huggingface Data" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Data-brightgreen"></a>
</p>

---
**Object-WIPER** is a training-free method for removing objects and their associated effects by leveraging video priors. We also introduce a real-world associated-effect benchmark and **TokSim**, a metric for evaluating object removal quality.


## 🗞️ News: 
* **Toksim:** A new evaluation object removal metric.
* **WIPER-Bench:** A new real-world object removal benchmark. ([Link](https://huggingface.co/datasets/sakshamsingh1/WIPER-bench))


## 🛠️ Installation

```bash
conda create -n toksim python=3.11 -y
conda activate toksim

# You may have to change the versions
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

conda install -c conda-forge ffmpeg -y
pip install av transformers pillow
```

## ⚖️ TokSim

  <p align="center">
    <img src="./assets/toksim_intro.png" alt="firstpage" style="width:80%;" />
  </p>

  Unlike other metrics, Toksim scores very high only when the object is fully removed and progressively becomes lower as the object removal reduces.

<details style="margin-top: -5px;">
  <summary><strong>More about Toksim</strong></summary>

  <p align="center">
    <img src="./assets/tok_sim.png" alt="firstpage" style="width:80%;" />
  </p>

</details>

```bash
cd toksim
python sample_run.py
```
<h3 align="center">Examples</h3>

| Input Video | Prediction 1 | Prediction 2 |
| :---: | :---: | :---: |
| <img src="./assets/gifs/sphere_1_1_gt_overlay.gif" width="260" controls></img> <br> TokSim score: | <img src="./assets/gifs/sphere_1_1_pred1.gif" width="260" controls></img><br> 33.19 | <img src="./assets/gifs/sphere_1_1_pred2.gif" width="260" controls></img><br> 22.37 |
| <img src="./assets/gifs/zebra_1_1_gt_overlay.gif" width="260" controls></img> <br>TokSim score: | <img src="./assets/gifs/zebra_1_1_pred1.gif" width="260" controls></img><br> 22.54 | <img src="./assets/gifs/zebra_1_1_pred2.gif" width="260" controls></img><br> 16.82 |

## 🤗 Citation
```
@article{kushwaha2026object,
  title={Object-WIPER: Training-Free Object and Associated Effect Removal in Videos},
  author={Kushwaha, Saksham Singh and Nag, Sayan and Tian, Yapeng and Kulkarni, Kuldeep},
  journal={arXiv preprint arXiv:2601.06391},
  year={2026}
}
```

## 📧 Contact

If you have any questions, suggestions, or run into issues, please open an issue or contact [sxk230060@utdallas.edu](mailto:sxk230060@utdallas.edu).

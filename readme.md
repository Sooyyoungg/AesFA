# AesFA: An Aesthetic Feature-Aware Arbitrary Neural Style Transfer (AAAI 2024)
Official Pytorch code for "AesFA: An Aesthetic Feature-Aware Arbitrary Neural Style Transfer" <br/>

- Project page: [The Official Website for AesFA.](https://aesfa-nst.github.io/AesFA/)
- arXiv preprint: <https://arxiv.org/abs/2312.05928>

<u>First co-authors</u>
- Joonwoo Kwon (joonkwon96@gmail.com, **pioneers@snu.ac.kr**)<br/>
- Sooyoung Kim (sooyyoungg513@gmail.com, **rlatndud0513@snu.ac.kr**) <br/>
If one of us doesn't reply, please contact the other :)

## Introduction
![Figure1](https://github.com/Sooyyoungg/AesFA/assets/43199011/e9eca171-3bc6-49fc-9677-75020c2d596d)
![fig_eiffel](https://github.com/Sooyyoungg/AesFA/assets/43199011/d50e5142-1af3-4f3b-aeb7-2430c2aa7446)
Neural style transfer (NST) has evolved significantly in recent years. Yet, despite its rapid progress and advancement, exist- ing NST methods either struggle to transfer aesthetic information from a style effectively or suffer from high computa- tional costs and inefficiencies in feature disentanglement due to using pre-trained models. This work proposes a lightweight but effective model, AesFA—Aesthetic Feature-Aware NST. The primary idea is to decompose the image via its frequencies to better disentangle aesthetic styles from the reference image while training the entire model in an end-to-end manner to exclude pre-trained models at inference completely. To improve the network’s ability to extract more distinct representations and further enhance the stylization quality, this work introduces a new aesthetic feature: contrastive loss. Ex- tensive experiments and ablations show the approach not only outperforms recent NST methods in terms of stylization quality, but it also achieves faster inference.


## Environment:
- python 3.7
- pytorch 1.13.1

## Getting Started:
**Clone this repo:**
```
git clone https://github.com/Sooyyoungg/AesFA
cd AesFA
```

**Test:**


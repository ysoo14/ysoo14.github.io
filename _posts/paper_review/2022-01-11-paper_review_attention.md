---
title: 1.Neural Machine Translation by Jointly Learning to Align and Translate (Paper Review)
layout: single
author_profile: true
comments: true
share: true
related: true
popular: true
categories:
- Paper Review
toc: true
toc_sticky: true
toc_label: 목차
description: 머신러닝의 정의 및 기본 배경지식을 소개하는 글
article_tag1: attention
article_tag2: 어텐션
meta_keywords: attention, 어텐션, 논문리뷰
last_modified_at: 2022-01-11T00:00:00+08:00
---
이번 포스트에서 다룰 논문은 ICLR 2015에 개제된 Dzmitry Bahdanau, KyungHyun Cho, Yoshua Bengio 저자의 **Neural Machine Translation by Jointly Learning to Align and Translate** 입니다. 지금은 흔하게 사용되고 있는 어텐션 (attention)이란 개념을 최초로 제안한 논문입니다. 제목만 보고 Transformer 모델을 제안한 'Attention is all you need'에서 attention을 제안했다고 오해하기 쉬우니 주의 바랍니다. 'Transformer 시대' 라는 말이 나올 정도로 거의 모든 분야에서 Transformer 모델이 활용되고 있고 좋은 성능을 보이고 있는데요. Transformer 모델의 근간이 되는 attention 개념을 제안한 논문이기 때문에 'Attention is all you need' 논문을 읽기 전에 꼭 읽어보시는 것을 추천드립니다.

## Introduction & Background

우리가 자주 사용하고 있는 구글 번역기 혹은 파파고는 모두 Neural Network로 구현되었습니다. 일명 **NMT(Neural Machine Translation)** 이라고 하는데요. NMT가 도입되기 전에는 대부분 통계 모델 기반인 **SMT(Statical Machine Translation)** 오랜 기간동안 주를 이뤘습니다.

<p align="center" style="color:gray; font-size:0.5em">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/paper_review/22.01.11/2.png" style="padding: 0;margin:0;">
</p>

SMT는 하지만, 큰 단점을 가지고 있었습니다. 하나의 모델이긴 하지만 3 개의 세부 모델 (translation model, language model, alignment model)로 구성되기 때문에 어느 한 모델에서 수정이 생겨도 모든 모델에 영향을 주기 때문에 유지보수가 힘들고 비용이 많이 듭니다.

<p align="center" style="color:gray; font-size:0.5em">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/paper_review/22.01.11/3.png" style="padding: 0;margin:0;">
  출처: "https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture08-nmt.pdf"
</p>
<br>
위 그림에서, 왼쪽이 확률 기반의 모델을 의미하는 데 이는 베이즈 룰에 의해서 오른쪽과 같이 표현이 가능합니다. 오른쪽 P(x\|y)는 translation model을 의미하고 P(y)는 language model을 의미합니다. 따라서, 확률 기반 모델은 translation model, language model로 세분화 될 수 있습니다. 또한, 번역하고자 하는 source text의 어느 단어 혹은 phrase가 Target text의 어는 component와 매칭 (align)이 되는지도 모든 경우의 수를 따져봐야합니다. 이를 담당하는 것이 alignment model입니다. 그래서 총 3 개의 세부 모델 (translation model, language model, alignment model)이 필요한 것이죠.

<p align="center" style="color:gray; font-size:0.5em">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/paper_review/22.01.11/4.png" style="padding: 0;margin:0;"><br>
  출처: "https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture08-nmt.pdf"
</p>

하지만 위의 사진처럼 모든 alignment의 경우의 수는 굉장히 많죠. 하지만 번역 모델의 이렇다할 경쟁상대가 없었기 때문에 오랜 기간동안 SMT가 주로 활용된 것입니다. 그러던 중에, [Seq2Seq](https://arxiv.org/pdf/1409.3215.pdf) 모델이 등장하면서 주류는 SMT에서 NMT로 넘어가게 되었습니다.

SMT와 다르게 NMT는 유지보수가 굉장히 용이하죠. 데이터만 있으면 모델이 알아서 훈련을 하기 때문에 세 가지 세부 모델로 구성된 SMT에 비해 사람이 손수 관리해야하는 부분은 많지 않은 것입니다. 게다가 Seq2Seq 모델의 등장으로 더 높은 성능까지 보여주게 되었죠. 하지만 Seq2Seq 모델에도 한계점이 존재했는데요. 그것은 바로 아래 사진처럼 Bottleneck 문제가 발생한다는 것입니다. Seq2Seq 모델의 Encoder는 Source text의 정보를 압축해서 하나의 context vector로 표현을 하는데요, text의 길이가 길어지면서 한정된 크기의 vector로 모든 정보를 담아내기 힘들다는 것이죠.

<p align="center" style="color:gray; font-size:0.5em">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/paper_review/22.01.11/6.png" style="padding: 0;margin:0;"><br>
  출처: "https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/slides/cs224n-2019-lecture08-nmt.pdf"
</p>
<br>
이러한 문제를 해결하고자 attention이 고안되었고 이 논문을 통해 제안되었습니다. 자, 이제 함꼐 살펴보시져.

## Learning To Align and Translate

## Experiments & Result


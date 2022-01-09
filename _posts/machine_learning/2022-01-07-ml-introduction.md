---
title: 머신러닝(Machine Learning) 이란? - 정의, 유형, 기본 지식
layout: single
author_profile: true
comments: true
share: true
related: true
popular: true
categories:
- Machine Learning
toc: true
toc_sticky: true
toc_label: 목차
description: 머신러닝의 정의 및 기본 배경지식을 소개하는 글
article_tag1: machine learning
article_tag2: 머신러닝
meta_keywords: 머신러닝이란, 머신러닝, machine learning
last_modified_at: 2022-01-07T00:00:00+08:00
---
본 포스트는 머신러닝에 대한 정의와 머신러닝을 이해하는 데에 필요한 기본 지식에 대해 소개하는 글이고,
Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow 2nd Edition을 참고하였습니다.

## 1. 머신러닝(Machine Learning) 이란?

 최근 들어, 가장 핫하고 주목받는 기술이 무엇이냐고 물어봤을 때, 대부분의 사람들이 아마도 인공지능 A.I. (Artificial Intelligence)라고 대답할 것이다.
기본적으로 인공지능은 사람이 하는 일이나 어떤 메커니즘을 기계가 대신 할 수 있는 기술을 의미하는데, 머신러닝(기계학습)은 인공지능을 구현하는 하나의 방법이다. 말 그대로, 기계를 사람처럼 행동하도록 학습 시키는 것이다.
머신러닝이라고 했을 때, 대부분의 보통 사람들은 판타지 영화에 나오는 굉장히 대단하고 화려한 기술이라고 생각하지만 자동 스팸 필터링, OCR(Optical Character Recognition), SNS 얼굴인식 등 이미 다양한 서비스에 많이 활용되고 있고 쉽게 접할 수 있다.


**그렇다면, 머신러닝은 무엇일까요?** 머신러닝은 앞서 말한 것 처럼, 인공지능을 구현하는 하나의 방법으로 기존 프로그래밍 처럼 시스템에 특정 규칙을 미리 정의하는 것이 아닌 데이터를 통해 그 규칙을 학습하는 것을 의미한다. 쉽게 말해서 시스템으로 하여금 기존 데이터를 보고 어떤 의사결정을 내릴 수 있도록 특정 규칙을 만들어 가는 것이라고 생각하면 된다.

<p align="center" style="color:gray">
  <img src="{{ site.url }}{{ site.baseurl }}/assets/images/22.01.07/ml/ml-intro1.png" style="padding: 0;margin:0;">
  출처: "https://bi.snu.ac.kr/~scai/Courses/ML2017/ch1.pdf"
</p>
